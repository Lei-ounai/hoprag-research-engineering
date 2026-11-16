"""Read only the score-visible material and optional context in the blind case packet."""

import re
from pathlib import Path


CASE_IDS = tuple(f"evb-{number:04d}" for number in range(1, 351))
CASE_HEADING = re.compile(r"## (evb-\d{4})")
WINDOW_HEADING = re.compile(r"\*\*([CIB]\d+) · (.+)\*\*")
TEXT_FENCE = re.compile(r"(~{3,})text")
RAW_HEADING = "<details><summary>完整原始 Chunk，仅供定位模型窗外的候选事实</summary>"
GROUP_HEADINGS = {
    "C": re.compile(r"### (挑战者)"),
    "I": re.compile(r"### (比较对象（原第 \d+ 位）)"),
    "B": re.compile(r"### (当前十篇文档的相关片段)"),
}


class PacketFormatError(ValueError):
    """The case packet is missing a required section or contains invalid markup."""


class PacketReader:
    def __init__(self, text: str):
        self.lines = text.split("\n")
        self.index = 0
        self.case_id = ""

    def fail(self, message: str):
        location = f"{self.case_id}，" if self.case_id else ""
        raise PacketFormatError(f"{location}盲审包第 {self.index + 1} 行：{message}")

    def current(self) -> str:
        return self.lines[self.index] if self.index < len(self.lines) else ""

    def skip_blanks(self):
        while self.index < len(self.lines) and not self.current().strip():
            self.index += 1

    def match(self, pattern: re.Pattern, message: str) -> re.Match:
        found = pattern.fullmatch(self.current())
        if found is None:
            self.fail(message)
        self.index += 1
        return found

    def source(self, prefix: str = "") -> tuple[str, str]:
        line = self.current()
        if prefix and not line.startswith(prefix):
            self.fail("缺少媒体信息")
        source, separator, published = line[len(prefix):].rpartition(" · 发布 ")
        if not separator or not source.strip() or not published.strip():
            self.fail("缺少来源或发布日期")
        self.index += 1
        self.skip_blanks()
        return source, published

    def document(self) -> dict:
        name = self.match(re.compile(r"\*\*(.+)\*\*"), "缺少文档标题").group(1)
        self.skip_blanks()
        source, published = self.source("媒体 ")
        url = self.match(re.compile(r"原报道 (.+)"), "缺少原报道地址").group(1)
        self.skip_blanks()
        return {"title": name, "source": source, "published": published, "url": url}

    def fenced_text(self) -> tuple[str, int]:
        fence = self.match(TEXT_FENCE, "缺少文本窗起始围栏").group(1)
        start = self.index
        while self.index < len(self.lines) and self.current() != fence:
            self.index += 1
        if self.index == len(self.lines):
            self.fail("文本窗围栏未闭合")
        text = "\n".join(self.lines[start:self.index])
        if not text.strip():
            self.fail("文本窗为空")
        self.index += 1
        self.skip_blanks()
        return text, start + 1

    def window(self, header: re.Match, doc: dict | None) -> dict:
        ref, title = header.groups()
        self.index += 1
        self.skip_blanks()
        meta = ""
        if doc is None:
            source, published = self.source()
        else:
            source, published = doc["source"], doc["published"]
            if self.current().startswith("日期字面 "):
                meta = self.current()
                self.index += 1
                self.skip_blanks()
        text, line = self.fenced_text()
        window = {
            "ref": ref, "title": title, "source": source, "published": published,
            "meta": meta, "text": text, "line": line,
        }
        if doc is not None and self.current() == RAW_HEADING:
            self.index += 1
            self.skip_blanks()
            raw, raw_line = self.fenced_text()
            if self.current() != "</details>":
                self.fail("折叠原始 Chunk 标记未闭合")
            if not raw.startswith(text):
                self.fail("折叠原始 Chunk 与评分可见文本边界不一致")
            window["raw"] = raw
            window["raw_line"] = raw_line
            self.index += 1
            self.skip_blanks()
        if doc is not None and (title == "模型可见部分，已截断") != ("raw" in window):
            self.fail("评分窗的截断说明与原始 Chunk 不一致")
        return window

    def group(self, key: str) -> dict:
        title = self.match(GROUP_HEADINGS[key], f"缺少 {key} 组标题").group(1)
        self.skip_blanks()
        doc = self.document() if key in {"C", "I"} else None
        windows = []
        while header := WINDOW_HEADING.fullmatch(self.current()):
            expected = f"{key}{len(windows) + 1}"
            if header.group(1) != expected:
                self.fail(f"预期见证编号 {expected}，得到 {header.group(1)}")
            windows.append(self.window(header, doc))
        if len(windows) not in ({10} if key == "B" else {1, 2, 3}):
            self.fail(f"{key} 组见证数量异常（{len(windows)}）")
        group = {"key": key, "title": title, "windows": windows}
        if doc is not None:
            group["document"] = doc
        return group

    def cases(self) -> dict[str, dict]:
        while self.index < len(self.lines) and not self.current().startswith("## "):
            self.index += 1
        result = {}
        for case_id in CASE_IDS:
            self.case_id = case_id
            parsed_id = self.match(CASE_HEADING, "缺少案例标题").group(1)
            if parsed_id != case_id:
                self.fail(f"案例编号顺序不完整，得到 {parsed_id}")
            self.skip_blanks()
            start = self.index
            while self.index < len(self.lines) and not self.current().startswith(("## ", "### ")):
                self.index += 1
            question = "\n".join(self.lines[start:self.index]).rstrip("\n")
            if not question.strip():
                self.fail("原始问题为空")
            result[case_id] = {
                "id": case_id,
                "question": question,
                "groups": [self.group(key) for key in GROUP_HEADINGS],
            }
        if self.index != len(self.lines):
            self.fail("第 350 案之后出现额外内容")
        return result


def load_cases(path: Path) -> dict[str, dict]:
    """Parse the original case order and retain exact visible and folded text."""
    return PacketReader(path.read_text(encoding="utf-8-sig")).cases()
