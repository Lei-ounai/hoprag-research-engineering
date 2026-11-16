"""本地人工证据核验工作台。仅使用 Python 标准库。"""

import argparse
import codecs
import csv
import hashlib
import io
import json
import os
import re
import socket
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from human_review_packets import CASE_IDS, load_cases


HERE = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
INPUT_NAMES = ("annotation_packets.md", "annotation_guide.md", "annotation_template.csv")
PROGRESS_NAME = "human_review_progress.json"
EXPORT_NAME = "HUMAN_ANNOTATION_350.csv"
ANNOTATOR_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
DEVICE_NAME = re.compile(r"con|prn|aux|nul|com[0-9]|lpt[0-9]")
CITABLE = {"DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTRADICTION"}
UI_FILES = {
    "/": ("human_review_workbench.html", "text/html; charset=utf-8"),
    "/human_review_workbench.css": ("human_review_workbench.css", "text/css; charset=utf-8"),
    "/human_review_workbench.js": ("human_review_workbench.js", "text/javascript; charset=utf-8"),
}

LABELS = {
    "evidence_support": {
        "DIRECT_SUPPORT": "直接支持至少一个完整的必要命题",
        "PARTIAL_SUPPORT": "只支持同一命题的一部分",
        "NO_SUPPORT": "没有支持题面命题",
        "CONTRADICTION": "明确反驳题面命题",
        "UNCERTAIN": "可见材料不足以判定",
    },
    "entity_role": {
        "MATCH": "实体、角色和方向相符",
        "MISMATCH": "实体、角色或方向错配",
        "NOT_APPLICABLE": "题面无相关角色要求",
        "UNCERTAIN": "角色或指代不明",
    },
    "relation_support": {
        "SUPPORTED": "目标关系得到支持",
        "NEGATED": "目标关系被明确否定",
        "HEDGED": "仅有推测、涉嫌或保留",
        "UNRELATED": "所述关系与目标不同",
        "UNCERTAIN": "关系无法确定",
    },
    "temporal_scope": {
        "CONSISTENT": "时间范围相符",
        "CONFLICT": "时间范围冲突",
        "UNKNOWN": "时间或作用域无法锚定",
        "NOT_APPLICABLE": "题面无时间约束",
    },
    "temporal_kind": {
        "EVENT": "事件时间",
        "PUBLICATION": "报道发布时间",
        "TENURE": "任职时间",
        "FOUNDING": "成立时间",
        "MULTIPLE": "涉及多类时间",
        "UNKNOWN": "无法确定时间种类",
        "NOT_APPLICABLE": "题面无时间种类要求",
    },
    "evidence_novelty": {
        "NOVEL": "可见前十未给出该事实",
        "REDUNDANT": "与可见前十的事实重复",
        "PARTIAL_NOVELTY": "兼有重合和新增",
        "UNCERTAIN": "受节选范围限制而无法确定",
    },
    "replacement_suitability": {
        "REPLACE": "换入挑战者有清楚的额外价值",
        "KEEP_INCUMBENT": "保留指定比较对象",
        "EITHER": "两者的问题证据价值相近",
        "UNCERTAIN": "无法确定替换损益",
    },
}
COLUMNS = ("case_id", *LABELS, "reason", "witness_ref", "witness_quote")
VALUE_FIELDS = COLUMNS[1:]


def input_hashes(directory):
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in INPUT_NAMES
    }


def load_fields(directory):
    with (directory / "annotation_template.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, strict=True))
    if not rows or tuple(rows[0]) != COLUMNS:
        raise ValueError("正式模板的列名或顺序与七维标注定义不一致。")
    if (any(len(row) != len(COLUMNS) or any(row[1:]) for row in rows[1:])
            or tuple(row[0] for row in rows[1:]) != CASE_IDS):
        raise ValueError("正式模板应含 350 个编号和空白标注字段。")

    guide = (directory / "annotation_guide.md").read_text(encoding="utf-8-sig")
    dimensions = re.findall(r"^\| (\w+) \| ([A-Z_ ·]+) \| ([^\n|]+) \|$", guide, re.M)
    if tuple(name for name, _, _ in dimensions) != tuple(LABELS):
        raise ValueError("标注指南的七个字段与正式模板不一致。")
    fields = []
    for name, allowed, description in dimensions:
        values = allowed.split(" · ")
        if set(values) != set(LABELS[name]) or len(values) != len(LABELS[name]):
            raise ValueError(f"标注指南的 {name} 枚举已改变，请核对工作台定义。")
        fields.append({
            "name": name,
            "description": description,
            "options": [{"value": v, "label": LABELS[name][v]} for v in values],
        })
    return fields


def safe_annotator(value):
    if (not isinstance(value, str) or not ANNOTATOR_ID.fullmatch(value)
            or DEVICE_NAME.fullmatch(value)):
        raise ValueError("标注者编号须为 1–40 位小写字母、数字、下划线或连字符，以字母或数字开头，并避开系统保留名。")
    return value


def atomic_write(path, content):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if temp_path.read_bytes() != content:
            raise OSError("临时文件写入校验失败。")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class Workbench:
    def __init__(self, directory, outputs):
        self.hashes = input_hashes(directory)
        self.fields = load_fields(directory)
        self.cases = load_cases(directory / "annotation_packets.md")
        if tuple(self.cases) != CASE_IDS:
            raise ValueError("案例包的 350 个案例编号缺失、重复或顺序有误。")
        self.visible = {
            case_id: {w["ref"]: w["text"] for group in case["groups"] for w in group["windows"]}
            for case_id, case in self.cases.items()
        }
        outputs = Path(os.path.abspath(outputs))
        self.outputs = outputs.parent.resolve(strict=True) / outputs.name
        self.outputs.mkdir(exist_ok=True)
        if self.outputs.resolve() != self.outputs:
            raise ValueError("标注输出目录须位于当前工作目录。")
        self.lock = threading.Lock()
        if input_hashes(directory) != self.hashes:
            raise ValueError("加载期间原始材料发生变化，请重启后重新检查。")

    def bootstrap(self):
        return {"fields": self.fields, "ids": CASE_IDS, "total": len(CASE_IDS)}

    def _require_case(self, case_id):
        if not isinstance(case_id, str) or case_id not in self.cases:
            raise ValueError("未找到该案例编号。")
        return case_id

    def _require_values(self, case_id, values):
        if not isinstance(values, dict) or set(values) != set(VALUE_FIELDS):
            raise ValueError("请填写七个标注字段、理由、见证编号及引文。")
        if any(not isinstance(values[field], str) for field in VALUE_FIELDS):
            raise ValueError("标注字段须为文本。")
        for field, allowed in LABELS.items():
            if values[field] not in allowed:
                raise ValueError(f"{field} 请选择规范中的合法标签。")
        if not values["reason"].strip():
            raise ValueError("请填写人工判断理由。")
        ref, quote = values["witness_ref"], values["witness_quote"]
        c_required = values["evidence_support"] in CITABLE
        if ref == "NONE":
            if c_required or quote:
                raise ValueError("NONE 仅适用于无支持或不确定且确实无可引用原文的情况。请说明原因，将引文留空。")
        elif ref not in self.visible[case_id]:
            raise ValueError("见证编号须为本案存在的 C、I 或 B 可见片段编号，或按指南填写 NONE。")
        elif c_required and not ref.startswith("C"):
            raise ValueError("直接支持、部分支持或反证的见证须引自挑战者 C 段。")
        elif not quote.strip() or quote not in self.visible[case_id][ref]:
            raise ValueError("引文与该编号的评分可见原文不完全一致，请回到原文核对并复制连续文本。")

    def _profile(self, annotator):
        profile = self.outputs / safe_annotator(annotator)
        if profile.resolve() != profile:
            raise ValueError("该标注者的目录无法安全使用。")
        return profile

    @staticmethod
    def _file(profile, name):
        path = profile / name
        if path.resolve() != path:
            raise ValueError("标注输出文件无法安全使用。")
        return path

    def _read_state(self, profile):
        path = self._file(profile, PROGRESS_NAME)
        if not path.is_file():
            raise ValueError("请先进入该标注者的工作台。")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(state, dict) or state.get("version") != 1
                    or state.get("annotator") != profile.name
                    or state.get("inputs") != self.hashes):
                raise ValueError("标注者或原始材料版本与保存记录不一致。")
            self._require_case(state["last_case"])
            for case_id, values in state["annotations"].items():
                self._require_values(self._require_case(case_id), values)
        except (AttributeError, KeyError, ValueError) as error:
            raise ValueError(f"无法读取或验证标注进度。请保留文件并检查错误：{error}") from error
        return state

    def _write_state(self, profile, state):
        path = self._file(profile, PROGRESS_NAME)
        content = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        atomic_write(path, content)
        saved = self._read_state(profile)
        if saved != state:
            raise OSError("重新读取的标注进度与保存内容不一致。")
        return saved

    @staticmethod
    def _summary(state):
        return {
            "annotator": state["annotator"],
            "last_case": state["last_case"],
            "completed": sorted(state["annotations"]),
        }

    def session(self, annotator):
        profile = self._profile(annotator)
        with self.lock:
            profile.mkdir(exist_ok=True)
            path = self._file(profile, PROGRESS_NAME)
            if path.exists():
                state = self._read_state(profile)
            else:
                state = self._write_state(profile, {
                    "version": 1,
                    "annotator": annotator,
                    "inputs": self.hashes,
                    "last_case": CASE_IDS[0],
                    "annotations": {},
                })
            return self._summary(state)

    def get_case(self, annotator, case_id):
        case_id = self._require_case(case_id)
        profile = self._profile(annotator)
        with self.lock:
            state = self._read_state(profile)
        return {"case": self.cases[case_id], "annotation": state["annotations"].get(case_id)}

    def save(self, annotator, case_id, values):
        case_id = self._require_case(case_id)
        self._require_values(case_id, values)
        profile = self._profile(annotator)
        with self.lock:
            state = self._read_state(profile)
            state["last_case"] = case_id
            state["annotations"][case_id] = values
            state = self._write_state(profile, state)
            return {**self._summary(state), "annotation": state["annotations"][case_id]}

    def position(self, annotator, case_id):
        case_id = self._require_case(case_id)
        profile = self._profile(annotator)
        with self.lock:
            state = self._read_state(profile)
            if state["last_case"] != case_id:
                state["last_case"] = case_id
                self._write_state(profile, state)
            return {"last_case": case_id}

    def export(self, annotator):
        profile = self._profile(annotator)
        with self.lock:
            annotations = self._read_state(profile)["annotations"]
            rows = [COLUMNS, *(
                (case_id, *(annotations.get(case_id, {}).get(field, "") for field in VALUE_FIELDS))
                for case_id in CASE_IDS
            )]
            stream = io.StringIO(newline="")
            csv.writer(stream).writerows(rows)
            content = stream.getvalue().encode("utf-8-sig")
            path = self._file(profile, EXPORT_NAME)
            atomic_write(path, content)
            saved = path.read_bytes()
            with path.open(encoding="utf-8-sig", newline="") as handle:
                if not saved.startswith(codecs.BOM_UTF8) or list(map(tuple, csv.reader(handle, strict=True))) != rows:
                    raise OSError("导出的 CSV 重新解析校验失败。")
            return saved


def lock_outputs(outputs):
    """Keep a single server responsible for writing this output directory."""
    path = Workbench._file(outputs, ".workbench.lock")
    handle = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ValueError(
            "标注输出目录正在被另一个工作台使用。请使用现有工作台，或用 --output-dir 指定另一个目录。"
        ) from error
    return handle


class WorkbenchServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = False

    def __init__(self, workbench, port):
        self.workbench = workbench
        self.assets = {
            route: ((HERE / filename).read_bytes(), content_type)
            for route, (filename, content_type) in UI_FILES.items()
        }
        self._output_lock = lock_outputs(workbench.outputs)
        try:
            super().__init__((HOST, port), WorkbenchHandler)
        except BaseException:
            self._output_lock.close()
            raise

    def server_close(self):
        try:
            super().server_close()
        finally:
            self._output_lock.close()

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class WorkbenchHandler(BaseHTTPRequestHandler):
    timeout = 15

    def _reply(self, content, content_type, status=HTTPStatus.OK, *, download=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(content)

    def _json(self, value, status=HTTPStatus.OK):
        self._reply(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8", status)

    def _local(self):
        host = f"{HOST}:{self.server.server_port}"
        if self.headers.get("Host") != host or self.headers.get("Origin") not in (None, f"http://{host}"):
            self._json({"error": "请从本机工作台地址访问。"}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def _body(self, expected_fields):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("工作台接口要求 JSON。")
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= 256 * 1024:
            raise ValueError("请求内容为空或超过允许大小。")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict) or set(data) != set(expected_fields):
            raise ValueError("请求字段不完整。")
        return data

    def do_GET(self):
        if not self._local():
            return
        try:
            target = urlsplit(self.path)
            if target.path in self.server.assets:
                self._reply(*self.server.assets[target.path])
            elif target.path == "/api/bootstrap":
                self._json(self.server.workbench.bootstrap())
            elif target.path == "/api/case":
                args = parse_qs(target.query, keep_blank_values=True)
                if set(args) != {"id", "annotator"} or any(len(v) != 1 for v in args.values()):
                    raise ValueError("请指定标注者和案例编号。")
                self._json(self.server.workbench.get_case(args["annotator"][0], args["id"][0]))
            else:
                self._json({"error": "该工作台地址不存在。"}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except OSError as error:
            self._json({"error": f"读取失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        if not self._local():
            return
        workbench = self.server.workbench
        try:
            if self.path == "/api/session":
                self._json(workbench.session(**self._body(("annotator",))))
            elif self.path == "/api/save":
                self._json(workbench.save(**self._body(("annotator", "case_id", "values"))))
            elif self.path == "/api/position":
                self._json(workbench.position(**self._body(("annotator", "case_id"))))
            elif self.path == "/api/export":
                data = workbench.export(**self._body(("annotator",)))
                self._reply(data, "text/csv; charset=utf-8", download=EXPORT_NAME)
            else:
                self._json({"error": "该工作台地址不存在。"}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except OSError as error:
            self._json({"error": f"文件操作失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        # 避免在终端逐条记录包含标注者编号的浏览请求。
        pass


def main():
    parser = argparse.ArgumentParser(description="启动 HopRAG 人工证据核验工作台。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE,
        help="包含 annotation_packets.md、annotation_guide.md 和 annotation_template.csv 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="标注进度与 CSV 输出目录；默认写入数据目录下的 human_review_outputs。",
    )
    parser.add_argument("--port", type=int, default=PORT, help=f"本机 HTTP 端口（默认 {PORT}）。")
    args = parser.parse_args()

    if not 0 <= args.port <= 65535:
        parser.error("--port 必须为 0–65535 范围内的整数。")
    try:
        data_dir = args.data_dir.expanduser().resolve(strict=True)
        if not data_dir.is_dir():
            parser.error("--data-dir 必须指向目录。")
        output_dir = (
            args.output_dir.expanduser()
            if args.output_dir is not None
            else data_dir / "human_review_outputs"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        workbench = Workbench(data_dir, output_dir)
        server = WorkbenchServer(workbench, args.port)
    except FileNotFoundError as error:
        parser.error(f"找不到启动所需的数据文件或目录：{error.filename}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    with server:
        print(f"HopRAG 人工证据核验工作台：已加载 {len(workbench.cases)} 案。")
        print(f"请访问 http://{HOST}:{server.server_port}/。按 Ctrl+C 停止。")
        print(f"标注进度保存目录：{workbench.outputs}")
        for name, digest in workbench.hashes.items():
            print(f"{name}  SHA-256 {digest}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n工作台已停止。")


if __name__ == "__main__":
    main()

