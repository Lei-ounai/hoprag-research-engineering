import argparse
import json
import sys
from dataclasses import asdict

from .retrieval import Retriever, load_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="检索并显示问题对应的文档证据链。")
    parser.add_argument("corpus", help="UTF-8 JSONL 语料路径")
    parser.add_argument("question", help="自然语言问题")
    parser.add_argument("--max-hops", type=int, default=1, help="每条证据链的最大链接数（默认 1）")
    parser.add_argument("--top-k", type=int, default=3, help="最多返回的证据链数（默认 3）")
    parser.add_argument("--json", action="store_true", help="输出 UTF-8 JSON")
    args = parser.parse_args()

    try:
        results = Retriever(load_corpus(args.corpus)).search(
            args.question, max_hops=args.max_hops, top_k=args.top_k
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"错误：{exc}\n")

    if args.json:
        print(json.dumps([asdict(chain) for chain in results], ensure_ascii=False, indent=2))
    elif not results:
        print("未找到与问题匹配的文档。")
    else:
        for rank, chain in enumerate(results, start=1):
            print(f"[{rank}] {' → '.join(doc.title for doc in chain.documents)}（得分 {chain.score:.3f}）")
            for position, doc in enumerate(chain.documents, start=1):
                print(f"  {position}. {doc.title} [{doc.id}]")
                print(f"     {doc.text}")
                if doc.source:
                    print(f"     来源 {doc.source}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
