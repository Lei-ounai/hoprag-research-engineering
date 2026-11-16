# HopRAG 检索研究与工程

从仓库根目录运行下列命令。项目包含静态链接多跳检索原型、三种检索基线、重排与证据集合实验，以及离线人工证据核验工作台。

## Python 环境

项目使用本机 Python 3.14.8。已有 `.venv` 时直接使用其中的解释器。新建同样的环境时，可在项目根目录运行：

```powershell
py -3.14 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## MultiHop-RAG 基线评测

将官方 `corpus.json` 和 `MultiHopRAG.json` 原样置于 `data/raw/multihop_rag/`。字段与数据审计可独立查看。评测要求 Ollama 在 `http://127.0.0.1:11434` 运行，且本机已安装 `bge-m3`。下载地址、数据许可和论文见 [数据归属说明](data/raw/multihop_rag/ATTRIBUTION.md)。

```powershell
& .\.venv\Scripts\python.exe -m hoprag --audit-only
& .\.venv\Scripts\python.exe -m hoprag
```

默认按 20 条、100 条、全量三个阶段运行。两个子样本按问题类型及证据条数分层，随机种子为 42。也可通过 `--sizes 20` 只跑冒烟阶段，或通过 `--output-dir results/another_run` 保留独立的结果目录。BM25、Dense 和 RRF 在每个阶段使用相同的文档及 Query ID。首轮需要为全语料建向量，即使只评测 20 条也需要先完成这一步。SQLite 在 `data/cache/` 按批提交，重跑时可接着计算未缓存的文本。

默认输出位于 `results/baseline/`，包含数据审计、模型和配置、查询 ID、三种排名、分阶段指标、Chunk ID 映射和原型缺陷复现。评测结果与口径见 [BASELINE_REPORT.md](BASELINE_REPORT.md)。

## Static-Link Graph Retrieval Baseline

`multihop/` 保留原型入口和算法。输入是 `data/example.jsonl` 中的手工链接图。原型用自己的 BM25 匹配起点，再沿 `links` 扩展，返回按分数排序的证据链。`--max-hops` 计算边数，默认为 1，`--top-k` 默认为 3，`--json` 输出证据链 JSON。

```powershell
& .\.venv\Scripts\python.exe -m multihop data/example.jsonl '《物种起源》的作者出生在哪座城市？'
& .\.venv\Scripts\python.exe -m multihop data/example.jsonl '《物种起源》的作者出生的城市位于英国哪个郡？' --max-hops 2 --top-k 1
& .\.venv\Scripts\python.exe -m hoprag --graph-only
```

自有语料使用 UTF-8 JSONL，每行包含唯一 `id`、`title`、`text`，可用 `links` 指向其他文档 ID，并提供可选 `source`。链接由语料维护者标注。

## 测试

~~~powershell
$suites = @(
    'tests',
    'analysis/evidence_verification_benchmark/tests',
    'analysis/besa_v0/tests',
    'analysis/ege_v0/tests',
    'analysis/egr_v0/tests',
    'analysis/reranking/tests',
    'analysis/reranking_optimization/tests'
)
foreach ($suite in $suites) {
    & .\.venv\Scripts\python.exe -B -m unittest discover -s $suite -v
    if ($LASTEXITCODE -ne 0) { throw "测试失败：$suite" }
}
~~~

工作台测试使用临时合成盲审包和临时标注目录。依赖原始数据集或冻结实验归档的测试会在输入缺失时报告跳过。已有 Ollama 和本地 bge-m3 时，可以单独运行真实集成用例。

~~~powershell
$env:HOPRAG_OLLAMA_TEST = '1'
try {
    & .\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_baseline_retrieval.py -v
} finally {
    Remove-Item Env:HOPRAG_OLLAMA_TEST
}
~~~

## 研究实验

- [Cross-Encoder 重排](analysis/reranking/RERANKING_REPORT.md)与[低预算重排](analysis/reranking_optimization/OPTIMIZATION_REPORT.md)
- [BESA 证据集合选择](analysis/besa_v0/BESA_V0_REPORT.md)
- [EGE 候选扩展](analysis/ege_v0/EGE_V0_REPORT.md)与 [EGR 证据门控替换](analysis/egr_v0/EGR_V0_REPORT.md)

完整复现实验需要自备有权使用的语料、上游实验产物和本地模型缓存。报告中引用的逐题结果及历史中间文件保留在研究环境。模型加载收据由重排运行在本地生成。

在已有基线排名和缓存、且需要首次准备重排模型时，使用以下命令。第一条安装实验依赖，第二条下载模型。缓存已有适配版本时可跳过前两条。构建候选池后，Jina 的统一窗口开发集运行会生成后续实验使用的本地模型收据。

~~~powershell
& .\.venv\Scripts\python.exe -m pip install 'fastembed==0.9.1' 'onnxruntime==1.31.1'
& .\.venv\Scripts\python.exe -B -m analysis.reranking.prepare_models
& .\.venv\Scripts\python.exe -B -m analysis.reranking.pool
& .\.venv\Scripts\python.exe -B -m analysis.reranking.run --model jina --context unified --stages dev_5
~~~

## 人工证据核验工作台

项目还包含 350 案例的本地人工证据核验工具，支持七维标注、标注者隔离、进度恢复、原文见证校验、案例筛选与 CSV 导出。工作台只依赖 Python 标准库，详细说明见 [工作台使用说明](analysis/evidence_verification_benchmark/README_HUMAN_REVIEW_WORKBENCH.md)。

在项目根目录运行：

~~~powershell
$benchmark = '.\analysis\evidence_verification_benchmark'
& .\.venv\Scripts\python.exe -B "$benchmark\human_review_workbench.py" --data-dir $benchmark --output-dir "$benchmark\human_review_outputs"
~~~

在本地将有权使用的 350 案盲审包保存为工作台目录中的 `annotation_packets.md`，然后打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。若端口已占用，可加 `--port 8766` 使用空闲端口。同一输出目录只能由一个服务实例写入。盲审包、个人进度、标注 CSV、原始数据集与模型缓存只保存在本地。

工作台测试：

~~~powershell
& .\.venv\Scripts\python.exe -m unittest discover -s analysis\evidence_verification_benchmark\tests -v
~~~
