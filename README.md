# HopRAG — 多跳检索研究与工程

**面向复杂问题的多跳证据检索与排序优化研究**

HopRAG 是一个基于 Python 构建的多跳检索研究工程项目，重点研究 RAG 系统在复杂问题中出现的一个核心问题：

**检索系统能够找到与问题相关的文档，却无法在有限的 Top-K 预算内找齐回答问题所需的全部证据。**

项目以 MultiHop-RAG 公开数据集为基础，从最初的静态链接图多跳检索原型出发，逐步构建 BM25、Dense、RRF 混合检索基线，并进一步研究 Cross-Encoder 重排序、证据集合选择、候选扩展和证据门控。

项目不仅关注单篇文档的相关性，还重点关注多篇文档之间的证据互补性、必要证据完整覆盖率，以及检索优化过程中可能产生的证据丢失。

目前已完成主要工程实现、检索实验、正式保留集验证及本地人工证据核验工作台，并公开发布代码。

---

## 1. 项目背景与研究问题

### 1.1 研究背景

传统 RAG 系统通常采用以下流程：

1. 用户输入问题。
2. 检索器从知识库中寻找相关文档。
3. 对检索结果进行排序。
4. 将 Top-K 文档交给大语言模型。
5. 大语言模型根据检索内容生成答案。

这种方式在单事实问题中通常能够正常工作。

但在多跳问题中，一个问题可能需要多个不同来源共同提供证据。

例如：

> 《物种起源》的作者出生的城市位于英国哪个郡？

回答这个问题至少需要：

- 证据一：《物种起源》的作者是查尔斯·达尔文。
- 证据二：查尔斯·达尔文出生于什鲁斯伯里。
- 证据三：什鲁斯伯里位于什罗普郡。

检索系统即使成功找到《物种起源》和达尔文相关的文档，也不代表它已经找齐回答问题所需的证据。

如果必要证据没有全部进入最终 Top-K，后续生成模型就可能无法得到完整、可靠的答案。

### 1.2 核心研究问题

本项目围绕以下问题展开：

1. BM25 和 Dense 检索各自存在哪些优势与不足？
2. 如何利用混合检索提高必要证据的召回能力？
3. 如何避免重排序将已经找到的必要证据挤出 Top-K？
4. 如何在有限的计算预算下，提高完整证据覆盖率？
5. 如何判断新增候选文档是否真正提供了新的有效证据？
6. 如何识别实体角色错配、关系不匹配和时间作用域错误？
7. 如何区分主题相关文档与真正支持问题的证据？
8. 如何建立独立的证据核验流程，评估证据门控策略的可靠性？

项目的核心评价目标不是简单提高相关文档命中率，而是：

**在固定检索预算下，尽可能找齐回答复杂问题所需的全部必要证据。**

---

## 2. 项目技术架构

### 2.1 整体流程

```text
                  MultiHop-RAG 数据集
                          |
                          v
                    数据加载与审计
                          |
                          v
                    文档与问题建模
                          |
               +----------+----------+
               |                     |
               v                     v
           BM25 检索              Dense 检索
               |                     |
               |                BGE-M3 Embedding
               |                     |
               +----------+----------+
                          |
                          v
                   RRF 混合检索
                          |
                          v
                    候选文档池
                          |
                          v
                 Cross-Encoder 重排序
                          |
                          v
                   P5-B 低预算融合
                          |
                          v
                    Top-10 证据集
                          |
                          v
                     离线评测
                          |
             +------------+------------+
             |            |            |
             v            v            v
          证据召回      完整覆盖      无答案分析
             |            |            |
             +------------+------------+
                          |
                          v
                 后续研究实验
                          |
             +------------+------------+
             |            |            |
             v            v            v
           BESA           EGE          EGR
             |            |            |
             +------------+------------+
                          |
                          v
                人工证据核验工作台
```

### 2.2 技术栈

| 技术 | 主要用途 |
|---|---|
| Python 3.14.8 | 项目主要开发语言 |
| NumPy 2.5.4 | 向量运算与相似度计算 |
| rank-bm25 0.2.2 | BM25 检索 |
| Ollama | 本地 Embedding 模型服务 |
| BGE-M3 | 文本向量化 |
| SQLite | Embedding 与模型评分缓存 |
| FastEmbed 0.9.1 | Cross-Encoder 模型加载与推理 |
| ONNX Runtime 1.31.1 | 本地 CPU 模型推理 |
| Jina Reranker Turbo | 文档相关性重排序 |
| MiniLM | 对照重排序模型 |
| unittest | 自动化测试 |
| HTML / CSS / JavaScript | 人工证据核验工作台 |

当前实验主要在本地 CPU 环境中完成，不依赖云端模型 API。

---

## 3. 核心模块设计

### 3.1 静态链接图多跳检索原型

目录：`multihop/`

这是项目最初的多跳检索原型。

主要功能：

- 加载 JSONL 文档语料。
- 使用 BM25 寻找起始文档。
- 根据文档中人工维护的链接进行扩展。
- 支持指定最大跳数。
- 对候选证据链进行排序。
- 输出完整证据链及来源信息。

数据格式：

```json
{
  "id": "origin-of-species",
  "title": "物种起源",
  "text": "《物种起源》的作者是查尔斯·达尔文。",
  "links": ["charles-darwin"],
  "source": "example"
}
```

运行示例：

```powershell
python -m multihop data/example.jsonl "《物种起源》的作者出生在哪座城市？"

python -m multihop data/example.jsonl "《物种起源》的作者出生的城市位于英国哪个郡？" --max-hops 2 --top-k 1
```

该模块主要用于验证多跳证据链的基本机制。

需要注意：这里的链接由语料维护者人工标注，并非自动知识图谱构建系统。

### 3.2 BM25 检索

目录：`hoprag/retrievers.py`

采用 `rank-bm25` 实现 BM25Okapi。

主要流程：

1. 对文档标题和正文进行文本预处理。
2. 使用 Unicode 文本归一化。
3. 对文本进行词项提取。
4. 建立 BM25 索引。
5. 根据用户问题计算候选文档分数。
6. 返回排序结果。

主要特点：

- 不需要 Embedding 模型。
- 能够有效利用明确的关键词匹配。
- 对指定实体、媒体名称和专有名词有一定优势。

### 3.3 Dense 向量检索

目录：`hoprag/embeddings.py`、`hoprag/retrievers.py`

使用 Ollama 本地运行 BGE-M3 Embedding 模型。

核心流程：

```text
原始文档
   |
   v
文本分块
   |
   v
BGE-M3 Embedding
   |
   v
向量归一化
   |
   v
SQLite 缓存
   |
   v
查询向量化
   |
   v
余弦相似度计算
   |
   v
父文档聚合排序
```

主要实现：

- Embedding 维度：1024。
- 分块上限：3072 UTF-8 字节。
- 批量处理：8。
- 向量使用 NumPy 进行余弦相似度计算。
- 同一父文档取最高 Chunk 分数。
- SQLite 使用模型名称和文本 SHA-256 作为缓存标识。
- 支持断点复用，避免重复 Embedding。

### 3.4 RRF 混合检索

将 BM25 和 Dense 两路检索结果进行融合。

采用 Reciprocal Rank Fusion：

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+rank_r(d)}
\]

其中：

- \(d\)：候选文档。
- \(R\)：不同检索方法。
- \(rank_r(d)\)：文档在某一路检索中的排名。
- \(k=60\)：融合参数。

实验中：

- BM25 获取 Top 100。
- Dense 获取 Top 100。
- 两路候选按父文档 ID 去重。
- 使用 RRF 计算融合排名。
- 最终输出 Top-K 文档。

RRF 是项目正式检索基线，也是后续优化实验的重要参照。

---

## 4. 检索评价体系

目录：`hoprag/evaluation.py`

项目不仅使用传统检索指标，还增加了面向多跳问题的完整证据评价指标。

### 4.1 传统检索指标

| 指标 | 含义 |
|---|---|
| Hits@4 | 前 4 篇是否命中至少一条必要证据 |
| Hits@10 | 前 10 篇是否命中至少一条必要证据 |
| MAP@10 | 前 10 名的平均精度 |
| MRR@10 | 首次命中必要证据的排名倒数 |

### 4.2 多跳证据指标

**Evidence Recall@K**

衡量 Top-K 中找到了多少必要证据。

\[
Recall@K=\frac{\text{已找到的必要事实数}}{\text{必要事实总数}}
\]

实际项目按每道题分别计算，再对可回答问题取平均。

**Complete Evidence Coverage@K**

衡量系统是否找齐回答问题所需的全部必要证据。

只有当一道题的所有必要事实均被覆盖时，才算完整命中。

该指标是本项目最重要的评价指标。

需要注意：项目使用文档级来源覆盖评价。某篇来源文档进入 Top-K 后，可以覆盖该文档对应的多条必要事实。这不等同于证明交给生成模型的实际文本窗口一定包含全部事实。

### 4.3 数据集规模

使用公开 MultiHop-RAG 数据集。

| 数据项目 | 数量 |
|---|---:|
| 原始文章 | 609 |
| 问题总数 | 2,556 |
| 可回答问题 | 2,255 |
| 无答案问题 | 301 |
| 必要证据记录 | 6,084 |

数据按问题类型和证据数量进行分层抽样，随机种子为 42。

数据集来源：

- [MultiHop-RAG 数据集](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
- [MultiHop-RAG 论文](https://arxiv.org/abs/2401.15391)
- [ODC-BY 1.0 数据许可](https://opendatacommons.org/licenses/by/1-0/)

原始数据未随 GitHub 仓库公开发布，需要自行下载并放入指定目录。

---

## 5. 核心实验与研究成果

### 5.1 第一阶段：BM25、Dense、RRF 基线

全量 2,556 题评测结果：

| 方法 | Hits@10 | MAP@10 | MRR@10 | 完整证据覆盖@10 |
|---|---:|---:|---:|---:|
| BM25 | 99.25% | 0.4672 | 0.8118 | 61.60% |
| Dense | 98.71% | 0.4308 | 0.7561 | 58.58% |
| RRF | 99.78% | 0.4774 | 0.8192 | **67.14%** |

主要发现：

- BM25 和 Dense 具有明显互补性。
- RRF 能够整合两路不同的检索结果。
- RRF 的完整证据覆盖率优于单路检索。

但 RRF 仍然存在大量未找齐证据的问题。

在 741 道 RRF Top-10 失败题中：

- 722 道主要属于排名问题。
- 19 道在 RRF Top 100 仍然缺少证据。
- 其中只有 5 道必要来源没有进入两路 Top 100 候选并集。

由此确定：下一阶段应该优先优化排序，而不是盲目扩大候选池。

详细报告：[基线实验报告](BASELINE_REPORT.md)

### 5.2 第二阶段：Cross-Encoder 重排序

目录：

`analysis/reranking/`

引入两种模型：

- MiniLM
- Jina Reranker Turbo

实验比较：

1. 原始 RRF。
2. MiniLM 单独重排。
3. RRF + MiniLM。
4. Jina Turbo 单独重排。
5. RRF + Jina Turbo。

100 道可回答开发题结果：

| 方法 | 完整证据覆盖@10 |
|---|---:|
| 原始 RRF | 63% |
| MiniLM | 46% |
| RRF + MiniLM | 60% |
| Jina Turbo | 61% |
| RRF + Jina Turbo | **70%** |

实验发现：

单纯使用 Cross-Encoder 重新排序，不一定能提高多跳问题的完整证据覆盖率。

原因是：

某篇文档的单独相关性分数较高，并不代表它能够替代其他提供不同必要事实的文档。

因此，后续实验重点转向控制候选池和降低必要证据回退。

### 5.3 第三阶段：P5-B 低预算重排序优化

目录：

`analysis/reranking_optimization/`

这是目前项目最重要的优化成果。

P5-B 的核心设计：

1. BM25 取 Top 15。
2. Dense 取 Top 15。
3. 两路候选去重。
4. 使用 Jina Turbo INT8 ONNX 对候选分块评分。
5. 根据 RRF 和 Jina 排名进行 2:1 加权融合。
6. 输出 10 篇唯一父文档。

其核心目标不是让 Cross-Encoder 完全接管排序，而是让原有检索排名与深度语义相关性评分互相补充。

在 1,804 道独立保留题上的正式结果：

| 方法 | 完整证据覆盖@10 |
|---|---:|
| BM25 | 60.64% |
| Dense | 58.87% |
| 原始 RRF | 66.63% |
| **P5-B** | **70.62%** |

<badge color="success">正式验证结果</badge>

P5-B 相比 RRF：

- 完整覆盖率提升 3.99 个百分点。
- 找回 112 道原本不完整的题。
- 40 道原本完整的题发生回退。
- 净增加 72 道完整证据题。

该方案是当前正式推荐的检索排序方案。

详细报告：

[低预算重排序与正式保留集验证](analysis/reranking_optimization/OPTIMIZATION_REPORT.md)

### 5.4 第四阶段：BESA 证据集合选择

目录：

`analysis/besa_v0/`

研究目标：

不再只根据单篇文档相关性排序，而是从整个候选集合中选择能够覆盖不同必要证据的文档。

核心思路：

- Need Planning：将问题拆分成多个检索需求。
- Need Coverage：评价候选文档对不同需求的覆盖。
- Redundancy Penalty：惩罚重复证据。
- Evidence Selection：选择互补性更高的文档集合。

实验结果：

| 方法 | Tune 完整覆盖 | Validation 完整覆盖 | 全开发集 |
|---|---:|---:|---:|
| P5-B | 74.17% | 73.33% | 73.84% |
| BESA C-B30 | 75.28% | 73.33% | 74.50% |

结论：

BESA 在 Tune 上有少量提升，但 Validation 没有提升，全部新增恢复均来自 Tune。

因此暂不将 BESA 替换为正式方案，P5-B 继续作为基线。

### 5.5 第五阶段：EGE 候选扩展

目录：

`analysis/ege_v0/`

研究目标：

解决必要证据没有进入初始候选池的问题。

尝试：

- 扩大 BM25 候选深度。
- 扩大 Dense 候选深度。
- 基于原问题生成词法变体。
- 沿一跳、二跳路径扩展候选。
- 使用 Jina 对扩展结果重新排序。

主要结果：

| 方法 | 全开发集完整覆盖@10 |
|---|---:|
| P5-B | 73.84% |
| 一跳扩展 + Jina | 73.61% |

二跳扩展没有补充新的必要来源。

结论：

候选池扩大能够提高候选完整覆盖上限，但不代表最终 Top-10 完整证据覆盖一定提高。

继续使用 P5-B。

### 5.6 第六阶段：EGR 证据门控替换

目录：

`analysis/egr_v0/`

研究目标：

在已有 Top-10 结果中，仅当新增候选能够提供明确、可核验的额外证据时，才允许替换原有文档。

主要核验：

- 实体角色。
- 主体与客体关系。
- 时间作用域。
- 来源约束。
- 证据新颖性。
- 原有文档是否提供独有证据。

实验结果：

| 方法 | 全开发集完整覆盖@10 |
|---|---:|
| P5-B | 73.84% |
| 双路 Top 30 + Jina | 73.17% |
| E4-B1 门控 | 73.84% |

门控避免了未经核验的直接替换，但实际通过门槛的候选极少，没有产生新的完整证据题。

因此，EGR 仍属于探索性方案，不能宣称已经优于 P5-B。

---

## 6. 人工证据核验工作台

目录：

`analysis/evidence_verification_benchmark/`

项目开发了一个独立的本地人工核验工具，用于评估新增候选是否真正具备替换价值。

### 6.1 主要功能

- 350 个案例加载。
- 七个独立标注维度。
- 标注者身份隔离。
- 独立进度保存。
- 页面刷新与服务重启恢复。
- 上一题、下一题、编号跳转。
- 全部、已完成、未完成筛选。
- 原文证据引用校验。
- CSV 导出。
- 未保存内容保护。
- 多服务实例输出目录隔离。

### 6.2 七个标注维度

| 字段 | 评价内容 |
|---|---|
| evidence_support | 证据支持程度 |
| entity_role | 实体角色是否匹配 |
| relation_support | 目标关系是否得到支持 |
| temporal_scope | 时间范围是否一致 |
| temporal_kind | 时间类型 |
| evidence_novelty | 证据新颖性 |
| replacement_suitability | 是否适合替换原文档 |

### 6.3 启动工作台

需要准备本地盲审数据文件 `annotation_packets.md`，该文件不包含在公开仓库中。

PowerShell：

```powershell
$benchmark = '.\analysis\evidence_verification_benchmark'

& .\.venv\Scripts\python.exe -B `
  "$benchmark\human_review_workbench.py" `
  --data-dir $benchmark `
  --output-dir "$benchmark\human_review_outputs"
```

浏览器访问：

http://127.0.0.1:8765/

如果端口被占用：

```powershell
& .\.venv\Scripts\python.exe -B `
  "$benchmark\human_review_workbench.py" `
  --data-dir $benchmark `
  --output-dir "$benchmark\human_review_outputs" `
  --port 8766
```

详细说明：

[人工证据核验工作台说明](analysis/evidence_verification_benchmark/README_HUMAN_REVIEW_WORKBENCH.md)

注意：工作台的开发和自动化验收已经完成，但正式的人工标注及最终人工一致性分析仍需单独完成。

---

## 7. 项目目录结构

```text
hoprag-research-engineering/
│
├── hoprag/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── config.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── graph_regressions.py
│   ├── models.py
│   └── retrievers.py
│
├── multihop/
│   ├── __init__.py
│   ├── __main__.py
│   └── retrieval.py
│
├── analysis/
│   ├── reranking/
│   ├── reranking_optimization/
│   ├── besa_design_feasibility/
│   ├── besa_v0/
│   ├── ege_v0/
│   ├── egr_v0/
│   ├── evidence_verification_benchmark/
│   ├── audit_evaluation.py
│   └── FAILURE_ANALYSIS_REPORT.md
│
├── tests/
│   ├── test_baseline_retrieval.py
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   └── test_retrieval.py
│
├── data/
│   └── example.jsonl
│
├── requirements.txt
├── .gitignore
├── BASELINE_REPORT.md
└── README.md
```

---

## 8. 环境配置

### 8.1 基础环境

推荐使用：

- Windows 10 / 11
- Python 3.14.8
- Ollama
- NumPy 2.5.4
- rank-bm25 0.2.2

### 8.2 创建虚拟环境

```powershell
py -3.14 -m venv .venv

& .\.venv\Scripts\python.exe -m pip install --upgrade pip

& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 8.3 配置 Ollama

安装 Ollama 后执行：

```powershell
ollama pull bge-m3

ollama list
```

确保 Ollama 服务可以正常访问：

```text
http://127.0.0.1:11434
```

### 8.4 准备数据集

下载官方 MultiHop-RAG 数据集中的：

- corpus.json
- MultiHopRAG.json

放入：

```text
data/
└── raw/
    └── multihop_rag/
        ├── corpus.json
        └── MultiHopRAG.json
```

原始数据文件不需要修改格式。

---

## 9. 运行项目

### 9.1 数据审计

```powershell
& .\.venv\Scripts\python.exe -B -m hoprag --audit-only
```

检查：

- 数据字段。
- 文档数量。
- 问题数量。
- 必要证据来源映射。
- 数据完整性。

### 9.2 运行完整检索基线

```powershell
& .\.venv\Scripts\python.exe -B -m hoprag
```

默认执行：

1. 20 题冒烟测试。
2. 100 题小规模测试。
3. 2,556 题全量测试。

### 9.3 只运行指定规模

```powershell
& .\.venv\Scripts\python.exe -B -m hoprag --sizes 20
```

也可以：

```powershell
& .\.venv\Scripts\python.exe -B -m hoprag --sizes 20 100 full
```

### 9.4 运行重排实验

安装额外依赖：

```powershell
& .\.venv\Scripts\python.exe -m pip install fastembed==0.9.1 onnxruntime==1.31.1
```

准备模型：

```powershell
& .\.venv\Scripts\python.exe -B -m analysis.reranking.prepare_models
```

构建冻结候选池：

```powershell
& .\.venv\Scripts\python.exe -B -m analysis.reranking.pool
```

运行 Jina：

```powershell
& .\.venv\Scripts\python.exe -B -m analysis.reranking.run --model jina --context unified
```

注意：完整实验依赖此前冻结的排名、语料、分块映射和模型缓存。不能仅凭公开源码和空目录保证完整实验一键复现。

---

## 10. 自动化测试

项目使用 Python 标准库 unittest。

从项目根目录执行：

```powershell
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

    if ($LASTEXITCODE -ne 0) {
        throw "测试失败：$suite"
    }
}
```

### 最新验收结果

| 测试模块 | 通过数量 | 失败数量 |
|---|---:|---:|
| Core | 27 | 0 |
| Evidence Workbench | 32 | 0 |
| BESA | 26 | 0 |
| EGE | 17 | 0 |
| EGR | 32 | 0 |
| Reranking | 15 | 0 |
| Reranking Optimization | 12 | 0 |
| **合计** | **161** | **0** |

以上为本机最新验收报告记录的结果。

另对仅包含待发布文件的临时副本进行了复测，158 项通过，3 项因本地 Ollama 或研究归档数据缺失而按预期跳过。

---

## 11. 当前项目的研究局限

项目虽然完成了主要工程开发，但仍有一些需要明确的研究边界。

### 11.1 文档级覆盖不等于生成级证据完整

目前主要研究检索阶段。

评价的是必要证据来源是否进入 Top-K，而不是大语言模型最终是否正确利用所有证据生成答案。

### 11.2 无答案识别仍需独立研究

301 道无答案题中，检索器仍然会返回 10 篇候选文档。

返回文档不代表问题可回答。

目前尚未建立经过独立人工验证的完整拒答机制。

### 11.3 Cross-Encoder 输入存在截断

P5-B 的 Jina 评分使用有限长度的统一输入窗口。

部分文档的重要证据可能位于模型实际看到的窗口之外。

### 11.4 BESA、EGE、EGR 尚未成为正式最优方案

这三个方向已经完成探索性实现与实验，但没有取得足以替换 P5-B 的独立验证收益。

因此目前不应将它们描述为已经超过正式基线的优化技术。

### 11.5 人工证据核验尚未完成

350 个案例已经准备了 AI 辅助标注和人工核验工作流。

但 AI 辅助标注不能替代独立人工审查，正式的人工精确率、召回率及标注一致性指标仍需后续获得。

---

## 12. 后续研究方向

后续不建议盲目增加算法模块，优先解决以下问题：

1. 完成 350 个案例的独立人工证据核验。
2. 计算证据门控的精确率、召回率、漏放率及各维度准确性。
3. 分析推断题、比较题、时间题中的证据丢失差异。
4. 研究如何保留原 Top-K 中已经存在的必要证据。
5. 进一步研究无答案检测与可靠拒答。
6. 在不重复使用已冻结保留集进行调参的前提下，设计新的独立验证实验。

当前正式基线保持 P5-B。

---

## 13. 项目发布

GitHub：

**https://github.com/Lei-ounai/hoprag-research-engineering**

默认分支：`main`

当前发布版本：

`c791cba847cf510cd3c267aa30de1f01ee34d86f`

公开仓库不包含：

- 原始盲审文章材料。
- 个人人工标注数据。
- 本地模型缓存。
- 原始实验临时数据。
- 本机敏感信息。

数据集的许可与软件仓库自身的许可需要分别处理。当前仓库没有独立的根目录 LICENSE 文件，公开可见不等于已经授予第三方特定的软件再使用许可。

---

## 14. 项目总结

HopRAG 从最初的静态链接多跳检索原型，逐步发展为包含混合检索、语义重排序、证据集合选择、候选扩展、证据门控和人工证据核验工具的研究工程。

项目最重要的成果不是单纯增加了多少种检索算法，而是通过多轮对照实验发现：

- 单篇文档相关性不等于多跳证据完整性。
- BM25 和 Dense 具有互补性。
- Cross-Encoder 单独重排可能损失必要证据。
- 候选池扩大不一定改善最终 Top-K。
- 对必要证据的保护比盲目追求高相关性更加重要。
- 在有限计算预算下，P5-B 能够有效提升完整证据覆盖率。

最终，P5-B 在 1,804 道独立保留题上取得 70.62% 的完整证据覆盖率，相比原始 RRF 提高 3.99 个百分点。

**HopRAG 的核心研究方向：从检索相关文档，转向检索完整、互补且能够支持答案的证据集合。**
