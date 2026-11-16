# HopRAG 人工证据核验工作台

本地离线标注工具，用于审查 evb-0001 至 evb-0350。前端使用原生 HTML/CSS/JavaScript，后端仅依赖 Python 标准库，不需要 Node.js、数据库、外部 API 或模型服务。

## 功能

- 按原始编号展示 350 个案例的问题、挑战者评分窗口 C、指定比较对象 I、当前 Top 10 片段。
- 展示七个标注维度及规范中的合法枚举。
- 仅允许引用当前案例实际可见窗口中的连续原文；直接支持、部分支持和反证必须引用挑战者 C 段。
- 每个标注者使用独立目录，进度以原子写入方式持久化；重新进入可恢复上次位置和已完成案例。
- 支持上一题、下一题、全部/未完成/已完成筛选、编号跳转和完成进度统计。
- 导出带 UTF-8 BOM 的正式 CSV，保留全部 350 个案例编号及模板列顺序。
- 限制服务只监听 127.0.0.1，校验 Host、Origin、请求类型、字段和见证文本。

## 数据输入

启动时，--data-dir 指定的目录必须包含以下三个文件：

- annotation_packets.md：350 个盲审案例及其可见证据窗口。
- annotation_guide.md：人工标注规范。
- annotation_template.csv：正式空白模板。

annotation_packets.md 包含研究用原文片段和盲审材料，不随公开代码仓库发布。请使用你有权访问的本地原始文件。annotation_guide.md 和 annotation_template.csv 保留在仓库中。

## 启动

在项目根目录的 PowerShell 中执行：

~~~powershell
$root = (git rev-parse --show-toplevel).Trim()
$benchmark = Join-Path $root 'analysis\evidence_verification_benchmark'
$python = Join-Path $root '.venv\Scripts\python.exe'
& $python -B (Join-Path $benchmark 'human_review_workbench.py') --data-dir $benchmark --output-dir (Join-Path $benchmark 'human_review_outputs')
~~~

浏览器打开：

~~~text
http://127.0.0.1:8765/
~~~

如果盲审数据保存在仓库目录之外，将 --data-dir 改为包含上述三个输入文件的本地目录。也可用 --port 8766 更换端口。停止服务时，在启动它的终端按 Ctrl+C。

## 标注与保存

1. 输入自己的独立标注者 ID。长度为 1–40 个字符，以小写英文字母或数字开头，后续可包含小写字母、数字、下划线或连字符；请避开 Windows 保留设备名。
2. 左栏查看原问题、挑战者 C、指定比较对象 I 和当前 Top 10 的可见片段。折叠的完整 Chunk 仅供定位，不能作为评分窗口外的证据。
3. 右栏逐项填写七个标签、判断理由、见证编号和原文引文。
4. 正向支持与反证必须引用 C 段。引文必须是对应可见窗口中的连续原文。确实没有可引用原文时，见证编号填写 NONE，引文留空，并在理由中说明缺失的事实。
5. 点击“保存并标记完成”。只有通过服务端校验并成功落盘的案例才计入完成数。
6. 离开未保存案例时，工作台会提示确认；关闭或刷新页面时，浏览器也会提示存在未保存修改。

## 数据位置与备份

默认输出目录：

~~~text
human_review_outputs/
  <标注者 ID>/
    human_review_progress.json
    HUMAN_ANNOTATION_350.csv
~~~

进度文件在每次保存时写入；CSV 在点击导出时生成。多人审查必须使用不同标注者 ID。同时启动第二个工作台时请指定不同的输出目录。每个输出目录由一个服务实例负责，避免两个进程相互覆盖进度。需要备份时，复制对应标注者目录。不要将个人标注进度、最终标注 CSV 或盲审原始材料提交到公开仓库。

## 测试

从项目根目录运行：

~~~powershell
& .\.venv\Scripts\python.exe -m unittest discover -s analysis\evidence_verification_benchmark\tests -v
~~~

工作台测试在临时目录构造 350 个合成案例，可在没有私有盲审包的公开仓库检出中运行。测试覆盖 350 案例加载、七维枚举、标注者隔离、进度恢复、非法标签与见证拒绝、CSV 中文/引号/换行往返、HTTP API、跨来源请求拒绝和标注输出完整性。
