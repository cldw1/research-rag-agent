# 使用指南

## 安装与启动

按 [快速开始](../README.md#快速开始) 安装依赖，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

个人文献通过文档工作台保存和索引，再进入科研对话；QASPER 与章节检索可在首页使用。

## 文档解析与分块

工作台支持批量 PDF、DOCX、MD、TXT。默认上限为 20 MB、PDF 80 页和 1500 个 Chunk。

1. 选择文件，点击“保存并解析上传文档”。
2. 在“解析结果”检查结构、位置和处理提示。
3. 选择分块策略，在“分块与来源”查看片段与 source_metadata。
4. 点击“构建或复用持久化索引”。匹配的内容与配置可复用。
5. 重启后选择已保存文档、恢复索引，继续检索或新建会话。

轻量预览使用 UTF-8 字节预算，正式索引用 BGE tokenizer 重新分块，数量以已保存索引为准。PDF 引用保存真实页码和解析块坐标，Word 使用段落序号，MD/TXT 使用段落或行号。

### OCR 与结构配置

| 路径 | 配置 |
|---|---|
| 轻量 PDF | PyMuPDF 文本提取；按需勾选 OCR 并配置 Tesseract 语言数据 |
| DOCX | OOXML 标题、段落、列表和基础表格 |
| MD / TXT | 标题、自然段、代码/公式围栏及基础表格，可显式选择编码 |
| Docling / RapidOCR | 可选依赖和本地模型，提供 PDF 版面与 OCR 解析入口 |
| HybridChunker | hybrid 分块，使用结构与 Embedding tokenizer 预算 |

需要可选 PDF 后端时：

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements.txt -r requirements-docling.txt
.\.venv\Scripts\python.exe scripts/prepare_document_models.py --help
```

按脚本参数准备模型，在工作台填写目录。OCR 受扫描与版面质量影响，处理提示及超额拆分标记便于复核原文。

## 科研对话与模型配置

选择 1–5 篇已索引文档并新建会话，填写供应商、模型名、API Key 与参数。可配置 DeepSeek、OpenAI Responses、Claude、Gemini、百炼和自定义兼容服务，模型及账号权限按所用服务设置。

多文档会话可提问“比较两篇论文的方法与局限”等问题。工具既可搜索全部所选文档，也可限定到其中一篇。来源面板按文件名、标题和真实位置区分文献；若本次工具证据未覆盖某篇，页面会提示，需补充取证后再判断该篇结论。选中文档与有充分证据支持答案是不同层次。

会话文档范围在创建时固定。调整文档组合请新建会话，已有历史和引用继续属于原组合。

Key 使用密码输入框，默认仅保存在服务内存与当前会话，也可在 `.env` 中设置可选默认值：

| 环境变量 | 用途 |
|---|---|
| DEEPSEEK_API_KEY | DeepSeek |
| OPENAI_API_KEY | OpenAI |
| ANTHROPIC_API_KEY | Claude |
| GEMINI_API_KEY | Gemini |
| DASHSCOPE_API_KEY | 百炼 |
| COPILOT_DATA_DIR | 文档与会话保存目录 |

自定义接口可配置 Base URL、模型名及 Tool Calling 能力。不支持工具时使用普通 RAG：程序先检索，再请求生成。百炼的地址、Key 和模型权限应匹配地域或工作空间。

生成时可查看流式文本并停止；重新生成保留原回答并追加回合，失败重试使用新请求 ID。会话可重命名、删除或导出 Markdown，回答引用对应到原文。

会话绑定所选文档及每篇的具体索引。任一文档重新分块或切换 Embedding 后重建索引，新问答使用新会话，旧引用保留原来源；只切换生成模型不需要重做同一 Embedding 索引。

### 检索与回答选项

多文档会话左侧的 **多文档证据顺序** 可选择 RRF（默认）、两路首位优先（BM25 先）或两路首位优先（Dense 先）。首位优先在候选池内优先安排关键词与语义检索的首位片段，仍按论文分配名额；可能挤掉原排名中的其他有用片段。单文档会话禁用该选项，继续标准检索。

该设置仅影响多篇联合混合检索，Agent 指定单篇取证和证据补检的单路搜索沿用各自规则。每个已保存回答同时展示本轮设置及实际正文检索顺序；详细调用范围保存在“本轮上下文和用量”的 `retrieval_settings`。历史和 Markdown 导出保留当轮设置，切换当前控件不会改写旧回答。

策略在加载本轮检索器时固定，每个生成任务使用独立检索器与调用记录。开始生成后切换设置只影响后续请求；同一个请求 ID 再次提交会恢复原结果，不会按新设置重复生成。本轮检索配置随回答保存。

科研对话页左侧提供 **证据不足时补检一次** 开关，默认关闭。开启后，回答明确表示证据不足时，系统可在本会话所选文档内补检并最多补答一次。它会保留原引用证据、按预算补充来源，并使用当前生成模型；可能增加耗时和模型用量。

多文档会话会逐篇进行关键词与语义补检，再按论文轮流安排新增片段，优先给原上下文未覆盖的论文取证。首答已引用的证据及每篇已有的首个片段会保留。新增片段合计最多4个，仍受上下文预算限制，因此不是每篇都一定有新证据。单篇会话继续使用原有两路补检。

生成过程中会显示补检和补答状态。保存后的回答提供“补检记录与新增来源”，可查看额外模型请求次数、新增片段和被替换的首答。无新证据、预算不足、补答失败、截断或引用编号无效时保留原回答；停止生成仍会取消整个未完成回合。补检开关在开始本轮任务时固定，切换页面设置不会改变已在运行的任务。

多文档补检记录还显示本地检索次数和各篇的新增片段数。2–5篇文档最多分别进行4–10次本地检索，不是4–10次生成模型请求；额外生成请求仍最多1次。单篇检索失败时保留首答，记录已尝试的本地检索次数，不保存未采用的试用引用。

聊天补答保留完整工具请求和返回消息，来源使用同一会话的稳定编号。原有工具证据与新增证据合计仍受输入预算限制，可能因此跳过补答。在线 Agent 场景与 [开发集普通 RAG 实验](evaluation.md#证据不足时补检) 的消息组织不同，实验分数不能直接视为界面中所有问题的回答质量。

左侧 **强调来源与实验条件** 为可选回答提示，默认关闭。开启后提醒当前模型区分本文实验和背景介绍，保留数值适用的任务、模型及测量条件；具体规则与真实对照见 [回答来源与条件](evaluation.md#来源与实验条件约束)。本轮版本保存在“本轮上下文和用量”，切换选项仅影响后续新请求。合并引用如 `[1, 2]` 会分别关联两个来源，未知编号仍单独提示。

首位优先的 [开发集对照](evaluation.md#多文献检索与回答) 比较直接 Top-5；对话仍可能使用 Anchor-First 邻段扩展以及 Agent 自选检索问题。界面提供显式选择，不将该小型实验的覆盖指标当作在线回答准确率。

## 数据管理与备份

默认目录 `data/copilot_v2` 相对于项目根目录解析。原始文件、索引和数据库共同构成文档库。

```powershell
.\.venv\Scripts\python.exe scripts/manage_documents.py list
.\.venv\Scripts\python.exe scripts/manage_documents.py jobs
.\.venv\Scripts\python.exe scripts/manage_documents.py schema-version
```

备份和恢复使用尚不存在的目标目录：

```powershell
.\.venv\Scripts\python.exe scripts/manage_documents.py backup data/backups/library_copy
.\.venv\Scripts\python.exe scripts/manage_documents.py restore data/backups/library_copy data/restored_library
```

将 `COPILOT_DATA_DIR` 指向恢复目录并重启，即可使用恢复后的库。已有库需要新增会话表时，页面提供“备份并启用聊天存储”，也可运行 `manage_documents.py upgrade-chat`，会先创建完整备份。

已有聊天库启用多文档时，页面提供“备份并启用多文档会话”。也可显式运行：

```powershell
.\.venv\Scripts\python.exe scripts/manage_documents.py upgrade-multidoc
```

该操作先完整备份，再新增会话文档关联表并回填原有单文档绑定，不改写原消息、引用或索引文件。未启用聊天存储的库先执行 `upgrade-chat`；备份可按上面的恢复命令还原到新目录。

文档移出采用可恢复的软删除，恢复后可显式复用或重建索引。中断任务可在界面或 `manage_documents.py recover` 中标记过期状态后重新提交。

## QASPER 与章节检索

交互示例按 [README](../README.md#qasper-数据准备) 准备。FAISS、向量和 Chunk 元数据应来自同一次构建；全量 Test 证据评测分别保存，见 [QASPER 实验](benchmark.md)。

HMC Section-Aware 使用指定文档的真实章节映射。提供 `HandbookChapter5.pdf`，SHA256 为：

```text
9e886059d25db512b670fed4d064d52cd5af41a8510f4d6adfeffff87ec43e4b
```

在首页上传建索引后可选择 Hybrid 或 Section-Aware，其他文档使用 Hybrid。实验条件见 [章节实验](experiments.md)。

## 常见处理提示

| 提示 | 处理方式 |
|---|---|
| 缺少 BGE 缓存 | 首次允许模型下载，完成后可设置离线环境变量 |
| 无有效文本 / OCR 提示 | 检查原页，按需启用 OCR 并准备识别依赖 |
| 解析部分成功 | 查看对应页面和内容提示，结合原文复核 |
| 模型身份不匹配 | 按当前 Embedding 配置重新构建索引 |
| 会话索引已变更 | 按当前索引新建会话，历史仍可查看 |
| 当前回合超出预算 | 缩小证据范围，或按模型能力调整应用预算 |
| 文件缺失或校验失败 | 从完整备份恢复，保留原目录用于核查 |

模型缓存与个人文献保存在本地。在线生成会向选定供应商发送当前问题、选中的历史和证据，使用时按文档内容选择相应服务。
