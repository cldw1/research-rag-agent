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

选择已索引文档并新建会话，填写供应商、模型名、API Key 与参数。可配置 DeepSeek、OpenAI Responses、Claude、Gemini、百炼和自定义兼容服务，模型及账号权限按所用服务设置。

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

会话绑定具体索引。重新分块或切换 Embedding 后重建索引，新问答使用新会话，旧引用保留原来源；只切换生成模型不需要重做同一 Embedding 索引。

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
