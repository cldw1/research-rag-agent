# Research RAG Agent

**基于 Hybrid Retrieval、Agent Tool Calling 与章节感知检索的科研论文智能问答系统**

Research RAG Agent 是一个面向英文科研论文阅读与知识检索的 RAG（Retrieval-Augmented Generation）项目。

系统结合 BM25 稀疏检索、BGE 语义向量检索、FAISS 向量索引、RRF 排名融合与 DeepSeek 大语言模型，实现论文检索、自然语言问答、工具调用及证据引用。

项目支持 QASPER 验证集论文和用户上传的 PDF，并提供实验性的 Section-Aware 检索策略，用于研究章节结构对检索质量的影响。

## 1. 核心功能

### 1.1 科研论文智能问答

- 支持使用自然语言询问论文的研究问题、方法、实验与结论。
- 通过 DeepSeek 实现基于论文证据的回答生成。
- 回答提供数字引用编号，并允许查看对应论文原文。
- 展示 Agent 的工具调用记录和引用来源。

### 1.2 Hybrid Retrieval 混合检索

结合两种检索方式：

- **BM25**：基于关键词的稀疏检索。
- **Dense Retrieval**：使用 `BAAI/bge-small-en-v1.5` 生成文本向量，并通过 FAISS 检索。

利用 Reciprocal Rank Fusion（RRF）融合两路检索排名：

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

这种设计同时利用关键词匹配和语义匹配的优势。

### 1.3 Agent 工具调用

系统通过 DeepSeek 的工具调用能力，在问答过程中选择工具：

- `get_paper_outline`：获取论文结构和章节概况。
- `search_research_paper`：检索与用户问题相关的论文正文。

Agent 可以根据问题进行检索、组织证据并生成回答，同时记录工具调用过程。

### 1.4 Anchor-First 上下文扩展

系统采用 Anchor-First 策略：

1. 优先保留原始高排名检索结果。
2. 根据文档位置扩展邻近段落。
3. 在上下文预算约束下组织证据。
4. 为证据分配引用编号。

该设计旨在减少片段化检索导致的上下文缺失，同时避免邻居段落挤掉核心检索结果。

### 1.5 Section-Aware 章节感知检索

在已验证的 HMC 测试论文上，系统支持可选的章节感知模式：

- 使用真实章节标题作为 Chunk 元数据。
- 根据问题与章节标题的匹配情况进行重排序。
- 对普通问答中的参考文献片段降权。
- 保留原始 Hybrid Top-1。
- 限制同一章节在最终 Top-5 中的 Chunk 数量。

这一模式采用保守的实验策略，**默认检索方式仍然是 Hybrid**。

当前章节感知功能依赖特定 HMC 测试 PDF 的已验证章节映射，尚未实现面向任意 PDF 的通用章节识别。

### 1.6 智能邻近证据扩展

针对已验证的 V3 章节索引，实现了基于章节边界的邻近 Chunk 扩展：

- 支持跨 PDF 页查找同章节内容。
- 根据关键词、距离和公式特征排序邻居。
- 优先保留原始 Anchor。
- 在上下文预算内补充相关证据。

在 HMC 论文测试中，系统成功检索并引用包含 Leapfrog 更新公式 5.18–5.20 的原文片段。

需要注意，PDF 数学公式的上下标和符号顺序可能存在解析误差。

### 1.7 Streamlit 交互界面

- 选择 QASPER 验证集论文。
- 上传本地英文 PDF。
- 查看 PDF 页数、Chunk 数量与 FAISS 向量数量。
- 使用多轮聊天界面询问论文。
- 查看工具调用记录与引用证据。
- 下载聊天记录 JSON。
- 在支持的测试 PDF 上切换 Hybrid / Section-Aware。

## 2. 技术栈

| 分类 | 技术 |
|---|---|
| 编程语言 | Python |
| 前端框架 | Streamlit |
| 大语言模型 | DeepSeek API |
| API 客户端 | OpenAI Python SDK |
| 语义 Embedding | BAAI/bge-small-en-v1.5 |
| 稀疏检索 | BM25 |
| 向量检索 | FAISS |
| 排名融合 | Reciprocal Rank Fusion |
| PDF 解析 | PyMuPDF |
| 数据处理 | NumPy、Pandas |
| 模型组件 | Sentence-Transformers |

## 3. 系统架构

```text
用户问题
   |
   v
Research Agent
   |
   +-- get_paper_outline
   |
   +-- search_research_paper
            |
            v
       Retrieval Layer
            |
            +-- BM25
            |
            +-- BGE Embedding + FAISS
            |
            v
          RRF
            |
            +-- Hybrid（默认）
            |
            +-- Section-Aware（实验性）
            |
            v
       Context Expansion
            |
            +-- Anchor-First
            |
            +-- Smart Section Expansion
            |
            v
       Evidence Context
            |
            v
       DeepSeek Generation
            |
            v
       回答 + 引用编号
            |
            v
       Streamlit 展示
```

系统将 PDF 处理、检索、证据扩展、回答生成和 Agent 工具调用拆分为独立模块，方便测试与维护。

## 4. 项目结构

```text
research-rag-agent/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── agent/
│   │   ├── paper_search_tool.py
│   │   ├── paper_outline_tool.py
│   │   └── citation_registry.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── load_qasper.py
│   │   └── pdf_chunker_v3.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── uploaded_pdf_retriever.py
│   │   ├── retrieval_strategy.py
│   │   ├── section_aware_reranker.py
│   │   ├── strategy_proxy.py
│   │   ├── verified_v3_pdf_retriever.py
│   │   └── smart_section_expander.py
│   │
│   └── generation/
│       ├── rag_pipeline.py
│       ├── context_builder.py
│       ├── strict_context_builder.py
│       └── answer_generator.py
│
├── scripts/
│   ├── demo_agent_live_dedup.py
│   ├── test_retrieval_strategy.py
│   ├── test_step69_strategy_proxy.py
│   └── test_step70_smart_expander.py
│
└── results/
    └── pdf_v3_section_mapping_report.json
```

以上为核心文件示意，实际仓库还包含其他实验、评测与调试脚本。

## 5. 实验与评估

### 5.1 章节级检索实验

使用论文 *MCMC Using Hamiltonian Dynamics* 中的已验证 V3 分块结果进行评测。

实验设置：

- 测试 PDF：50 页。
- V3 Chunk 数量：247。
- 匹配章节标题：50 个。
- 评测问题：7 道。
- 检索返回数量：Top-5。

比较原始 V3 Hybrid 与受约束 Section-Aware 重排序。

| 指标 | Hybrid | Section-Aware |
|---|---:|---:|
| Hit@5 | 85.7% | 100.0% |
| MRR@5 | 0.786 | 0.929 |
| 证据充分率 | 85.7% | 85.7% |
| 平均证据评分（0–2） | 1.857 | 1.857 |

### 5.2 结果分析

Section-Aware 在这七道测试问题上改善了章节命中率和首次命中章节的排名。

但人工证据充分性评分显示，两种方法均为 6/7。

这说明：

**更高的章节检索指标不一定直接转化为更充分的回答证据。**

因此，项目保留 Hybrid 作为默认策略，并将 Section-Aware 作为可选实验功能。

### 5.3 实验局限

- 测试集仅包含同一篇 HMC 论文的七道章节级问题。
- 问题与主要章节标签根据论文内容设计，标签仍需进一步人工核验。
- 证据充分性评分来自单轮人工评估，而非多评审独立标注。
- 实验结果不能视为跨领域、跨论文的通用准确率。
- 数学公式的 PDF 文本提取仍存在符号与排版误差。

## 6. 安装与运行

### 6.1 克隆项目

项目发布到 GitHub 后：

```bash
git clone <your-repository-url>
cd research-rag-agent
```

其中 `<your-repository-url>` 需要替换为实际仓库地址。

### 6.2 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 6.3 安装依赖

```powershell
python -m pip install -r requirements.txt
```

首次使用时，Sentence-Transformers 可能需要下载 BGE Embedding 模型。

模型下载需要网络连接，模型缓存不包含在 Git 仓库内。

### 6.4 配置 API Key

复制项目根目录的 `.env.example`：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

请使用自己的有效 API Key，不要将真实密钥提交到 GitHub。

当前应用通过 OpenAI Python SDK 兼容接口连接 DeepSeek API。

### 6.5 启动应用

在项目根目录执行：

```powershell
python -m streamlit run app.py --server.port 8501
```

浏览器访问：

`http://localhost:8501`

### 6.6 使用说明

项目提供两种论文来源。

**QASPER 验证集论文**

该模式依赖本地预先准备的 QASPER 数据及检索索引。由于原始数据与索引文件不随 Git 仓库发布，使用前需要根据项目的数据准备脚本完成构建。

**上传自己的 PDF**

用户可以通过 Streamlit 页面上传文本型 PDF，并在本地构建检索索引。当前版本主要面向可提取文本的英文 PDF，不支持扫描版 OCR。

普通 PDF 默认使用 Hybrid 检索。

对于已验证的特定 HMC PDF，系统可以根据文件哈希与已发布的章节映射启用 V3 Section-Aware 模式。其他 PDF 不会自动获得这一模式。

当前 PDF 上传实现的主要限制为：

- 文件不超过 20 MB。
- PDF 不超过 80 页。
- Chunk 数量不超过 1500。
- 扫描版 PDF 暂不支持 OCR。

## 7. 基础测试

运行检索策略测试：

```powershell
python scripts/test_retrieval_strategy.py
```

测试策略代理：

```powershell
python scripts/test_step69_strategy_proxy.py
```

测试智能章节邻近扩展：

```powershell
python scripts/test_step70_smart_expander.py
```

这些测试主要验证接口行为、章节约束和邻近扩展规则。完整的 Agent 问答还依赖 API Key、模型和本地检索索引。

## 8. 功能截图

项目已实现以下可展示页面：

- QASPER 论文智能问答。
- 普通 PDF 上传与检索。
- Section-Aware 检索策略选择。
- Agent 工具调用过程。
- 原文引用与 Chunk 来源追踪。

正式发布时可在 `docs/images/` 中补充经过脱敏的截图。

## 9. 后续优化方向

1. 实现适用于不同论文的通用章节识别。
2. 改善数学公式的结构化解析与展示。
3. 支持更大规模的 PDF 与多论文检索。
4. 建立更大、人工核验的问答评测集。
5. 增加回答忠实度、引用准确率及检索延迟评估。
6. 优化索引缓存、数据持久化和模型加载速度。

## 10. 项目说明

本项目主要用于学习和实践 RAG、Agent Tool Calling、混合检索、信息检索评估与 Python 工程化开发。

目前已完成本地功能演示、基础单元测试以及部分检索实验，但仍属于可持续改进的原型系统，并非已经完成大规模生产环境验证的服务。

**技术关键词：** RAG · AI Agent · DeepSeek · Hybrid Retrieval · BM25 · FAISS · BGE · RRF · Section-Aware Retrieval · Streamlit
