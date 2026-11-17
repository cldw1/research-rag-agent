# 真实 PDF 单论文问答评测集

面向上传文献的实际处理流程：PDF → 统一解析 → 分块 → SQLite / FAISS 持久化 → 单论文检索 → 证据上下文 → 回答与引用审核。

[开发集协议](development_protocol.json) 固定检索参数和评分规则；[完整标注](corpus.json) 包含 10 篇论文、150 道候选题及参考答案要点。[完整审核表](corpus_questions.md) 可直接阅读。标注已由 AI 对照原文核查，独立人工复核状态保留为 `pending`。

## 论文与规模

| 论文 | 固定版本 | PDF 页数 | 题数 | 用途 |
|---|---|---:|---:|---|
| Attention Is All You Need | [1706.03762v7](https://arxiv.org/abs/1706.03762v7) | 15 | 15 | 开发 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401v4](https://arxiv.org/abs/2005.11401v4) | 19 | 15 | 开发 |
| LoRA: Low-Rank Adaptation of Large Language Models | [2106.09685v2](https://arxiv.org/abs/2106.09685v2) | 26 | 15 | 开发 |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | [1810.04805v2](https://arxiv.org/abs/1810.04805v2) | 16 | 15 | 预留测试 |
| Adam: A Method for Stochastic Optimization | [1412.6980v9](https://arxiv.org/abs/1412.6980v9) | 15 | 15 | 预留测试 |
| Deep Residual Learning for Image Recognition | [1512.03385v1](https://arxiv.org/abs/1512.03385v1) | 12 | 15 | 预留测试 |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | [1502.03167v3](https://arxiv.org/abs/1502.03167v3) | 11 | 15 | 预留测试 |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings | [2104.08821v4](https://arxiv.org/abs/2104.08821v4) | 17 | 15 | 预留测试 |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [2205.14135v2](https://arxiv.org/abs/2205.14135v2) | 34 | 15 | 预留测试 |
| On Calibration of Modern Neural Networks | [1706.04599v2](https://arxiv.org/abs/1706.04599v2) | 14 | 15 | 预留测试 |

共 179 页、140 道有答案问题、10 道无答案问题、205 处证据定位，包括事实、方法、实验、局限、多章节推理与原文冲突。定位中的页码从 PDF 第一页开始计数；不是期刊印刷页码。

论文清单及 3 篇开发 / 7 篇预留测试划分已在新增论文检索前固定，见 [选择与去重记录](selection.json)。10 篇论文均不在所核查的 QASPER 官方 Train / Validation / Test 论文 ID 集合内。该题库选自英文机器学习论文，不代表所有学科；经典论文也可能已在生成模型训练中出现。

[开发视图](development.json) 含 45 题；[预留测试视图](test_candidates.json) 含 105 题，均从完整标注派生。[版本清单](candidate_manifest.json) 校验文件哈希、论文划分、逐题内容及数量，防止不同视图漂移。数据集当前为候选标注版本；105 道测试题在独立复核和最终冻结前不评分。固定的 [30 题兼容样例](pilot.json) 保留原始内容。

## 标注方式

每道有答案问题保存参考答案要点和一个或多个证据单元。证据绑定原始 PDF 的 SHA256、真实页码、页内文字块序号、bbox、字符区间、片段哈希和章节标签，与 Chunk ID 无关。章节标签用于评审定位，不会注入索引，也不会启用 HMC 特定章节策略。

文字块序号采用 `get_text("blocks", sort=True)` 返回的完整记录列表下标，从 0 开始；字符区间对应该块文字 `strip()` 后的 Python 字符串，左闭右开。提取引擎版本记录在论文清单中，版本变化时先重新核验并发布数据集新版本。

人工复核时：

1. 打开对应固定版本 PDF，核对问题、答案和每处证据是否充分。
2. 检查数字、表头、公式适用条件与跨章节依赖，不能只核对关键词。
3. 对无答案题阅读全文及相关实验设置，确认所问具体信息没有报告；不能仅因一次检索失败就标为无答案。
4. 记录复核者、日期、修改原因；实际完成复核后才能将 `human_review` 改为 `verified`。
5. 最终测试前冻结论文、划分、标注及配置，保存文件哈希；测试后修改会产生新版本。

`attention-14` 要求识别原文中英法 BLEU 的冲突：摘要和表 2 为 41.8，6.1 节正文为 41.0。评分应奖励明确说明冲突，不能强制选择一个数值。无答案题覆盖未报告的延迟、能耗、实验配置、语言范围与分布漂移保证，不能从其他设置的结果推算答案。

## 复现

在项目根目录执行。下载仅访问清单中固定版本的公开 arXiv PDF；已有文件不覆盖，校验不匹配即停止。

```powershell
.\.venv\Scripts\python.exe scripts/fetch_pdf_benchmark.py --dataset benchmarks/pdf_qa/corpus.json
.\.venv\Scripts\python.exe scripts/check_pdf_benchmark.py --dataset benchmarks/pdf_qa/corpus.json --release-root benchmarks/pdf_qa
.\.venv\Scripts\python.exe scripts/run_pdf_benchmark.py --dataset benchmarks/pdf_qa/development.json --protocol benchmarks/pdf_qa/development_protocol.json --threads 8
```

下载后，校验和评测均可离线完成。运行器复用本地缓存 BGE，调用实际文档存储和检索模块，在独立实验目录新建数据库，不修改应用文档库。生成 API 调用数为 0。

回答生成由独立的 [运行器](../../scripts/run_pdf_answers.py) 执行。45 道开发题已使用冻结的 Anchor-First Prompt 和同一 DeepSeek 模型完成真实回答，并逐题进行 AI 来源初审。人工审核列保留空白，评分规则见 [审核量表](answer_review_rubric.json)，配置、结果与复现见 [回答报告](../../docs/evaluation.md#真实-pdf-问答)。

每次运行在 `data/pdf_benchmark/runs/` 的独立目录保存：

- 数据集、协议、原始文件身份及模型权重配置指纹。
- `source_snapshot/`：与运行时哈希对应的核心源码副本。
- 解析结果、证据映射、逐题排名和上下文覆盖、工程耗时。
- `generation_requests.jsonl`：真实 RAG Prompt 和来源；不含参考答案、Gold 标签或评分标准。
- `answer_review.csv`：空白回答和评分表，用于后续真实生成及人工审查；空白值不是零分。
- `summary.json`：检索诊断汇总，回答质量指标保持 `null`。

原始 PDF、索引、Prompt 和长证据文本均在本地实验目录，公开标注只保存来源坐标和哈希，不附整篇 PDF 或全文副本。

## 指标口径

开发集检索评测只统计其中 42 道有答案问题；3 道无答案题单独用于后续拒答评审。证据单元与 Chunk 的来源字符区间相交即可计入 `evidence_recall`；完整覆盖全部字符才计入 `full_evidence_recall`。多单元问题分别计数，`all_evidence_covered` 要求全部完整覆盖。Precision@K 固定除以 K，MRR / nDCG 使用相关 Chunk 排名。

先独立验证原始 PDF 的 Gold 坐标，再映射至实际解析结果。证据丢失或位置不唯一时保留在分母中计为未命中，另行报告，不能删除难题抬高分数。当前映射为同页、bbox 相交且原文片段精确唯一匹配，不进行模糊猜测；解析器改变文本或拆分原始文字块可能导致映射失败，需要单独审核。

BM25、Dense 和 Hybrid 使用同一分块、同一论文范围；每路候选 50、RRF k=60，K=1/5/10。Top-5 和 Anchor-First 使用相同 Hybrid 检索与 7400 字符上限，记录实际证据长度。BGE Token 数只作为证据长度参考，不能当作生成 API 账单。

有答案题需要人工评审正确性、忠实度和引用；无答案题单独评审拒答。引用编号存在只说明格式有效，不代表证据支持结论。只检索到任意一个证据不等于答案完整。

三篇开发论文的结果用于检查流程和发现问题；不计算用于泛化结论的置信区间。最终评测应按论文分组并成对比较，不能把同一论文的多道题当成相互独立的论文样本。[开发集报告](../../docs/evaluation.md#真实-pdf-问答) 同时记录命中、排序及证据完整性。

原有 [QASPER 主评测](../../docs/benchmark.md) 与 [HMC 专项](../../docs/experiments.md) 保留原始口径，不合并为一个分数。当前通用 PDF 路径没有启用 HMC 专用 Section-Aware，不能据此推断其跨论文表现。
