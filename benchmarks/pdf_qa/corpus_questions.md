# PDF 问答题目与标注审核

本表由已通过来源校验的标注生成。AI 已核对来源，独立人工复核待完成；预留测试题不参与开发评分。

定位页码为 PDF 阅读器从 1 开始的页码。参考答案是评分要点，不是模型生成回答。

## attention-01

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

How many encoder layers are used, and what is their output dimension?

参考要点：

- 编码器含 6 层。
- 子层和嵌入输出维度为 512。

原文定位：

- e1：PDF 第 3 页，章节 3.1，页内块 3，字符 [0, 630)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-02

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

How many attention heads are used, and what are the key and value dimensions per head?

参考要点：

- 8 个头；每个头的 key 和 value 维度均为 64。

原文定位：

- e1：PDF 第 5 页，章节 3.2.2，页内块 11，字符 [0, 257)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-03

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

What are the input/output and inner dimensions of the position-wise feed-forward network?

参考要点：

- 输入输出维度 512，内部维度 2048。

原文定位：

- e1：PDF 第 5 页，章节 3.3，页内块 20，字符 [0, 305)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-04

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How does decoder self-attention preserve the autoregressive property?

参考要点：

- 在 softmax 前将不允许的未来位置设为负无穷，阻止访问后续 token。

原文定位：

- e1：PDF 第 5 页，章节 3.2.3，页内块 16，字符 [0, 445)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-05

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How is scaled dot-product attention computed, and why are the dot products scaled?

参考要点：

- Q 与 K 点积后除以 key 维度的平方根，经 softmax 得到 V 的权重。
- 缩放用于缓解大点积使 softmax 进入极小梯度区域的问题。

原文定位：

- e1：PDF 第 4 页，章节 3.2.1，页内块 4，字符 [0, 301)。
- e2：PDF 第 4 页，章节 3.2.1，页内块 10，字符 [0, 394)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-06

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How are the outputs of multiple attention heads combined, and what benefit is claimed?

参考要点：

- 各头输出拼接后再投影。
- 可同时关注不同位置的不同表示子空间。

原文定位：

- e1：PDF 第 5 页，章节 3.2.2，页内块 0，字符 [0, 119)。
- e2：PDF 第 5 页，章节 3.2.2，页内块 1，字符 [0, 189)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-07

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

Why are positional encodings needed, and why did the authors choose sinusoids over learned embeddings?

参考要点：

- 模型没有循环或卷积，需要额外注入顺序。
- 两种编码实验结果接近；选择正弦编码基于可能向更长序列外推的假设，并非已证明。

原文定位：

- e1：PDF 第 6 页，章节 3.5，页内块 3，字符 [0, 520)。
- e2：PDF 第 6 页，章节 3.5，页内块 8，字符 [0, 311)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-08

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

What training data size and tokenization vocabulary were used for English-German and English-French translation?

参考要点：

- 英德约 450 万句对、约 37000 共享 BPE 词表。
- 英法 3600 万句、32000 word-piece 词表。

原文定位：

- e1：PDF 第 7 页，章节 5.1，页内块 6，字符 [0, 591)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-09

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

What hardware and training schedules are reported for the base and big models?

参考要点：

- 使用 8 张 P100。
- base 10 万步约 12 小时；big 30 万步约 3.5 天。

原文定位：

- e1：PDF 第 7 页，章节 5.2，页内块 8，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-10

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

Which Adam settings and learning-rate warmup schedule were used?

参考要点：

- Adam 参数 beta1=0.9、beta2=0.98、epsilon=1e-9。
- 预热 4000 步线性增加，之后按步数平方根的倒数衰减。

原文定位：

- e1：PDF 第 7 页，章节 5.3，页内块 10，字符 [0, 153)。
- e2：PDF 第 7 页，章节 5.3，页内块 13，字符 [0, 220)。

核查备注：同节多段联合证据。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-11

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

How were translation checkpoints combined and beam search configured?

参考要点：

- base 平均最后 5 个 checkpoint，big 平均最后 20 个。
- beam size=4，length penalty=0.6；最大输出长度为输入长度加 50，允许提前停止。

原文定位：

- e1：PDF 第 8 页，章节 6.1，页内块 18，字符 [0, 456)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-12

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

According to Table 2, what English-German BLEU scores are reported for Transformer base and big?

参考要点：

- base 为 27.3，big 为 28.4。

原文定位：

- e1：PDF 第 8 页，章节 Table 2，页内块 10，字符 [0, 45)。
- e2：PDF 第 8 页，章节 Table 2，页内块 11，字符 [0, 38)。

核查备注：表格渲染已目视核对列与行，不能将此论文结果当作本项目结果。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-13

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`limitation`；可回答：True；人工复核：`pending`

Under what sequence-length condition is self-attention described as computationally faster than recurrence, and what is proposed for very long sequences?

参考要点：

- 文中限定 n 小于表示维度 d。
- 长序列可限制局部邻域，但会增加最大路径长度；这是后续研究建议。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 14，字符 [0, 300)。
- e2：PDF 第 7 页，章节 4，页内块 0，字符 [0, 581)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-14

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`source_conflict`；可回答：True；人工复核：`pending`

Are the English-French BLEU figures in the abstract, Table 2, and Section 6.1 consistent?

参考要点：

- 摘要和表 2 为 41.8，而 6.1 节正文为 41.0。
- 应明确指出原文冲突，不能擅自统一或猜测原因。

原文定位：

- e1：PDF 第 1 页，章节 Abstract，页内块 16，字符 [0, 1140)。
- e2：PDF 第 8 页，章节 Table 2，页内块 11，字符 [0, 38)。
- e3：PDF 第 8 页，章节 6.1，页内块 17，字符 [0, 337)。

核查备注：PDF 第 8 页已目视核对；保留冲突，不修订原文。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-15

论文：Attention Is All You Need（1706.03762v7）

划分：`dev`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What batch-size-one inference latency in milliseconds is reported for Transformer big?

参考要点：

- 本文没有提供该具体推理延迟，应说明无法由文献确定。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：全文未发现 latency、inference time 或 batch size 对应报告；训练单步耗时不能替代推理延迟。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-01

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

Which pretrained models provide the retriever and generator, and how large is the generator?

参考要点：

- 检索器初始化自 DPR；生成器为约 4 亿参数的 BART-large。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 30，字符 [0, 1075)。
- e2：PDF 第 3 页，章节 2.3，页内块 31，字符 [0, 583)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-02

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How do RAG-Sequence and RAG-Token differ in their use of latent documents?

参考要点：

- Sequence 对整段输出使用同一潜在文档；Token 可为不同 token 使用不同文档。

原文定位：

- e1：PDF 第 3 页，章节 2.1，页内块 1，字符 [0, 563)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-03

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How are queries and documents encoded and how are top documents retrieved?

参考要点：

- 采用 BERT 双编码器，分别编码查询和文档。
- 检索最高内积分数的文档，使用 MIPS。

原文定位：

- e1：PDF 第 3 页，章节 2.2，页内块 28，字符 [0, 32)。
- e2：PDF 第 3 页，章节 2.2，页内块 29，字符 [0, 672)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-04

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

Which components are updated during task fine-tuning and which are held fixed, and why?

参考要点：

- 训练查询编码器和 BART；文档编码器及索引保持固定。
- 更新文档编码器需反复重建索引，成本较高。

原文定位：

- e1：PDF 第 4 页，章节 2.4，页内块 1，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-05

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

Does RAG task fine-tuning require direct labels identifying the document to retrieve?

参考要点：

- 不需要直接标注应检索哪篇文档；使用输入输出对训练。

原文定位：

- e1：PDF 第 3 页，章节 2.4，页内块 33，字符 [0, 194)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-06

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

Which Wikipedia snapshot, passage size, corpus size, and vector index are used?

参考要点：

- 2018 年 12 月 Wikipedia；不重叠的 100 词片段，约 2100 万片段。
- FAISS MIPS 索引使用 HNSW 近似。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 10，字符 [0, 733)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-07

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

Which four open-domain QA datasets are evaluated, and which metric is reported?

参考要点：

- NQ、TriviaQA、WebQuestions、CuratedTrec；指标为 Exact Match。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 12，字符 [0, 1040)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-08

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

How is FEVER formulated as a generation task, and which labels are used?

参考要点：

- 将类别映射为单个输出 token。
- 三分类为支持、反驳、信息不足，同时评测二分类版本。

原文定位：

- e1：PDF 第 5 页，章节 3.4，页内块 6，字符 [0, 1181)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-09

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`limitation`；可回答：True；人工复核：`pending`

In the retrieval ablation, on which task does BM25 perform best, and what explanation do the authors suggest?

参考要点：

- FEVER；作者推测该任务实体集中，适合词重叠匹配。
- 该解释属于推测，不能表述为已验证因果。

原文定位：

- e1：PDF 第 7 页，章节 4.5，页内块 39，字符 [0, 481)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-10

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

How does the index hot-swapping experiment update knowledge, and what accuracies are reported with matched and mismatched indices?

参考要点：

- 用 2016 与 2018 年索引分别查询 82 位发生变化的领导人。
- 年份匹配正确率为 70% 和 68%；2018 索引回答 2016 信息为 12%，反向为 4%。

原文定位：

- e1：PDF 第 7 页，章节 4.5，页内块 40，字符 [0, 479)。
- e2：PDF 第 8 页，章节 4.5，页内块 11，字符 [0, 483)。

核查备注：同节跨页证据。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-11

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

How does retrieving more documents at test time affect open-domain QA for the two RAG variants?

参考要点：

- 文中 RAG-Sequence 随检索文档数增加而改善；RAG-Token 在 10 篇时达到峰值。
- 结论限定本文设置，不能概括成检索越多越好。

原文定位：

- e1：PDF 第 8 页，章节 4.5，页内块 12，字符 [0, 662)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-12

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

In the Hemingway generation example, how do parametric and non-parametric memories cooperate?

参考要点：

- 检索文档引导书名的起始生成，后续书名可由参数知识补全。

原文定位：

- e1：PDF 第 6 页，章节 4.3，页内块 17，字符 [0, 1338)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-13

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`limitation`；可回答：True；人工复核：`pending`

Does retrieving Wikipedia guarantee factual and unbiased answers according to the broader-impact discussion?

参考要点：

- 不保证；外部知识本身也可能错误或有偏差，生成仍有误用风险。

原文定位：

- e1：PDF 第 10 页，章节 Broader Impact，页内块 2，字符 [0, 737)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-14

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`multi_section`；可回答：True；人工复核：`pending`

Are a fixed document index during training and index replacement at test time contradictory? Explain the two operations.

参考要点：

- 训练时固定索引以避免反复更新文档编码。
- 推理时可直接替换非参数记忆以更新知识，无需重新训练参数模型。

原文定位：

- e1：PDF 第 4 页，章节 2.4，页内块 1，字符 [0, 398)。
- e2：PDF 第 7 页，章节 4.5，页内块 40，字符 [0, 479)。
- e3：PDF 第 8 页，章节 4.5，页内块 11，字符 [0, 483)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-15

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

划分：`dev`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What average per-query inference latency in milliseconds is reported for RAG-Sequence on NQ at k=5?

参考要点：

- 本文未报告该设置的毫秒级平均延迟，应说明无法从论文确定。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查正文、附录和全文时间关键词，未见该设置的延迟；k 的效果曲线不等于延迟数据。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-01

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

Which model weights are frozen and which matrices are trained in LoRA?

参考要点：

- 冻结原始权重 W0，仅训练低秩更新中的 A 和 B。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 7，字符 [0, 951)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-02

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How is a weight update parameterized and what are the shapes of its two factors?

参考要点：

- 对 d×k 的 W0，更新为 BA，其中 B 为 d×r、A 为 r×k，r 远小于 d 和 k。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 7，字符 [0, 951)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-03

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How are A and B initialized and why does this preserve the initial pretrained mapping?

参考要点：

- A 用随机高斯初始化、B 为零，因此初始 BA=0。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 9，字符 [0, 182)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-04

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

What scaling is applied to the low-rank branch and what purpose is stated?

参考要点：

- 更新分支乘 alpha/r，用于减少改变秩时重新调参的需要。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 9，字符 [0, 182)。
- e2：PDF 第 4 页，章节 4.1，页内块 10，字符 [0, 329)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-05

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

Which Transformer modules are adapted or frozen in the study described in Section 4.2?

参考要点：

- 研究主要适配注意力权重，冻结 MLP；并非已经实验验证所有模块。

原文定位：

- e1：PDF 第 5 页，章节 4.2，页内块 2，字符 [0, 838)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-06

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`fact`；可回答：True；人工复核：`pending`

Which two attention projections receive LoRA in most experiments?

参考要点：

- 多数实验仅适配 query 与 value 投影 Wq、Wv。

原文定位：

- e1：PDF 第 6 页，章节 5.1，页内块 11，字符 [0, 391)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-07

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`method`；可回答：True；人工复核：`pending`

How can LoRA avoid extra inference latency after deployment?

参考要点：

- 预先将 BA 合并进 W0，以 W0+BA 执行原有矩阵乘法。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 12，字符 [0, 369)。
- e2：PDF 第 5 页，章节 4.1，页内块 0，字符 [0, 122)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-08

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

What GPT-3 training-memory and task-checkpoint sizes are reported for the r=4 setup?

参考要点：

- 文中训练显存从 1.2TB 降到 350GB；仅适配 Q/V、r=4 的任务 checkpoint 从约 350GB 降到 35MB。

原文定位：

- e1：PDF 第 5 页，章节 4.2，页内块 3，字符 [0, 1079)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-09

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

Which model families and downstream benchmarks are included in the main evaluation?

参考要点：

- RoBERTa、DeBERTa、GPT-2、GPT-3；包括 GLUE、WikiSQL、SAMSum 等任务。

原文定位：

- e1：PDF 第 5 页，章节 5，页内块 6，字符 [0, 730)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-10

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

Which GPU is used for the GPT-2-medium adapter latency measurements and how many trials are averaged?

参考要点：

- 表 1 说明使用 Quadro RTX8000，平均 100 次单次前向测量。

原文定位：

- e1：PDF 第 4 页，章节 Table 1 caption，页内块 3，字符 [0, 463)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-11

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`experiment`；可回答：True；人工复核：`pending`

What GPT-3 training throughput per V100 GPU is reported for full fine-tuning and LoRA?

参考要点：

- 在相同权重分片设置下，完整微调为 32.5 token/s，LoRA 为 43.1 token/s。

原文定位：

- e1：PDF 第 5 页，章节 4.2 footnote 5，页内块 10，字符 [0, 366)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-12

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`limitation`；可回答：True；人工复核：`pending`

What batching limitation arises when task-specific LoRA updates are merged into the base weights?

参考要点：

- 不同任务使用不同合并权重，难以在同一次前向中批处理；可选择不合并并动态选择模块，但存在延迟取舍。

原文定位：

- e1：PDF 第 5 页，章节 4.2，页内块 4，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-13

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`multi_section`；可回答：True；人工复核：`pending`

Does a small task checkpoint remove the need to store the pretrained GPT-3 weights at deployment?

参考要点：

- 不需要为每个任务存完整权重，但部署仍需约 350GB 的基础模型；35MB 是任务增量。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 7，字符 [0, 951)。
- e2：PDF 第 5 页，章节 4.2 footnote 4，页内块 10，字符 [0, 366)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-14

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`multi_section`；可回答：True；人工复核：`pending`

How do inference-weight merging and mixed-task batching create a deployment tradeoff?

参考要点：

- 合并可避免额外推理运算，但不同任务的合并权重不同；不合并有利于动态选模块和混合任务，需权衡延迟。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 12，字符 [0, 369)。
- e2：PDF 第 5 页，章节 4.2，页内块 4，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## lora-15

论文：LoRA: Low-Rank Adaptation of Large Language Models（2106.09685v2）

划分：`dev`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What energy consumption in joules per generated token is reported for r=4 LoRA on GPT-3?

参考要点：

- 该具体能耗指标未报告，不能从显存或 token/s 推算。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：已检索 energy、joule、power、watt 等全文关键词并核查效率实验；需独立确认未报告该设置的能耗。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-01

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What layer, hidden-size, attention-head and parameter counts define BERT base and large?

参考要点：

- base：12 层、768 隐层、12 头、110M 参数；large：24 层、1024 隐层、16 头、340M 参数。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 25，字符 [0, 160)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-02

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What tokenizer vocabulary size and special classification/separator tokens are used?

参考要点：

- 约 30000 WordPiece；CLS 用于序列表示，SEP 区分文本段。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 2，字符 [0, 731)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-03

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What corpora and sizes are used for pretraining?

参考要点：

- BooksCorpus 约 8 亿词，英文 Wikipedia 约 25 亿词。

原文定位：

- e1：PDF 第 5 页，章节 3.1，页内块 9，字符 [0, 520)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-04

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Which tokens contribute to the masked-language-model prediction loss?

参考要点：

- 随机选择 15% 的 WordPiece 位置，只预测这些被选择位置的原词，而非重建整个输入。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 1，字符 [0, 670)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-05

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are the selected MLM positions corrupted to reduce the train-finetune mismatch?

参考要点：

- 选中位置中 80% 换为 MASK、10% 换为随机词、10% 保持原词，缓解微调时无 MASK 的差异。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 3，字符 [0, 704)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-06

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are positive and negative examples constructed for next-sentence prediction?

参考要点：

- 50% 使用真实后续句，50% 使用语料中随机句，分别标注 IsNext 和 NotNext。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 8，字符 [0, 911)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-07

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Are BERT encoder parameters frozen during downstream fine-tuning?

参考要点：

- 不冻结；接入任务输入输出后，端到端微调全部参数。

原文定位：

- e1：PDF 第 3 页，章节 Figure 1 caption，页内块 17，字符 [0, 456)。
- e2：PDF 第 5 页，章节 3.2，页内块 18，字符 [0, 337)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-08

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How is the GLUE classification head attached and trained?

参考要点：

- 将 CLS 最终隐向量接入 K×H 的分类权重，用 softmax 分类损失微调。

原文定位：

- e1：PDF 第 5 页，章节 4.1，页内块 15，字符 [0, 451)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-09

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How does the SQuAD 2.0 model represent a question with no answer span?

参考要点：

- 将无答案的起止位置放在 CLS，并比较无答案分数与最佳非空跨度分数。

原文定位：

- e1：PDF 第 7 页，章节 4.3，页内块 25，字符 [0, 430)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-10

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What pretraining batch size and number of optimization steps are used?

参考要点：

- batch 为 256 个序列，训练 100 万步；文中给出每批 256×512 token 的规模。

原文定位：

- e1：PDF 第 13 页，章节 A.2，页内块 44，字符 [0, 140)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-11

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What Adam settings, warmup and weight decay are reported for pretraining?

参考要点：

- 学习率 1e-4，beta1=0.9、beta2=0.999，L2 权重衰减 0.01；前 10000 步预热，随后线性衰减。

原文定位：

- e1：PDF 第 13 页，章节 A.2，页内块 30，字符 [0, 489)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-12

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What TPU configurations and pretraining duration are reported for base and large?

参考要点：

- base 使用 4 个 Cloud TPU（16 芯片），large 使用 16 个（64 芯片）；各训练约 4 天。

原文定位：

- e1：PDF 第 13 页，章节 A.2，页内块 34，字符 [0, 213)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-13

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How does the bidirectional pretraining design connect to the downstream GLUE sequence representation?

参考要点：

- 预训练通过 MLM 使用左右文信息；GLUE 使用 CLS 的最终表示接分类头，并微调。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 1，字符 [0, 670)。
- e2：PDF 第 5 页，章节 4.1，页内块 15，字符 [0, 451)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-14

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How does pretraining differ from fine-tuning in both input corruption and parameter updating?

参考要点：

- 预训练选择位置并按 80/10/10 扰动；微调不使用 MASK 扰动，端到端更新参数。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 3，字符 [0, 704)。
- e2：PDF 第 5 页，章节 3.2，页内块 18，字符 [0, 337)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## bert-15

论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（1810.04805v2）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What total pretraining energy consumption in kWh is reported for BERT large?

参考要点：

- 文中未给出该总能耗，训练芯片数和天数不能替代 kWh。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：已核查预训练附录与 energy、kWh、watt 等全文关键词；需独立确认能耗未报告。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-01

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Which two earlier optimizers motivate Adam and what situations motivate combining them?

参考要点：

- AdaGrad 适合稀疏梯度，RMSProp 适合在线和非平稳目标；Adam 试图结合二者优点。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 10，字符 [0, 1004)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-02

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

What do the first and second running moments estimate, and is the second a centered variance?

参考要点：

- 分别估计梯度均值和梯度平方的二阶原始矩；后者不是中心化方差。

原文定位：

- e1：PDF 第 2 页，章节 2，页内块 12，字符 [0, 745)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-03

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What default learning rate, decay coefficients and epsilon are recommended for the tested problems?

参考要点：

- alpha=0.001，beta1=0.9，beta2=0.999，epsilon=1e-8；是文中测试问题的默认值。

原文定位：

- e1：PDF 第 2 页，章节 Algorithm 1，页内块 2，字符 [0, 207)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-04

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are the running moments initialized and what initial bias does this introduce?

参考要点：

- 两个矩初值为零，早期估计因此向零偏，beta 接近 1 时尤其明显。

原文定位：

- e1：PDF 第 2 页，章节 Algorithm 1，页内块 4，字符 [0, 126)。
- e2：PDF 第 2 页，章节 2，页内块 12，字符 [0, 745)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-05

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Which divisors correct the first and second moment initialization biases?

参考要点：

- 分别除以 1-beta1^t 和 1-beta2^t，再用于参数更新。

原文定位：

- e1：PDF 第 2 页，章节 Algorithm 1，页内块 6，字符 [0, 59)。
- e2：PDF 第 2 页，章节 Algorithm 1，页内块 7，字符 [0, 63)。
- e3：PDF 第 2 页，章节 Algorithm 1，页内块 8，字符 [0, 146)。

核查备注：算法公式已目视核对；原始 PDF 将幂指数拆分在多个文字块中。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-06

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How does the paper distinguish Adam from RMSProp with momentum?

参考要点：

- Adam 直接用梯度的一二阶矩估计更新，RMSProp momentum 对已缩放梯度使用动量；后者缺少同样的初始化偏差修正。

原文定位：

- e1：PDF 第 5 页，章节 5，页内块 2，字符 [0, 730)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-07

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What quantity is used to evaluate the online convex optimization analysis?

参考要点：

- 使用累计 regret，与同一可行集中的最佳固定参数比较累计损失。

原文定位：

- e1：PDF 第 4 页，章节 4，页内块 2，字符 [0, 615)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-08

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What controls are used when comparing optimizers in the empirical study?

参考要点：

- 使用相同的参数初始化，对学习率、动量等做密集网格搜索，报告各自最佳配置。

原文定位：

- e1：PDF 第 5 页，章节 6，页内块 24，字符 [0, 248)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-09

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

How is the MNIST logistic-regression experiment configured and which optimizers are compared?

参考要点：

- L2 正则多类逻辑回归，784 维图像输入，batch=128；比较 Adam、Nesterov SGD 和 AdaGrad。

原文定位：

- e1：PDF 第 5 页，章节 6.1，页内块 27，字符 [0, 338)。
- e2：PDF 第 5 页，章节 6.1，页内块 29，字符 [0, 391)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-10

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

How are sparse IMDB review features constructed and what dropout setting is discussed?

参考要点：

- 使用最高频 10000 词的稀疏词袋；讨论对特征施加 50% dropout 防止过拟合。

原文定位：

- e1：PDF 第 5 页，章节 6.1，页内块 31，字符 [0, 475)。
- e2：PDF 第 6 页，章节 6.1，页内块 23，字符 [0, 410)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-11

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What VAE architecture is used to study the bias-correction ablation?

参考要点：

- 单隐层 500 个 softplus 单元，50 维球形高斯潜变量；改变 beta1、beta2 和学习率。

原文定位：

- e1：PDF 第 8 页，章节 6.4，页内块 9，字符 [0, 879)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-12

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What norm defines AdaMax and which default hyperparameters are given?

参考要点：

- 基于无穷范数；alpha=0.002，beta1=0.9，beta2=0.999。

原文定位：

- e1：PDF 第 9 页，章节 Algorithm 2，页内块 1，字符 [0, 205)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-13

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How does the explanation of initialization bias relate to the VAE ablation when beta2 is close to one?

参考要点：

- 零初始化使早期矩估计偏小；VAE 消融中 beta2 接近 1 且不修正时出现早期不稳定。

原文定位：

- e1：PDF 第 2 页，章节 2，页内块 12，字符 [0, 745)。
- e2：PDF 第 8 页，章节 6.4，页内块 9，字符 [0, 879)。
- e3：PDF 第 8 页，章节 6.4，页内块 10，字符 [0, 481)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-14

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Does the convex regret analysis by itself prove convergence for all deep-network experiments in the paper?

参考要点：

- 不能；理论段讨论凸目标并作有界梯度、参数距离等假设，神经网络实验不能据此得到无条件收敛保证。

原文定位：

- e1：PDF 第 4 页，章节 4，页内块 2，字符 [0, 615)。
- e2：PDF 第 4 页，章节 Theorem 4.1，页内块 16，字符 [0, 367)。
- e3：PDF 第 5 页，章节 6，页内块 23，字符 [0, 327)。

核查备注：仅核对原文的分析范围，不将该早期理论陈述表述为后来已确立的普遍保证。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## adam-15

论文：Adam: A Method for Stochastic Optimization（1412.6980v9）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What GPU energy consumption in joules per MNIST training step is reported for Adam?

参考要点：

- 未报告该设置的每步 GPU 能耗，优化迭代速度不能替代能耗。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：已检索 energy、joule、watt、power 并核查 MNIST 实验，未见该数值指标。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-01

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

How do the authors define internal covariate shift?

参考要点：

- 作者将其定义为训练中参数变化引起的网络激活分布变化；应归属于本文解释。

原文定位：

- e1：PDF 第 2 页，章节 2，页内块 2，字符 [0, 893)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-02

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Why does BatchNorm use per-feature mini-batch statistics instead of full whitening?

参考要点：

- 完整协方差及其逆平方根计算昂贵；逐维均值方差更简单，batch 小于维数时联合协方差还可能奇异。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 11，字符 [0, 568)。
- e2：PDF 第 3 页，章节 3，页内块 12，字符 [0, 810)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-03

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

What do gamma and beta do and how do they help preserve representational capacity?

参考要点：

- 对归一化激活进行可学习的缩放与平移，可恢复原始激活，避免强制归一化限制表达。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 0，字符 [0, 116)。
- e2：PDF 第 3 页，章节 3，页内块 4，字符 [0, 149)。
- e3：PDF 第 3 页，章节 3，页内块 7，字符 [0, 110)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-04

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Why is epsilon added to the mini-batch variance?

参考要点：

- 用于数值稳定。

原文定位：

- e1：PDF 第 3 页，章节 3 / Algorithm 1，页内块 19，字符 [0, 171)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-05

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Why can training-time outputs depend on other batch examples while inference is deterministic?

参考要点：

- 训练使用当前 batch 统计，带来样本间依赖和随机性；推理改用固定总体统计，只依赖当前输入。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 0，字符 [0, 305)。
- e2：PDF 第 4 页，章节 3，页内块 4，字符 [0, 668)。
- e3：PDF 第 5 页，章节 3.4，页内块 20，字符 [0, 495)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-06

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How can the normalization and learned affine transform be combined at inference?

参考要点：

- 固定均值方差后归一化为线性变换，可与 gamma 缩放和 beta 平移合并为一个线性变换。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 9，字符 [0, 447)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-07

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Over which axes is convolutional BatchNorm computed and how many affine parameters are learned?

参考要点：

- 同一特征图跨 batch 和全部空间位置共同归一化，有效样本数 m*p*q；每个特征图一对 gamma、beta。

原文定位：

- e1：PDF 第 5 页，章节 3.2，页内块 18，字符 [0, 800)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-08

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Where is BatchNorm inserted relative to the linear transform and nonlinearity?

参考要点：

- 对 Wu+b 归一化，放在非线性激活之前。

原文定位：

- e1：PDF 第 5 页，章节 3.2，页内块 12，字符 [0, 737)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-09

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Why can the bias before BatchNorm be omitted?

参考要点：

- 减均值会抵消该偏置，其作用可由 beta 承担。

原文定位：

- e1：PDF 第 5 页，章节 3.2，页内块 13，字符 [0, 218)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-10

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What effect on Dropout do the authors report in their BatchNorm experiments?

参考要点：

- 在文中设置中 Dropout 可以去掉或减弱；不是所有网络均可无条件移除。

原文定位：

- e1：PDF 第 5 页，章节 3.4，页内块 20，字符 [0, 495)。
- e2：PDF 第 6 页，章节 4.2.1，页内块 21，字符 [0, 202)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-11

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What hidden layers, activation, training steps and batch size are used in the MNIST demonstration?

参考要点：

- 3 个各 100 单元的全连接隐层，sigmoid；50000 步，batch=60。

原文定位：

- e1：PDF 第 6 页，章节 4.1，页内块 20，字符 [0, 665)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-12

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What training batch size and validation measure are used for the Inception baseline?

参考要点：

- batch=32；使用单裁剪的 ImageNet 验证集 top-1 正确率。

原文定位：

- e1：PDF 第 6 页，章节 4.2，页内块 14，字符 [0, 571)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-13

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Why should the reported fourteen-fold training-step reduction not be attributed to BatchNorm alone?

参考要点：

- 仅加 BN 的 baseline 少于一半步数；14 倍对应 BN-x5，还改变学习率、Dropout、L2 等多项设置。

原文定位：

- e1：PDF 第 6 页，章节 4.2.1，页内块 21，字符 [0, 202)。
- e2：PDF 第 6 页，章节 4.2.1，页内块 23，字符 [0, 264)。
- e3：PDF 第 7 页，章节 4.2.2，页内块 15，字符 [0, 258)。
- e4：PDF 第 7 页，章节 4.2.2，页内块 23，字符 [0, 508)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-14

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`limitation`；可回答：True；人工复核：`pending`

Does independently normalizing features make them jointly uncorrelated?

参考要点：

- 不保证；本文的简化是逐维标准化，明确不要求像完整白化一样去相关。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 22，字符 [0, 426)。
- e2：PDF 第 3 页，章节 3，页内块 40，字符 [0, 199)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## batchnorm-15

论文：Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（1502.03167v3）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What ImageNet top-1 accuracy is reported for BatchNorm trained with a mini-batch size of one?

参考要点：

- 未报告该训练配置的 ImageNet 正确率；不能把推理可用单样本与训练配置混同。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查训练算法 m>1、ImageNet batch=32 与全文 batch 相关段落，未见 batch=1 的该实验结果。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-01

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`limitation`；可回答：True；人工复核：`pending`

Why is the degradation problem distinguished from ordinary overfitting?

参考要点：

- 加深普通网络时训练误差也升高，说明有优化困难，而不仅是验证集泛化变差。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 28，字符 [0, 447)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-02

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How is a desired mapping H(x) rewritten in residual learning?

参考要点：

- 学习 F(x)=H(x)-x，输出 F(x)+x。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 12，字符 [0, 704)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-03

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Do identity shortcuts introduce additional trainable weights?

参考要点：

- 恒等 shortcut 不新增可训练权重，输出通过逐元素相加合并。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 14，字符 [0, 607)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-04

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How can shortcuts handle a change in dimensions?

参考要点：

- 可零填充或用 1×1 投影匹配维度；跨不同空间分辨率时使用步长 2。

原文定位：

- e1：PDF 第 3 页，章节 3.2，页内块 5，字符 [0, 215)。
- e2：PDF 第 4 页，章节 3.3，页内块 29，字符 [0, 749)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-05

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Where is the second ReLU applied in the original residual block?

参考要点：

- 原文设计在残差与 shortcut 相加后施加第二个非线性；不要套用后续 pre-activation 版本。

原文定位：

- e1：PDF 第 3 页，章节 3.2，页内块 1，字符 [0, 230)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-06

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

What is the three-layer bottleneck structure and what do the 1x1 layers do?

参考要点：

- 依次 1×1、3×3、1×1；先降维、再恢复维度，降低中间 3×3 的成本。

原文定位：

- e1：PDF 第 6 页，章节 4.1，页内块 29，字符 [0, 570)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-07

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

How does the ImageNet architecture terminate after convolutional feature extraction?

参考要点：

- 全局平均池化后接 1000 类全连接与 softmax。

原文定位：

- e1：PDF 第 3 页，章节 3.3，页内块 18，字符 [0, 670)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-08

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What optimizer, batch size, initial learning rate, momentum and weight decay are used on ImageNet?

参考要点：

- SGD，batch=256，初始学习率 0.1，momentum=0.9，weight decay=1e-4。

原文定位：

- e1：PDF 第 4 页，章节 3.4，页内块 73，字符 [0, 819)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-09

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What resize and crop augmentation is used in ImageNet training?

参考要点：

- 短边随机取 256–480，随机 224×224 裁剪或水平翻转，并减像素均值。

原文定位：

- e1：PDF 第 4 页，章节 3.4，页内块 73，字符 [0, 819)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-10

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What ImageNet split sizes and error measures are used?

参考要点：

- 约 128 万训练图、5 万验证图、10 万测试图，报告 top-1 与 top-5 错误率。

原文定位：

- e1：PDF 第 4 页，章节 4.1，页内块 111，字符 [0, 338)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-11

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

How is the CIFAR network depth related to n and what are its three feature-map sizes?

参考要点：

- 共 6n+2 个带权层；特征图空间尺寸为 32、16、8。

原文定位：

- e1：PDF 第 7 页，章节 4.2，页内块 12，字符 [0, 649)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-12

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

How is the 110-layer CIFAR model warmed up before using learning rate 0.1?

参考要点：

- 先用 0.01，约 400 次迭代使训练错误率低于 80%，再回到 0.1。

原文定位：

- e1：PDF 第 7 页，章节 4.2，页内块 14，字符 [0, 474)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-13

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Why is parameter-free residual learning useful for a controlled plain-versus-residual comparison?

参考要点：

- 可保持深度、宽度、参数量与主要计算量一致；最初的 ImageNet 对照用恒等和零填充 shortcut，避免额外投影参数。

原文定位：

- e1：PDF 第 3 页，章节 3.2，页内块 4，字符 [0, 408)。
- e2：PDF 第 5 页，章节 4.1，页内块 123，字符 [0, 456)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-14

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Does solving degradation imply that arbitrarily deep CIFAR networks cannot overfit?

参考要点：

- 不意味着；最初退化涉及训练误差，1202 层模型另有过拟合风险，作者认为该小数据集上模型过大。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 28，字符 [0, 447)。
- e2：PDF 第 8 页，章节 4.2，页内块 51，字符 [0, 584)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## resnet-15

论文：Deep Residual Learning for Image Recognition（1512.03385v1）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What top-1 ImageNet error is reported for a 500-layer residual network?

参考要点：

- 本文未报告 500 层 ImageNet 模型；不能挪用 CIFAR 深网络的结果。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查 ImageNet 18/34/50/101/152 层实验与全文 500-layer 关键词，未见所问配置。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-01

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are positive pairs constructed in unsupervised SimCSE?

参考要点：

- 同一句话经同一编码器处理两次，使用独立 dropout 掩码形成不同表示。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 7，字符 [0, 346)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-02

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Where do the negatives in unsupervised SimCSE come from?

参考要点：

- 同一个 mini-batch 中的其他句子作为负例。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 8，字符 [0, 551)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-03

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What happens when dropout is removed or the same dropout mask is used for both views?

参考要点：

- 两种设置使正对表示相同，性能明显下降；文中观察到 alignment 退化。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 33，字符 [0, 301)。
- e2：PDF 第 4 页，章节 3，页内块 39，字符 [0, 927)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-04

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are NLI entailment pairs used instead of a three-way NLI classification objective?

参考要点：

- 将 premise 与 entailment hypothesis 直接用作对比学习正对，不仅训练三分类器。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 15，字符 [0, 735)。
- e2：PDF 第 4 页，章节 4，页内块 34，字符 [0, 521)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-05

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Which NLI relationship supplies hard negatives and how are they associated with a positive pair?

参考要点：

- 同一 premise 对应的 contradiction hypothesis 作为 hard negative。

原文定位：

- e1：PDF 第 5 页，章节 4，页内块 9，字符 [0, 509)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-06

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Which two NLI datasets supply the supervised entailment pairs?

参考要点：

- SNLI 与 MNLI。

原文定位：

- e1：PDF 第 4 页，章节 4，页内块 37，字符 [0, 458)。
- e2：PDF 第 5 页，章节 4，页内块 2，字符 [0, 242)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-07

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What corpus and number of sentences are used for unsupervised training?

参考要点：

- 从英文 Wikipedia 随机抽取 100 万句。

原文定位：

- e1：PDF 第 6 页，章节 6.1，页内块 57，字符 [0, 381)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-08

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Which seven semantic similarity tasks are used in evaluation?

参考要点：

- STS 2012–2016、STS Benchmark 和 SICK-Relatedness。

原文定位：

- e1：PDF 第 6 页，章节 6.1，页内块 50，字符 [0, 526)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-09

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

Which correlation measure, aggregation setting and regression-head convention are used for STS?

参考要点：

- 无额外回归器，使用 Spearman 相关与 all 聚合。

原文定位：

- e1：PDF 第 6 页，章节 6.1，页内块 50，字符 [0, 526)。
- e2：PDF 第 6 页，章节 6.1，页内块 56，字符 [0, 382)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-10

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Does supervised SimCSE train on the STS training sets in this evaluation?

参考要点：

- 不使用 STS 训练集；supervised 指额外 NLI 标注，而非 STS 任务监督。

原文定位：

- e1：PDF 第 6 页，章节 6.1，页内块 18，字符 [0, 391)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-11

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What average BERT-base STS correlations are reported for unsupervised and supervised SimCSE in the introduction?

参考要点：

- 约为 76.3% 与 81.6% Spearman 相关；是论文结果，不是本项目结果。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 47，字符 [0, 695)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-12

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How is the CLS MLP handled differently at test time in unsupervised and supervised SimCSE?

参考要点：

- 无监督版本训练时用 MLP、测试时去掉；有监督默认保留 CLS 上的 MLP。

原文定位：

- e1：PDF 第 8 页，章节 6.3，页内块 28，字符 [0, 877)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-13

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How does dropout-based augmentation relate to the alignment and uniformity observations?

参考要点：

- 独立 dropout 避免正对完全相同导致的 alignment 退化；无监督对比学习改善 uniformity，同时保持较好 alignment。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 39，字符 [0, 927)。
- e2：PDF 第 8 页，章节 7，页内块 26，字符 [0, 675)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-14

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How do NLI supervision and contrastive learning address different aspects of embedding geometry?

参考要点：

- 对比学习将负例拉开以改善均匀性；NLI 正对监督进一步改善语义正对的 alignment。

原文定位：

- e1：PDF 第 5 页，章节 4，页内块 9，字符 [0, 509)。
- e2：PDF 第 5 页，章节 5，页内块 28，字符 [0, 401)。
- e3：PDF 第 8 页，章节 7，页内块 26，字符 [0, 675)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## simcse-15

论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings（2104.08821v4）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

Which Chinese semantic similarity benchmark and numerical score are reported for SimCSE?

参考要点：

- 本文没有报告所问中文基准及分数，不能将英文 STS 结果移用于中文。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查全部数据集说明和 Chinese、Chinese STS 等全文关键词；需独立确认没有中文评测。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-01

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Which two GPU memory levels does FlashAttention organize computation around?

参考要点：

- GPU 的高带宽显存 HBM 与片上 SRAM。

原文定位：

- e1：PDF 第 3 页，章节 2.1，页内块 10，字符 [0, 663)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-02

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`limitation`；可回答：True；人工复核：`pending`

Why can reducing attention FLOPs fail to improve wall-clock time?

参考要点：

- 内存访问等 IO 开销可能成为瓶颈，单看 FLOP 减少不保证真实加速。

原文定位：

- e1：PDF 第 1 页，章节 1，页内块 14，字符 [0, 599)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-03

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How does tiling move Q, K and V through memory?

参考要点：

- 将输入分块，从 HBM 载入 SRAM，在片上完成对应注意力计算，减少中间结果写回。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 18，字符 [0, 484)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-04

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Why does blocked softmax require normalization statistics rather than independent block softmaxes?

参考要点：

- softmax 在列方向耦合；不同块的结果需按正确归一化因子缩放合并，才能得到完整注意力。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 18，字符 [0, 484)。
- e2：PDF 第 4 页，章节 3.1，页内块 19，字符 [0, 197)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-05

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

What is saved for backward recomputation instead of the full attention matrices?

参考要点：

- 保留输出 O 和 softmax 归一化统计 m、ell；反向由 Q、K、V 块重新计算 S、P。

原文定位：

- e1：PDF 第 5 页，章节 3.1，页内块 20，字符 [0, 811)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-06

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How does CUDA kernel fusion reduce additional memory traffic?

参考要点：

- 在同一个 kernel 中完成矩阵乘、softmax、可选 mask/dropout 与后续乘法，减少反复 HBM 读写。

原文定位：

- e1：PDF 第 5 页，章节 3.1，页内块 21，字符 [0, 394)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-07

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Does the original FlashAttention replace exact attention with a sparse approximation?

参考要点：

- 原版计算精确注意力；另行提出的 block-sparse 扩展属于近似注意力。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 41，字符 [0, 1252)。
- e2：PDF 第 2 页，章节 1，页内块 43，字符 [0, 611)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-08

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

Under which SRAM-size range does Theorem 2 compare HBM-access complexity?

参考要点：

- d≤M≤N*d；标准为 Theta(N*d+N^2)，FlashAttention 为 Theta(N^2*d^2/M)。

原文定位：

- e1：PDF 第 6 页，章节 3.2 / Theorem 2，页内块 24，字符 [0, 232)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-09

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What hardware, number of runs and target MLM accuracy are used in the BERT-large speed comparison?

参考要点：

- 8 张 A100，平均 10 次运行，MLM 目标正确率 72.0%；相同 MLPerf 初始化。

原文定位：

- e1：PDF 第 7 页，章节 4.1，页内块 13，字符 [0, 529)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-10

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What happens when GPT-2 context length is increased from 1K to 4K in the long-context experiment?

参考要点：

- 文中 4K FlashAttention 仍比 1K Megatron 快约 30%，困惑度改善约 0.7；比较中上下文长度不同。

原文定位：

- e1：PDF 第 8 页，章节 4.2，页内块 5，字符 [0, 598)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-11

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

Which long sequence lengths are used for Path-X and Path-256, and which variant enables each?

参考要点：

- Path-X 为 16K、由 FlashAttention 支持；Path-256 为 64K、由 block-sparse 版本支持。

原文定位：

- e1：PDF 第 7 页，章节 4，页内块 10，字符 [0, 652)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-12

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

How does increasing tile size affect runtime and what eventually limits it?

参考要点：

- 减少 HBM 访问可降低耗时；块足够大后算术等因素成为瓶颈，且受到 SRAM 容量限制。

原文定位：

- e1：PDF 第 6 页，章节 3.2，页内块 31，字符 [0, 794)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-13

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

How does the memory footprint differ from standard attention without claiming a linear exact-attention arithmetic cost?

参考要点：

- 标准实现将 N×N 中间矩阵写入 HBM；FlashAttention 避免存储，观测内存占用随长度线性增长，但不能据此说精确注意力的算术复杂度线性。

原文定位：

- e1：PDF 第 4 页，章节 2.2，页内块 5，字符 [0, 353)。
- e2：PDF 第 5 页，章节 3.1，页内块 20，字符 [0, 811)。
- e3：PDF 第 7 页，章节 4，页内块 11，字符 [0, 516)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-14

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Why can backward recomputation increase FLOPs yet still accelerate execution?

参考要点：

- 重算避免存取完整中间矩阵，减少 HBM 访问；内存受限条件下额外算术仍可换来更短运行时间。

原文定位：

- e1：PDF 第 3 页，章节 2.1，页内块 14，字符 [0, 286)。
- e2：PDF 第 5 页，章节 3.1，页内块 20，字符 [0, 811)。
- e3：PDF 第 6 页，章节 3.2，页内块 31，字符 [0, 794)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## flashattention-15

论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（2205.14135v2）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What CPU-only speedup over standard attention is measured for FlashAttention?

参考要点：

- 本文未报告所问 CPU-only 加速比，不能挪用 A100 GPU 的实验数据。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查运行时、硬件和附录实验，以及 CPU 关键词，未见 CPU-only 对照测量。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-01

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What does a reliability diagram show for a perfectly calibrated classifier?

参考要点：

- 各置信度区间的正确率应落在置信度等于正确率的对角线上。

原文定位：

- e1：PDF 第 2 页，章节 2，页内块 8，字符 [0, 404)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-02

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How are ECE and MCE distinguished in their treatment of calibration gaps?

参考要点：

- ECE 加权平均各 bin 的差距，MCE 取最大差距；完美校准时均为零。

原文定位：

- e1：PDF 第 3 页，章节 2，页内块 45，字符 [0, 228)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-03

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

How many bins are used for the primary ECE comparison in Table 1?

参考要点：

- M=15。

原文定位：

- e1：PDF 第 6 页，章节 Table 1 caption，页内块 3，字符 [0, 191)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-04

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How does temperature scaling transform logits and how many parameters does it learn?

参考要点：

- 将所有类别 logits 除以同一个正标量 T，再做 softmax；只学习一个参数。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 25，字符 [0, 174)。
- e2：PDF 第 6 页，章节 Equation 9，页内块 28，字符 [0, 9)。
- e3：PDF 第 6 页，章节 Equation 9，页内块 29，字符 [0, 19)。

核查备注：公式和列顺序已目视核对；不能把逆温度误写成固定乘法系数。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-05

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

On which data and objective is the temperature fitted?

参考要点：

- 在保留的验证集上最小化 NLL，不在最终测试集上拟合。

原文定位：

- e1：PDF 第 4 页，章节 4，页内块 18，字符 [0, 396)。
- e2：PDF 第 6 页，章节 4，页内块 4，字符 [0, 467)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-06

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

Why does positive-temperature scaling preserve classification accuracy?

参考要点：

- 同除正 T 不改变 logits 的 argmax，因此类别预测保持不变；改变的是置信度。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 4，字符 [0, 467)。
- e2：PDF 第 6 页，章节 4，页内块 8，字符 [0, 96)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-07

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`fact`；可回答：True；人工复核：`pending`

What does a temperature greater than one do to predicted probabilities?

参考要点：

- 使 softmax 更平缓、熵更大；T 趋于无穷时趋向均匀分布，T=1 恢复原输出。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 4，字符 [0, 467)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-08

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`method`；可回答：True；人工复核：`pending`

How do matrix scaling and vector scaling differ in parameterization?

参考要点：

- matrix scaling 对 logits 做 Wz+b；vector scaling 将 W 限制为对角矩阵。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 15，字符 [0, 224)。
- e2：PDF 第 6 页，章节 4，页内块 23，字符 [0, 260)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-09

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

What relation between increasing depth or width and calibration is observed on CIFAR-100?

参考要点：

- 本文观察到分类误差可降低而 ECE 上升；是实验关联，不应推广为已证明的普遍因果。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 32，字符 [0, 371)。
- e2：PDF 第 3 页，章节 3，页内块 49，字符 [0, 860)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-10

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

How does the BatchNorm experiment separate classification accuracy from calibration?

参考要点：

- 文中 6 层 ConvNet 使用 BN 后正确率略升、校准却变差。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 15，字符 [0, 528)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-11

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`experiment`；可回答：True；人工复核：`pending`

Which dataset is specifically noted as not benefiting from temperature scaling and why?

参考要点：

- Reuters；初始 ECE 已不超过 1%，改善空间有限，也可能受划分或分箱方式影响。

原文定位：

- e1：PDF 第 7 页，章节 5，页内块 12，字符 [0, 455)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-12

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`limitation`；可回答：True；人工复核：`pending`

Why can matrix scaling fail with many classes and limited validation data?

参考要点：

- 参数随类别数平方增长，容易在小验证集过拟合；文中 ImageNet 的 matrix scaling 未收敛。

原文定位：

- e1：PDF 第 8 页，章节 5，页内块 28，字符 [0, 400)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-13

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Does better classification accuracy imply calibrated confidence, and why is a separate calibration procedure needed?

参考要点：

- 不意味着；容量和 BN 实验中正确率与 ECE 可相反变化，校准后处理需用独立验证集估计置信度映射。

原文定位：

- e1：PDF 第 3 页，章节 3，页内块 49，字符 [0, 860)。
- e2：PDF 第 4 页，章节 3，页内块 15，字符 [0, 528)。
- e3：PDF 第 4 页，章节 4，页内块 18，字符 [0, 396)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-14

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`multi_section`；可回答：True；人工复核：`pending`

Why can one-parameter temperature scaling be preferable to a more expressive matrix calibrator?

参考要点：

- 文中温度缩放效果好且参数少；矩阵法参数随类别数平方增长，在有限验证数据上更容易过拟合。

原文定位：

- e1：PDF 第 6 页，章节 4，页内块 25，字符 [0, 174)。
- e2：PDF 第 7 页，章节 5，页内块 8，字符 [0, 696)。
- e3：PDF 第 8 页，章节 5，页内块 28，字符 [0, 400)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## calibration-15

论文：On Calibration of Modern Neural Networks（1706.04599v2）

划分：`test`；类型：`unanswerable`；可回答：False；人工复核：`pending`

What numerical ECE guarantee under adversarial distribution shift is proved for temperature scaling?

参考要点：

- 本文没有给出这种分布漂移下的数值保证；其后处理实验假设训练、验证、测试同分布。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查理论补充与 shift、adversarial 段落；引用其他在线校准工作不等于本文温度缩放的保证。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______
