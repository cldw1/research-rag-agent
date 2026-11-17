# PDF 问答题目与标注审核

本表由已通过来源校验的标注生成。当前为开发候选题，AI 已核对来源，人工复核待完成。

定位页码为 PDF 阅读器从 1 开始的页码。参考答案是评分要点，不是模型生成回答。

## attention-01

论文：Attention Is All You Need（1706.03762v7）

类型：`fact`；可回答：True；人工复核：`pending`

How many encoder layers are used, and what is their output dimension?

参考要点：

- 编码器含 6 层。
- 子层和嵌入输出维度为 512。

原文定位：

- e1：PDF 第 3 页，章节 3.1，页内块 3，字符 [0, 630)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-02

论文：Attention Is All You Need（1706.03762v7）

类型：`fact`；可回答：True；人工复核：`pending`

How many attention heads are used, and what are the key and value dimensions per head?

参考要点：

- 8 个头；每个头的 key 和 value 维度均为 64。

原文定位：

- e1：PDF 第 5 页，章节 3.2.2，页内块 11，字符 [0, 257)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-03

论文：Attention Is All You Need（1706.03762v7）

类型：`fact`；可回答：True；人工复核：`pending`

What are the input/output and inner dimensions of the position-wise feed-forward network?

参考要点：

- 输入输出维度 512，内部维度 2048。

原文定位：

- e1：PDF 第 5 页，章节 3.3，页内块 20，字符 [0, 305)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-04

论文：Attention Is All You Need（1706.03762v7）

类型：`method`；可回答：True；人工复核：`pending`

How does decoder self-attention preserve the autoregressive property?

参考要点：

- 在 softmax 前将不允许的未来位置设为负无穷，阻止访问后续 token。

原文定位：

- e1：PDF 第 5 页，章节 3.2.3，页内块 16，字符 [0, 445)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-05

论文：Attention Is All You Need（1706.03762v7）

类型：`method`；可回答：True；人工复核：`pending`

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

类型：`method`；可回答：True；人工复核：`pending`

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

类型：`method`；可回答：True；人工复核：`pending`

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

类型：`experiment`；可回答：True；人工复核：`pending`

What training data size and tokenization vocabulary were used for English-German and English-French translation?

参考要点：

- 英德约 450 万句对、约 37000 共享 BPE 词表。
- 英法 3600 万句、32000 word-piece 词表。

原文定位：

- e1：PDF 第 7 页，章节 5.1，页内块 6，字符 [0, 591)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-09

论文：Attention Is All You Need（1706.03762v7）

类型：`experiment`；可回答：True；人工复核：`pending`

What hardware and training schedules are reported for the base and big models?

参考要点：

- 使用 8 张 P100。
- base 10 万步约 12 小时；big 30 万步约 3.5 天。

原文定位：

- e1：PDF 第 7 页，章节 5.2，页内块 8，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-10

论文：Attention Is All You Need（1706.03762v7）

类型：`method`；可回答：True；人工复核：`pending`

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

类型：`experiment`；可回答：True；人工复核：`pending`

How were translation checkpoints combined and beam search configured?

参考要点：

- base 平均最后 5 个 checkpoint，big 平均最后 20 个。
- beam size=4，length penalty=0.6；最大输出长度为输入长度加 50，允许提前停止。

原文定位：

- e1：PDF 第 8 页，章节 6.1，页内块 18，字符 [0, 456)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## attention-12

论文：Attention Is All You Need（1706.03762v7）

类型：`experiment`；可回答：True；人工复核：`pending`

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

类型：`limitation`；可回答：True；人工复核：`pending`

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

类型：`source_conflict`；可回答：True；人工复核：`pending`

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

类型：`unanswerable`；可回答：False；人工复核：`pending`

What batch-size-one inference latency in milliseconds is reported for Transformer big?

参考要点：

- 本文没有提供该具体推理延迟，应说明无法由文献确定。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：全文未发现 latency、inference time 或 batch size 对应报告；训练单步耗时不能替代推理延迟。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-01

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`fact`；可回答：True；人工复核：`pending`

Which pretrained models provide the retriever and generator, and how large is the generator?

参考要点：

- 检索器初始化自 DPR；生成器为约 4 亿参数的 BART-large。

原文定位：

- e1：PDF 第 2 页，章节 1，页内块 30，字符 [0, 1075)。
- e2：PDF 第 3 页，章节 2.3，页内块 31，字符 [0, 583)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-02

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`method`；可回答：True；人工复核：`pending`

How do RAG-Sequence and RAG-Token differ in their use of latent documents?

参考要点：

- Sequence 对整段输出使用同一潜在文档；Token 可为不同 token 使用不同文档。

原文定位：

- e1：PDF 第 3 页，章节 2.1，页内块 1，字符 [0, 563)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-03

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`method`；可回答：True；人工复核：`pending`

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

类型：`method`；可回答：True；人工复核：`pending`

Which components are updated during task fine-tuning and which are held fixed, and why?

参考要点：

- 训练查询编码器和 BART；文档编码器及索引保持固定。
- 更新文档编码器需反复重建索引，成本较高。

原文定位：

- e1：PDF 第 4 页，章节 2.4，页内块 1，字符 [0, 398)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-05

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`method`；可回答：True；人工复核：`pending`

Does RAG task fine-tuning require direct labels identifying the document to retrieve?

参考要点：

- 不需要直接标注应检索哪篇文档；使用输入输出对训练。

原文定位：

- e1：PDF 第 3 页，章节 2.4，页内块 33，字符 [0, 194)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-06

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`experiment`；可回答：True；人工复核：`pending`

Which Wikipedia snapshot, passage size, corpus size, and vector index are used?

参考要点：

- 2018 年 12 月 Wikipedia；不重叠的 100 词片段，约 2100 万片段。
- FAISS MIPS 索引使用 HNSW 近似。

原文定位：

- e1：PDF 第 4 页，章节 3，页内块 10，字符 [0, 733)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-07

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`experiment`；可回答：True；人工复核：`pending`

Which four open-domain QA datasets are evaluated, and which metric is reported?

参考要点：

- NQ、TriviaQA、WebQuestions、CuratedTrec；指标为 Exact Match。

原文定位：

- e1：PDF 第 4 页，章节 3.1，页内块 12，字符 [0, 1040)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-08

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`experiment`；可回答：True；人工复核：`pending`

How is FEVER formulated as a generation task, and which labels are used?

参考要点：

- 将类别映射为单个输出 token。
- 三分类为支持、反驳、信息不足，同时评测二分类版本。

原文定位：

- e1：PDF 第 5 页，章节 3.4，页内块 6，字符 [0, 1181)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-09

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`limitation`；可回答：True；人工复核：`pending`

In the retrieval ablation, on which task does BM25 perform best, and what explanation do the authors suggest?

参考要点：

- FEVER；作者推测该任务实体集中，适合词重叠匹配。
- 该解释属于推测，不能表述为已验证因果。

原文定位：

- e1：PDF 第 7 页，章节 4.5，页内块 39，字符 [0, 481)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-10

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`experiment`；可回答：True；人工复核：`pending`

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

类型：`experiment`；可回答：True；人工复核：`pending`

How does retrieving more documents at test time affect open-domain QA for the two RAG variants?

参考要点：

- 文中 RAG-Sequence 随检索文档数增加而改善；RAG-Token 在 10 篇时达到峰值。
- 结论限定本文设置，不能概括成检索越多越好。

原文定位：

- e1：PDF 第 8 页，章节 4.5，页内块 12，字符 [0, 662)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-12

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`method`；可回答：True；人工复核：`pending`

In the Hemingway generation example, how do parametric and non-parametric memories cooperate?

参考要点：

- 检索文档引导书名的起始生成，后续书名可由参数知识补全。

原文定位：

- e1：PDF 第 6 页，章节 4.3，页内块 17，字符 [0, 1338)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-13

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`limitation`；可回答：True；人工复核：`pending`

Does retrieving Wikipedia guarantee factual and unbiased answers according to the broader-impact discussion?

参考要点：

- 不保证；外部知识本身也可能错误或有偏差，生成仍有误用风险。

原文定位：

- e1：PDF 第 10 页，章节 Broader Impact，页内块 2，字符 [0, 737)。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______

## rag-14

论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks（2005.11401v4）

类型：`multi_section`；可回答：True；人工复核：`pending`

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

类型：`unanswerable`；可回答：False；人工复核：`pending`

What average per-query inference latency in milliseconds is reported for RAG-Sequence on NQ at k=5?

参考要点：

- 本文未报告该设置的毫秒级平均延迟，应说明无法从论文确定。

原文定位：

- 未报告所问具体信息；需独立复核全文。

核查备注：核查正文、附录和全文时间关键词，未见该设置的延迟；k 的效果曲线不等于延迟数据。

人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______
