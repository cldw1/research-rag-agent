# 多文档对比问题审核表

12 道开发候选题，派生自已有三篇开发论文；不是额外的独立论文样本。参考要点与证据坐标继承自 [原始开发标注](../pdf_qa/development.json)，均待独立人工审核。

## md-01

Compare which components are trained and frozen in RAG task fine-tuning versus LoRA. Explain any stated reason for keeping components fixed.

- rag（rag-04）：训练查询编码器和 BART；文档编码器及索引保持固定。；更新文档编码器需反复重建索引，成本较高。
- lora（lora-01）：冻结原始权重 W0，仅训练低秩更新中的 A 和 B。

## md-02

What role do attention projections play in the Transformer multi-head mechanism, and which two projections receive LoRA in most experiments? Attribute each statement to its paper.

- attention（attention-06）：各头输出拼接后再投影。；可同时关注不同位置的不同表示子空间。
- lora（lora-06）：多数实验仅适配 query 与 value 投影 Wq、Wv。

## md-03

Contrast RAG index replacement at inference with LoRA weight merging at deployment. What does each operation change?

- rag（rag-14）：训练时固定索引以避免反复更新文档编码。；推理时可直接替换非参数记忆以更新知识，无需重新训练参数模型。
- lora（lora-07）：预先将 BA 合并进 W0，以 W0+BA 执行原有矩阵乘法。

## md-04

Distinguish the hardware and training schedule reported for Transformer from the GPU and number of trials used for the LoRA paper adapter-latency measurement. Do not equate training duration with inference latency.

- attention（attention-09）：使用 8 张 P100。；base 10 万步约 12 小时；big 30 万步约 3.5 天。
- lora（lora-10）：表 1 说明使用 Quadro RTX8000，平均 100 次单次前向测量。

## md-05

List the open-domain QA evaluation datasets and metric in RAG, and the model families and downstream benchmarks evaluated in LoRA. Keep the two evaluation settings separate.

- rag（rag-07）：NQ、TriviaQA、WebQuestions、CuratedTrec；指标为 Exact Match。
- lora（lora-09）：RoBERTa、DeBERTa、GPT-2、GPT-3；包括 GLUE、WikiSQL、SAMSum 等任务。

## md-06

Report Transformer base/big English-German BLEU and the LoRA paper GPT-3 full-fine-tuning/LoRA training throughput. Why should these reported quantities not be treated as the same performance measure?

- attention（attention-12）：base 为 27.3，big 为 28.4。
- lora（lora-11）：在相同权重分片设置下，完整微调为 32.5 token/s，LoRA 为 43.1 token/s。

## md-07

State the sequence-length condition in the Transformer self-attention efficiency discussion and the LoRA limitation when merging different task updates for batching. Keep their conditions distinct.

- attention（attention-13）：文中限定 n 小于表示维度 d。；长序列可限制局部邻域，但会增加最大路径长度；这是后续研究建议。
- lora（lora-12）：不同任务使用不同合并权重，难以在同一次前向中批处理；可选择不合并并动态选择模块，但存在延迟取舍。

## md-08

Does RAG retrieval guarantee unbiased factual answers, and does a small LoRA task checkpoint eliminate storage of the base model? Answer separately using each paper.

- rag（rag-13）：不保证；外部知识本身也可能错误或有偏差，生成仍有误用风险。
- lora（lora-13）：不需要为每个任务存完整权重，但部署仍需约 350GB 的基础模型；35MB 是任务增量。

## md-09

Describe the RAG Wikipedia snapshot, passage size, corpus size and index, then the GPT-3 r=4 LoRA training-memory and task-checkpoint sizes. Distinguish retrieval storage from model-adaptation storage.

- rag（rag-06）：2018 年 12 月 Wikipedia；不重叠的 100 词片段，约 2100 万片段。；FAISS MIPS 索引使用 HNSW 近似。
- lora（lora-08）：文中训练显存从 1.2TB 降到 350GB；仅适配 Q/V、r=4 的任务 checkpoint 从约 350GB 降到 35MB。

## md-10

What batch-size-one Transformer big inference latency and RAG-Sequence NQ k=5 mean query latency are reported in milliseconds? If either is unreported, say so separately; do not substitute another measurement.

- attention（attention-15）：本文没有提供该具体推理延迟，应说明无法由文献确定。
- rag（rag-15）：本文未报告该设置的毫秒级平均延迟，应说明无法从论文确定。

## md-11

Report the Transformer training hardware and schedule. Separately, can the LoRA paper supply joules per generated token for GPT-3 r=4? Do not convert the other paper training duration into energy.

- attention（attention-09）：使用 8 张 P100。；base 10 万步约 12 小时；big 30 万步约 3.5 天。
- lora（lora-15）：该具体能耗指标未报告，不能从显存或 token/s 推算。

## md-12

Do the three papers report comparable deployment measurements: Transformer big batch-size-one latency, RAG-Sequence NQ k=5 mean latency, and GPT-3 r=4 LoRA energy per generated token? Identify missing measurements without inventing or ranking them.

- attention（attention-15）：本文没有提供该具体推理延迟，应说明无法由文献确定。
- rag（rag-15）：本文未报告该设置的毫秒级平均延迟，应说明无法从论文确定。
- lora（lora-15）：该具体能耗指标未报告，不能从显存或 token/s 推算。
