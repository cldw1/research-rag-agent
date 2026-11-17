# 评测结果与复现

评测按文本证据检索、真实 PDF 问答、多文献任务和工程功能分别组织。默认检索采用自适应分块、BM25 + BGE/FAISS、等权 RRF（常数 60）与 Anchor-First。各项对照保留固定配置、逐题结果和适用范围。

## 数据与指标

| 数据 | 规模 | 用途 |
|---|---|---|
| QASPER Test | 建索引 416 篇，纳入评分 400 篇、1216 题 | BM25、Dense、Hybrid 与证据组织对照 |
| QASPER Validation | 建索引 281 篇，纳入评分 274 篇、841 题 | 融合参数、分块策略对照 |
| 指定 HMC PDF | 1 篇、50 页、247 个 Chunk、7 题 | 章节感知排序专项实验 |
| 真实 PDF 语料 | 10 篇、150 道候选题、205 处证据定位 | 45 道开发题与 105 道按论文预留的测试题 |
| 多文献开发题 | 3 篇论文、12 题，其中 10 题有可评分 Gold | 跨文献证据覆盖、回答与来源归属 |

QASPER 使用数据集标注并将 Gold 映射到原始段落，仅纳入符合固定规则的可回答文本证据题。真实 PDF 参考标注经过 AI 原文核查，保留独立人工复核状态。多文献题目来自同一开发语料，不与单论文开发题重复计算独立样本量。

检索记录 Precision@K、Recall@K、Hit@K、MRR@K、nDCG@K；来源段落覆盖与完整段落覆盖分别计算。回答审核区分任务完成、证据支持、引用支持和来源归属。AI 审核结果按其审核口径解释。

## QASPER 检索基线

| 方法 | Hit@5 | 来源 Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 64.72% | 51.04% | 0.3908 | 0.3767 |
| Dense | 66.94% | 54.29% | 0.4279 | 0.4098 |
| Hybrid | 73.77% | 60.05% | 0.4723 | 0.4507 |

Hybrid 相对 Dense 的 Hit@5 增加 6.83 个百分点，按论文聚类的配对 bootstrap 95% 区间为 [4.42, 9.26] 个百分点。三种方法使用相同分块、问题与论文范围。完整指标、排除规则、运行配置与成本见 [QASPER 协议](benchmark.md) 和 [Test 汇总](../results/qasper_test_summary.json)。

相同证据预算下，Anchor-First 的来源段落 Recall 从 60.05% 增至 64.37%，完整段落 Recall 从 56.02% 增至 60.38%；平均上下文由 5212.7 增至 6613.0 字符，证据 F1 从 0.2163 变为 0.2082。覆盖提升伴随更长上下文与精度权衡。

## 分块与融合参数

同一批 841 道 Validation 问题、相同 Embedding 与检索参数下比较分块方式：

| 分块 | Hybrid Hit@5 | 来源 Recall@5 | Anchor-First 完整段落 Recall | Chunk 数 |
|---|---:|---:|---:|---:|
| Adaptive | 73.72% | 61.86% | 61.37% | 11,956 |
| Structure | 68.25% | 54.00% | 58.70% | 17,588 |

Adaptive 使用 BGE tokenizer，以 480 Token、1200 字符、0 overlap 为预算，按结构合并短段并在段落或句子边界拆长段。完整段落 Recall 的配对差值为 2.66 个百分点，论文级 95% 区间为 [0.64, 4.79] 个百分点。该结论限定于本组模型、参数和数据。

另在同一 Validation 样本比较 RRF 常数与 Dense 权重，保留等权、常数 60 的默认配置。结果、逐题指标与上下文成本分别保存在 [分块汇总](../results/qasper_chunking_validation.json)、[逐题记录](../results/qasper_chunking_validation_per_question.csv)、[上下文记录](../results/qasper_chunking_validation_contexts.csv) 和 [融合参数对照](../results/qasper_fusion_validation.json)。

检索失误按来源缺失、排名靠后、片段不完整和证据扩展问题区分，公开 [失败审计](../results/qasper_failure_audit.json)、[逐题失败样例](../results/qasper_test_bad_cases.json) 与 [案例检查](../results/qasper_case_inspection.json)。

## 真实 PDF 问答

固定版本的 Transformer、RAG、LoRA 三篇论文构成 45 题开发集，其中 42 题有可回答证据、3 题用于缺失信息判断。其余七篇论文的 105 题按论文预留。论文版本、题目、参考要点和原文位置见 [评测集说明](../benchmarks/pdf_qa/README.md) 与 [完整审核表](../benchmarks/pdf_qa/corpus_questions.md)。

42 道可回答题的 Hit@5 分别为 BM25 30/42、Dense 32/42、Hybrid 34/42。检索经过统一解析、结构化分块和 SQLite / FAISS 持久化流程，[开发集结果](../results/pdf_qa_development_summary.json) 保留配置与证据对齐信息。

45 道开发题使用同一 DeepSeek 模型生成回答，使用 102,471 Token。按来源进行的 AI 初审为 38 道完整、5 道部分完成、2 道未完成。结果保留引用有效性、来源支持和人工审核状态，见 [回答汇总](../results/pdf_qa_answer_summary.json)。预留测试题尚未计入成绩。

### 证据不足时补检

同一开发集比较首答与最多一次补检补答。补检保护原有引用、按预算补充候选；无新证据、预算不足或请求失败时保留首答。

固定策略的最终 AI 初审为 41 道完整、4 道部分完成，较首答有 3 题改善、0 题退步。8 个已返回补答使用 20,339 Token；另有 1 条请求状态未知并采用首答。此结果对应开发集普通 RAG 流程，在线 Agent 的消息组织与工具行为单独检查。

公开 [补检协议](../benchmarks/pdf_qa/retry_protocol_v2.json)、[触发计划](../results/pdf_qa_retry_v2_plan.json)、[最终汇总](../results/pdf_qa_retry_v2_summary.json)，并保留 [另一组补检对照](../results/pdf_qa_retry_summary.json) 供核查。

## 多文献检索与回答

12 道开发题基于 Transformer、RAG、LoRA 的真实索引，包括 10 道有 Gold 的问题和 2 道缺失信息问题；有 Gold 的问题中包含混合缺失场景。题目与指标见 [固定协议](../benchmarks/multidoc_qa/protocol.json)、[问题与参考证据](../benchmarks/multidoc_qa/questions.json)。

| 检索对照 | 目标论文出现率 | 完整证据 Recall | 全部 Gold 覆盖 |
|---|---:|---:|---:|
| 全局排名 | 77.78% | 32.50% | 见逐题记录 |
| 按论文分配证据 | 100% | 44.17% | 2/10 |
| 两路首位优先（BM25 先） | 见逐题记录 | 54.17% | 4/10 |

首位优先与 RRF 的直接 Top-5 对照中有 3 题改善、1 题退步，变化集中在 LoRA 相关证据。界面提供显式选择，默认仍用 RRF。查询规划另有一组对照，使用 4690 Token，完整证据 Recall 为 43.33%，联合检索为 44.17%。所有结果保存在 [多文献检索](../results/multidoc_qa_development.json)、[首位对照](../results/multidoc_head_comparison.json)、[查询规划对照](../results/multidoc_query_comparison.json) 与 [证据诊断](../results/multidoc_stage_audit.json)。

固定 DeepSeek 模型，对 12 题分别使用 RRF 和 BM25 首位优先生成 24 份回答，共使用 54,533 Token。AI 初审完整回答分别为 5/12 与 7/12，逐题 3 题改善、8 题相同、1 题退步。来源归属和实验条件错误单独保留，见 [回答对照](../results/multidoc_answer_comparison.json) 与 [审核标准](../benchmarks/multidoc_qa/answer_rubric.json)。

### 来源与实验条件约束

可选提示要求区分本文实验、背景介绍及跨论文分析，数值保留任务、模型与测量条件。合并引用如 `[1, 2]` 分别关联到两个来源。

在 6 道固定问题上，基础提示与两组约束提示进行了共 24 次真实生成调用，使用 61,123 Token。两组对照的忠实度分别为 4/6 对 2/6、5/6 对 5/6；后一组的引用支持为 5/6 对 4/6，完成度均为 4/6。约束提示未形成稳定的整体收益，因此作为默认关闭的选项保留。配置与逐题审核见 [提示对照协议](../benchmarks/multidoc_qa/answer_policy_protocol.json)、[对照一](../results/answer_policy_v1_comparison.json)、[对照二](../results/answer_policy_v2_comparison.json)。

## 章节检索与工程功能

指定 HMC PDF 的 7 题对照中，Hybrid 与 Section-Aware 的 Hit@5 为 6/7 与 7/7，MRR@5 为 0.786 与 0.929；人工证据充分率均为 6/7。该组实验使用指定文档的章节映射，独立于 QASPER 主基线。方法与逐题来源见 [章节实验](experiments.md)。

[工作流检查](../scripts/acceptance_workflow.py) 覆盖 PDF、DOCX、MD、TXT 入库，两个进程间的索引恢复、会话与引用恢复、Markdown 来源导出、备份还原和索引变更后的会话校验。已记录 4 次 DeepSeek 请求、11,094 Token 的真实调用检查。结果见 [工作流记录](../results/workflow_acceptance.json) 与 [界面检查](../results/workflow_ui_acceptance.json)。

工程测试默认使用固定样本和 Mock。DeepSeek 已进行真实调用检查；其他模型协议通过适配与 Mock 测试验证。OCR 需要本地识别模型与语言数据，相关功能检查与扫描识别质量分别记录。

## 复现入口

在项目根目录使用以下脚本。各脚本的输入、输出和保护参数以 `--help` 为准；模型与数据准备完成后再运行相应实验。生成实验需要有效 Key，会产生供应商费用。

| 任务 | 脚本 |
|---|---|
| QASPER Test 基线 | [run_qasper_benchmark.py](../scripts/run_qasper_benchmark.py) |
| 分块对照 | [compare_chunking.py](../scripts/compare_chunking.py) |
| 融合参数对照 | [compare_rank_fusion.py](../scripts/compare_rank_fusion.py) |
| 获取固定 PDF | [fetch_pdf_benchmark.py](../scripts/fetch_pdf_benchmark.py) |
| PDF 检索与回答 | [run_pdf_benchmark.py](../scripts/run_pdf_benchmark.py)、[run_pdf_answers.py](../scripts/run_pdf_answers.py) |
| PDF 补检 | [run_pdf_retry.py](../scripts/run_pdf_retry.py) |
| 多文献检索与回答 | [run_multidoc_benchmark.py](../scripts/run_multidoc_benchmark.py)、[run_multidoc_answers.py](../scripts/run_multidoc_answers.py) |
| 证据顺序与查询规划 | [compare_branch_heads.py](../scripts/compare_branch_heads.py)、[compare_multidoc_queries.py](../scripts/compare_multidoc_queries.py) |
| 回答约束 | [run_answer_policy_check.py](../scripts/run_answer_policy_check.py) |

结构化产物保留运行配置、模型身份、数据指纹与逐题记录，检查和导出脚本负责重算指标与核对哈希。公开仓库保留协议、题目和汇总；个人文献、运行数据库、索引及完整请求文件由本地管理。
