# QASPER 证据级检索实验

## 实验协议

主实验使用 QASPER 官方 Test split，按题所属论文检索。Train 不参与索引或参数选择；Validation 样例用于开发。协议在检索评分前固定，参数不根据 Test 成绩调整。

- 数据来源：[QASPER 数据卡](https://huggingface.co/datasets/allenai/qasper)、[数据集论文](https://aclanthology.org/2021.naacl-main.365/)。本地 Test 缓存包含 416 篇论文、1451 个问题。
- 纳入条件：至少有一个可回答标注及文本证据；所有纳入的文本证据经 Unicode NFKC、空白规范化后，必须能唯一匹配同篇正文中的完整段落。无答案、仅图表证据、无法匹配和歧义分别记录，不按检索成绩筛选题目。
- 多标注：主 Recall 的 Gold 是所有可回答标注中文本证据段落的并集；补充 Evidence F1 取各参考标注中的最大值。图表证据不进入文本任务。
- 分块：复用 StructuredChunker adaptive，480 BGE Token、1200 字符、0 overlap；标题与正文结构保留。问题、答案、Gold 和 highlighted_evidence 不进入文档向量或 BM25。
- 编码：现有 `BAAI/bge-small-en-v1.5`，CPU、归一化向量、FAISS IndexFlatIP；查询使用应用现有指令前缀。相同 Chunk、模型和范围比较 BM25、Dense、Hybrid，RRF k=60，每路候选 50。
- 主指标：Precision@K、证据段落 Recall@K、Hit@K、MRR@K、nDCG@K，K 为 1、5、10。查询级取平均；95% 区间以论文为单位作 2000 次 bootstrap，随机种子 42，并报告同题方法差值。
- 证据组织：复用同一次 Hybrid Top-5，在相同 7400 字符预算下比较直接 Top-5 与 Anchor-First，记录来源覆盖、全文段落覆盖、Evidence F1、实际证据块数和参考 tokenizer 长度。
- 工程指标：分块、文档编码与索引构建时间；实际查询编码加检索的平均/P50/P95 耗时。缓存构建与新编码分开记录，模型初始化单独统计。

## 指标口径

Chunk 的来源 span 与任一 Gold 段落相交时，记为相关 Chunk。Precision@K 固定除以 K；MRR 和 nDCG 使用相关 Chunk 的名次，nDCG 的理想排序由同一分块中的全部相关 Chunk 数计算。

Recall@K 对覆盖的 Gold 段落去重，不因同一段的多个片段重复计分。`full_paragraph_recall` 额外要求所选片段覆盖该 Gold 段落的全部解析字符；`all_gold_covered` 要求并集中的全部 Gold 段落都完整覆盖。来源命中和完整覆盖是两个不同指标。

Evidence F1 对所选 Chunk 涉及的正文段落集合与每份参考证据集合计算 F1 后取最大值，与官方评测的多参考思路一致。这里评估检索证据，不是生成答案的官方 QASPER leaderboard 分数。

## 数据规模与筛选

| 项目 | 数量 |
|---|---:|
| Test 论文 | 416 |
| 原始问题 | 1451 |
| 评分问题 | 1216 |
| 评分问题所属论文 | 400 |
| 索引 Chunk | 17474 |
| 不可回答 | 79 |
| 无文本证据 | 62 |
| 文本证据无法完整匹配 | 89 |
| 重复文本定位歧义 | 5 |

所有 416 篇 Test 论文独立建立索引，评分题来自其中 400 篇。Train、Validation 与 Test 的论文 ID 集合互不重叠。

## 检索结果

| 方法 | Precision@5 | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | Evidence F1@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.1622 | 0.5104 | 0.6472 | 0.3908 | 0.3767 | 0.1780 |
| Dense | 0.1707 | 0.5429 | 0.6694 | 0.4279 | 0.4098 | 0.2239 |
| Hybrid | 0.1895 | 0.6005 | 0.7377 | 0.4723 | 0.4507 | 0.2163 |

### 论文级 95% bootstrap 区间

| 方法 | Hit@5 | Recall@5 |
|---|---|---|
| BM25 | 64.72% [61.85, 67.66] | 51.04% [48.28, 53.71] |
| Dense | 66.94% [64.17, 69.63] | 54.29% [51.80, 56.80] |
| Hybrid | 73.77% [71.18, 76.29] | 60.05% [57.61, 62.46] |

配对比较使用同一组问题和论文抽样：

| Hybrid − Dense | 平均差值 | 95% 区间 |
|---|---:|---|
| Hit@5（百分点） | 6.8257 | [4.4234, 9.2640] |
| Recall@5（百分点） | 5.7700 | [3.6770, 7.8291] |
| MRR@5 | 0.0444 | [0.0257, 0.0629] |
| nDCG@5 | 0.0410 | [0.0256, 0.0562] |

在该文本证据子集上，Hybrid 提高了来源命中与排序指标；Evidence F1 同时反映所取段落数量与密度，需结合覆盖率观察。K=1、5、10 的完整结果保存在结构化报告中。

## 证据组织对比

| 方法 | 段落 Recall | 全文段落覆盖 | 全部 Gold 完整覆盖 | Evidence F1 | 平均块数 | 平均字符 | BGE 参考 Token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top-5 | 0.6005 | 0.5602 | 0.4301 | 0.2163 | 5.00 | 5212.7 | 1332.1 |
| Anchor-First | 0.6437 | 0.6038 | 0.4712 | 0.2082 | 7.77 | 6613.0 | 1748.5 |

两组使用相同 Hybrid Anchor 和 7400 字符上限。Anchor-First 补充邻段后覆盖率更高，同时上下文更长、Evidence F1 略低，体现覆盖与证据密度的取舍。参考 Token 使用 BGE tokenizer 计数，属于证据长度统计，不是生成 API 的账单用量。

## 失败案例统计

| 类型 | 问题数 |
|---|---:|
| 三种方法均未命中 | 190 |
| BM25 未命中、Dense 命中 | 217 |
| Dense 未命中、BM25 命中 | 190 |
| Dense 命中、Hybrid 未命中 | 73 |
| Hybrid 命中、Dense 未命中 | 156 |

案例文件保存题目 ID、论文 ID 与 Chunk 排名，便于回到官方数据逐题核查，不自动将检索失败归因为解析、公式或语义错误。

这两组失败案例的来源覆盖、原始排名复现及独立 Validation 参数对照，见 [检索误差分析](retrieval_analysis.md)。

## 工程耗时

| 方法 | 平均查询 ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| BM25 | 0.233 | 0.221 | 0.362 |
| Dense | 9.928 | 9.227 | 12.828 |
| Hybrid | 9.324 | 8.931 | 12.251 |

本机 Windows / Python 3.13.9 / CPU 8 线程，模型加载 9.20 秒，分块 13.88 秒，416 篇文档的编码与索引构建合计 483.95 秒，缓存命中 0 次。查询时间包含在线问题编码与检索，排除模型初始化。耗时是本机热模型单次运行观测。

## 公开产物与复现

- [汇总、配置与统计区间](../results/qasper_test_summary.json)
- [逐题 @5 指标 CSV](../results/qasper_test_per_question.csv)：1216 题 × 3 种方法，共 3648 行。
- [失败案例](../results/qasper_test_bad_cases.json)：统计与样例 ID。
- [评测脚本](../scripts/run_qasper_benchmark.py) 和 [校验导出器](../scripts/export_qasper_results.py)。

已有官方 Arrow 缓存时直接运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_qasper_benchmark.py --split test --threads 8
```

也可将官方 Test Parquet 下载到本地后传入；脚本不自动下载数据或模型：

```powershell
.\.venv\Scripts\python.exe scripts/run_qasper_benchmark.py --split test --dataset-file data/qasper-test.parquet --threads 8
```

运行目录保存冻结协议、排除清单、逐题排名和全部 K 的指标。已具备完整 BGE 缓存时，编码和评分全程离线。报告中的源文件 SHA256、题目集合哈希和模型配置指纹用于复核同一输入。

导出前逐项重算所有 K 的均值并验证逐题文件哈希，输出目标存在时拒绝覆盖。每次新实验保存在独立目录，原七题 [章节实验](experiments.md) 作为专项分析保留。

评测范围是单论文内的文本证据检索。Gold 并集、严格匹配子集和来源跨度口径见上文；证据覆盖不等同于生成答案正确率。
