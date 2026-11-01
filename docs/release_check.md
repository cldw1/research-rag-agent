# 本地发布质量检查记录

检查日期：2026-11-01。开始时工作区干净，分支为 `main`。检查阶段仅修改本地工作树，没有改写历史或调用付费 LLM API。发布材料与六张截图经作者审查后，已获授权执行本地 Git 提交；未获授权执行 push。

## 修改文件与目的

| 文件 | 修改目的 |
|---|---|
| [README.md](../README.md) | 修复公式和代码块；加入 Mermaid、功能边界、完整安装步骤、数据/模型准备、实验结果与截图区域 |
| [docs/experiments.md](experiments.md) | 记录单篇七题的协议、逐题结果、指标定义、复现条件与局限 |
| [docs/resume_project.md](resume_project.md) | 提供介绍、四条简历经历、技术栈、可验证数据及面试问答 |
| [docs/images/README.md](images/README.md) | 建立真实截图采集与脱敏说明，不引用不存在的图片 |
| [main.png](images/main.png)、[hmc-hybrid-qa.png](images/hmc-hybrid-qa.png)、[agent-tools.png](images/agent-tools.png)、[citations.png](images/citations.png)、[section-aware.png](images/section-aware.png) | 后续收录作者提供的五张原始截图，README 按实际场景展示；HMC 图片不标为普通 PDF |
| [pdf-qa.png](images/pdf-qa.png) | 再次补充的普通 PDF 原图：20 页、487 个 Chunk / 向量，默认 Hybrid 与数字引用，补齐普通 PDF 展示 |
| 本文件 | 保存本次验证范围、修改清单和剩余手动步骤 |
| [requirements.txt](../requirements.txt) | 固定当前验证的直接依赖；移除没有直接导入的 pandas；pytest 分离为开发依赖 |
| [requirements-dev.txt](../requirements-dev.txt) | 提供开发测试安装入口 |
| [.env.example](../.env.example) | 与实际变量 `DEEPSEEK_API_KEY` 对齐，说明本地索引无需密钥 |
| [.gitignore](../.gitignore) | 忽略导出聊天、凭据、日志和索引；仅放行两份公开结果文件 |
| [app.py](../app.py) | 无 Key 可建索引；缺 QASPER 资源可切换 PDF；增加上传预检查、请求超时及聊天输入上限 |
| [src/agent/paper_outline_tool.py](../src/agent/paper_outline_tool.py) | 修复已验证 V3 以页号聚合目录时覆盖同页多个标题的问题 |
| [src/generation/context_builder.py](../src/generation/context_builder.py) | 预算计入分隔符，拒绝非正预算，延迟导入仅演示需要的检索器 |
| [src/generation/answer_generator.py](../src/generation/answer_generator.py) | 将仅独立演示流程使用的检索器导入移到调用位置 |
| [src/retrieval/validation_paper_scoped_retriever.py](../src/retrieval/validation_paper_scoped_retriever.py) | 提前报告缺失文件，校验非空索引与向量维度 |
| [scripts/check_github_safety.py](../scripts/check_github_safety.py) | 发现问题返回非零退出码，覆盖 `.env.example`，保持不输出密钥值 |
| [scripts/export_experiment_summary.py](../scripts/export_experiment_summary.py) | 从作者原报告重算并核验历史指标，导出不含论文原文的结构化摘要 |
| [scripts/evaluate_hmc_v3.py](../scripts/evaluate_hmc_v3.py) | 复用核心检索模块和公开问题的离线重跑入口，不依赖私有中间报告 |
| [results/pdf_v3_section_mapping_report.json](../results/pdf_v3_section_mapping_report.json) | 保留运行定位与统计，仅去除长段原文预览；旧完整版保存在被忽略的本地备份 |
| [results/hmc_v3_public_summary.json](../results/hmc_v3_public_summary.json) | 发布七题的排名、章节、人工分数、汇总和原报告校验和，不含聊天或全文 |
| [pytest.ini](../pytest.ini) | 只收集 tests，临时目录位于本地已忽略缓存 |
| [tests/test_offline_regressions.py](../tests/test_offline_regressions.py) | 原四套测试、策略/代理、Agent 工具循环、邻居预算、引用及跨论文保护 |
| [tests/test_pdf_index.py](../tests/test_pdf_index.py) | 真实 PDF 解析/BM25/FAISS，以及无文本、大小和页数边界；向量为测试夹具 |
| [tests/test_release_artifacts.py](../tests/test_release_artifacts.py) | 指标重算、摘要字段、相对链接、Python 语法、安全扫描行为 |
| [tests/test_app_startup.py](../tests/test_app_startup.py) | 无 Key 且缺 QASPER 索引时的启动与 PDF 模式切换；套接字不可用时明确跳过 |

## 已通过的验证

- 标准沙箱内 `python -m pytest -q`：**35 passed, 1 skipped**。跳过项是 Streamlit AppTest，原因是 Windows 沙箱不允许所需本地套接字；它会在进入事件循环前检测限制。
- 该 Streamlit AppTest 单独在沙箱外运行：**1 passed**。因此 36 个测试用例均已分别验证通过；没有宣称沙箱内一次运行全部通过。
- 原有四个离线脚本均通过；`scripts/check_project_readiness.py` 在作者现有归档上通过全部历史结果一致性检查。
- Python 语法检查覆盖 `app.py`、`src/`、`scripts/`、`tests/` 共 **100 个文件**；没有执行带界面的实验脚本顶层代码。
- 现有 QASPER Validation 索引自检：**30 篇论文、1631 条向量/元数据**，数量和维度匹配，首向量自检相似度为 1.0；断网模式下实际 BGE + FAISS Top-5 搜索通过，结果均属于所选论文。
- 本地 BGE 缓存断网加载与编码通过，输出 384 维向量。
- `pip check` 无依赖冲突；11 项直接运行/开发依赖的固定版本与本地安装元数据一致。没有从零重装环境。
- README 与 docs 的相对链接检查通过；代码块闭合，数学公式采用 GitHub 显示数学语法，架构使用 Mermaid。没有执行 GitHub 在线页面的像素级渲染验证。
- 安全扫描未发现规则可识别的密钥；已确认 `.env`、环境、模型、数据、原始实验报告、备份和默认聊天导出均被忽略，当前已跟踪路径没有这些敏感目录。只公开章节映射和脱敏实验摘要。
- 两份原始历史评估输入报告的 SHA256 与公开摘要一致；原始成绩未改动。

## 未执行与仍有的边界

自动化发布检查没有调用真实 DeepSeek 问答，遵守本次禁止付费 API 的限制；Agent 自动化测试使用模拟模型响应。首次验收时没有指定 HMC PDF，因此没有重新运行真实 HMC V3 检索或重新人工评分。后续作者分两次提供了六张真实界面截图，现已原样收录；这不等于自动化检查重新运行了问答或评测。没有完整删除缓存再安装、下载数据并重建索引，以保留作者已有数据与环境；已检查实际准备脚本并提供准确命令。

六张截图已覆盖 QASPER 主界面、普通 PDF Hybrid 问答、HMC Hybrid 问答、Agent 工具调用、原文追踪及 HMC Section-Aware 选择；普通 PDF 与已验证 HMC V3 已分别标注。HMC 截图中的公式及编号仍需核对原文。没有进行跨平台、生产并发、回答语义忠实度、费用/延迟和大规模跨论文评测。依赖只固定直接包版本，模型与上游数据修订尚未完全锁定。

## 手动发布步骤

1. 审查已收录的六张截图与图注，所需展示场景已补齐，不必为了截图重复运行问答。
2. 使用有权使用且哈希匹配的 HMC PDF，按 [实验复现说明](experiments.md#重新运行检索的条件) 重跑；新结果与历史报告分开保存。
3. 在单独的新环境验证 README 安装步骤；如需验证真实问答，请自行决定并承担 DeepSeek API 调用。
4. 在 GitHub 预览 Mermaid/数学公式，审查下列提交差异。当前本地真实 `.env` 不需要修改。

```powershell
git diff --check
git status --short
git add README.md .env.example .gitignore app.py requirements.txt requirements-dev.txt pytest.ini docs tests src/agent/paper_outline_tool.py src/generation/context_builder.py src/generation/answer_generator.py src/retrieval/validation_paper_scoped_retriever.py scripts/check_github_safety.py scripts/export_experiment_summary.py scripts/evaluate_hmc_v3.py results/pdf_v3_section_mapping_report.json results/hmc_v3_public_summary.json
git diff --cached --stat
git diff --cached
python scripts/check_github_safety.py
git commit -m "Improve release reproducibility, documentation and offline regression coverage"
```

以上为可复用的审查与提交命令。本轮作者已完成审查并授权本地提交；远端推送仍需单独授权。
