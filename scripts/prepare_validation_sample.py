
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 1. 设置项目目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import build_chunks


# ==========================================
# 2. 实验配置
# ==========================================

RANDOM_SEED = 42
SAMPLE_SIZE = 30

QUESTION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

CHUNK_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30_chunks.jsonl"
)


# ==========================================
# 3. 提取人工文本证据
# ==========================================

def collect_evidence(obj):

    evidence = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if key == "evidence":

                if isinstance(value, str):
                    evidence.append(value)

                elif isinstance(value, list):

                    for item in value:
                        if isinstance(item, str):
                            evidence.append(item)
                        else:
                            evidence.extend(
                                collect_evidence(item)
                            )

            else:
                evidence.extend(
                    collect_evidence(value)
                )

    elif isinstance(obj, list):

        for item in obj:
            evidence.extend(
                collect_evidence(item)
            )

    # 保留普通文本证据
    cleaned = []

    for text in evidence:

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        # 图表证据暂时不参与本次文本 RAG 实验
        if text.startswith("FLOAT SELECTED"):
            continue

        cleaned.append(text)

    return list(dict.fromkeys(cleaned))


# ==========================================
# 4. 读取 Validation 数据
# ==========================================

def load_validation():

    base_url = (
        "https://huggingface.co/datasets/"
        "allenai/qasper/resolve/"
        "refs%2Fconvert%2Fparquet/qasper"
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "validation": (
                f"{base_url}/validation/0000.parquet"
            )
        },
        split="validation",
        cache_dir=str(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "hf_cache"
        )
    )

    return dataset


# ==========================================
# 5. 为一篇论文选择一道问题
# ==========================================

def select_question(paper, rng):

    qas = paper["qas"]

    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers = qas.get("answers", [])

    candidates = []

    n = min(
        len(questions),
        len(question_ids),
        len(answers)
    )

    for i in range(n):

        question = str(questions[i]).strip()
        question_id = str(question_ids[i])

        evidence = collect_evidence(
            answers[i]
        )

        if not question or not evidence:
            continue

        candidates.append({
            "paper_id": str(paper["id"]),
            "question_id": question_id,
            "question": question,
            "gold_evidence": evidence
        })

    if not candidates:
        return None

    return rng.choice(candidates)


# ==========================================
# 6. 创建固定验证集
# ==========================================

def main():

    print("=" * 65)
    print("QASPER 独立验证集准备")
    print("=" * 65)

    rng = random.Random(RANDOM_SEED)

    print("\n[1] 加载 Validation 数据...")

    dataset = load_validation()

    print("Validation 论文数量:", len(dataset))

    paper_indices = list(range(len(dataset)))
    rng.shuffle(paper_indices)

    selected_questions = []
    selected_papers = []
    selected_chunks = []

    print("\n[2] 随机选择论文和问题...")

    for paper_index in paper_indices:

        paper = dataset[paper_index]

        # 每篇论文最多选择一道问题
        selected_question = select_question(
            paper,
            rng
        )

        if selected_question is None:
            continue

        # 使用之前已经实现的分块方法
        chunks = build_chunks(
            paper,
            max_chars=1200
        )

        if not chunks:
            continue

        paper_id = str(paper["id"])

        selected_questions.append(
            selected_question
        )

        selected_papers.append({
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "chunk_count": len(chunks)
        })

        selected_chunks.extend(chunks)

        print(
            f"[{len(selected_questions):02d}/{SAMPLE_SIZE}] "
            f"论文 {paper_id} | "
            f"{len(chunks)} 个 Chunk"
        )

        if len(selected_questions) >= SAMPLE_SIZE:
            break

    if len(selected_questions) < SAMPLE_SIZE:

        raise RuntimeError(
            "满足条件的论文数量不足 30 篇，"
            "请检查数据集结构。"
        )

    # ======================================
    # 7. 保存问题和人工证据
    # ======================================

    print("\n[3] 保存验证集问题...")

    QUESTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "dataset": "QASPER",
        "split": "validation",
        "random_seed": RANDOM_SEED,
        "sample_size": len(selected_questions),
        "selection": (
            "one question with textual evidence "
            "per sampled paper"
        ),
        "papers": selected_papers,
        "questions": selected_questions
    }

    with open(
        QUESTION_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ======================================
    # 8. 保存全部论文 Chunk
    # ======================================

    print("\n[4] 保存验证集 Chunk...")

    CHUNK_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CHUNK_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in selected_chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    # ======================================
    # 9. 输出统计结果
    # ======================================

    paper_ids = {
        item["paper_id"]
        for item in selected_questions
    }

    chunk_ids = [
        item["chunk_id"]
        for item in selected_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("发现重复的 Chunk ID")

    print("\n" + "=" * 65)
    print("Validation 数据准备完成")
    print("=" * 65)

    print("固定随机种子:", RANDOM_SEED)
    print("问题数量:", len(selected_questions))
    print("不同论文数量:", len(paper_ids))
    print("Chunk 总数量:", len(selected_chunks))

    avg_chunks = (
        len(selected_chunks)
        / len(paper_ids)
    )

    print(
        "平均每篇论文 Chunk 数:",
        f"{avg_chunks:.2f}"
    )

    print("\n问题文件:")
    print(QUESTION_OUTPUT)

    print("\nChunk 文件:")
    print(CHUNK_OUTPUT)

    print("\n前 3 道验证集问题:")

    for i, item in enumerate(
        selected_questions[:3],
        start=1
    ):

        print(f"\n{i}. {item['question']}")
        print("论文:", item["paper_id"])
        print(
            "人工证据数量:",
            len(item["gold_evidence"])
        )

    print("\n说明：")
    print("这是全新的 Validation 论文样本。")
    print("当前没有调用 DeepSeek，也没有计算评估成绩。")
    print("人工答案证据不会输入后续检索器。")


if __name__ == "__main__":
    main()
