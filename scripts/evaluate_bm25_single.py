import sys
from pathlib import Path

from datasets import load_dataset


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 保证可以导入 src 中的模块
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever


def normalize_text(text):
    """统一大小写与空白字符"""
    return " ".join(text.lower().split())


def is_evidence_match(chunk_text, evidence_text):
    """
    判断检索到的 Chunk 是否匹配人工标注证据。

    当前使用文本包含关系进行初步匹配。
    后续会进一步完善正式评测方法。
    """

    chunk = normalize_text(chunk_text)
    evidence = normalize_text(evidence_text)

    if not chunk or not evidence:
        return False

    # 完整证据出现在 Chunk 中
    if evidence in chunk:
        return True

    # 较长的 Chunk 是完整证据的一部分
    # 适用于一个长段落被拆成多个 Chunk 的情况
    if len(chunk) >= 80 and chunk in evidence:
        return True

    return False


def main():

    print("=" * 60)
    print("科研 RAG Agent - BM25 单问题检索评测")
    print("=" * 60)

    # ==================================
    # 1. 加载 QASPER 训练集
    # ==================================

    print("\n[1] 加载 QASPER 数据集...")

    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(
            PROJECT_ROOT / "data" / "raw" / "hf_cache"
        )
    )

    # ==================================
    # 2. 获取标准问题和证据
    # ==================================

    target_paper_id = "1909.00694"
    target_question = "What is the seed lexicon?"

    paper = next(
        (
            item for item in dataset
            if item["id"] == target_paper_id
        ),
        None
    )

    if paper is None:
        print("未找到目标论文")
        return

    qas = paper["qas"]

    question_index = None

    for i, question in enumerate(qas["question"]):
        if question.strip().lower() == target_question.lower():
            question_index = i
            break

    if question_index is None:
        print("未找到目标问题")
        return

    annotations = qas["answers"][question_index]["answer"]

    gold_evidences = []

    for annotation in annotations:

        if annotation["unanswerable"]:
            continue

        for evidence in annotation["evidence"]:

            if not evidence:
                continue

            # 暂时仅评测普通文本证据
            if evidence.strip().startswith("FLOAT SELECTED"):
                continue

            gold_evidences.append(evidence)

    if not gold_evidences:
        print("该问题没有可用的文本证据")
        return

    print("目标论文:", target_paper_id)
    print("评测问题:", target_question)
    print("人工标注证据数量:", len(gold_evidences))

    # ==================================
    # 3. 构建 BM25 检索器
    # ==================================

    print("\n[2] 构建 BM25 检索器...")

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    retriever = BM25Retriever(corpus_path)

    # ==================================
    # 4. 获取 Top-10 检索结果
    # ==================================

    print("\n[3] 执行检索...")

    results = retriever.search(
        target_question,
        top_k=10
    )

    hit_ranks = []

    for rank, result in enumerate(results, start=1):

        is_hit = False

        if result["paper_id"] == target_paper_id:

            for evidence in gold_evidences:

                if is_evidence_match(
                    result["text"],
                    evidence
                ):
                    is_hit = True
                    break

        if is_hit:
            hit_ranks.append(rank)

        print(
            f"排名 {rank:2d} | "
            f"分数 {result['score']:.4f} | "
            f"证据命中: {'是' if is_hit else '否'}"
        )

    # ==================================
    # 5. 计算 Hit@K
    # ==================================

    print("\n[4] 单问题评测结果")

    for k in [1, 3, 5, 10]:

        hit = int(
            any(rank <= k for rank in hit_ranks)
        )

        print(f"Hit@{k}: {hit}")

    if hit_ranks:
        print("\n首次命中排名:", min(hit_ranks))
    else:
        print("\nTop-10 未命中标准证据")

    print("\n注意：这是单问题、全库检索的初步检查，")
    print("不是最终的 QASPER 测试集评测结果。")

    print("\n" + "=" * 60)
    print("BM25 单问题检索评测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()