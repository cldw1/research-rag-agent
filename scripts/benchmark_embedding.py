import json
import random
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


SAMPLE_SIZE = 128
BATCH_SIZE = 16
RANDOM_SEED = 42


def sample_corpus(corpus_path):

    """
    从知识库随机抽取文本。

    使用蓄水池抽样：
    不需要一次性把所有Chunk加载到内存。
    """

    rng = random.Random(RANDOM_SEED)

    samples = []
    total_count = 0

    with open(corpus_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            text = chunk.get("text", "").strip()

            if not text:
                continue

            total_count += 1

            if len(samples) < SAMPLE_SIZE:

                samples.append(text)

            else:

                index = rng.randint(
                    0,
                    total_count - 1
                )

                if index < SAMPLE_SIZE:
                    samples[index] = text

    return samples, total_count


def main():

    print("=" * 60)
    print("科研 RAG Agent - Embedding CPU 性能测试")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]

    corpus_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    cache_dir = (
        project_root
        / "models"
        / "embedding_cache"
    )

    # 1. 抽取测试数据
    print("\n[1] 抽取测试 Chunk...")

    texts, total_count = sample_corpus(corpus_path)

    print("知识库 Chunk 总数:", total_count)
    print("抽样 Chunk 数量:", len(texts))

    average_length = sum(
        len(text) for text in texts
    ) / len(texts)

    print(
        "样本平均字符数:",
        f"{average_length:.2f}"
    )

    # 2. 加载模型
    print("\n[2] 加载 Embedding 模型...")

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(cache_dir)
    )

    print("模型加载成功！")

    # 3. 预热模型
    print("\n[3] 模型预热...")

    model.encode(
        texts[:2],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print("预热完成！")

    # 4. 正式计时
    print("\n[4] 开始 CPU 编码测试...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    elapsed = time.perf_counter() - start_time

    # 5. 计算性能指标
    print("\n[5] 性能测试结果")

    speed = len(texts) / elapsed

    estimated_seconds = total_count / speed

    estimated_minutes = estimated_seconds / 60

    embedding_dim = embeddings.shape[1]

    estimated_storage_mb = (
        total_count
        * embedding_dim
        * 4
        / (1024 ** 2)
    )

    print("测试 Chunk 数量:", len(texts))

    print(
        "实际耗时:",
        f"{elapsed:.2f} 秒"
    )

    print(
        "平均编码速度:",
        f"{speed:.2f} Chunk/秒"
    )

    print(
        "完整知识库预计编码时间:",
        f"{estimated_minutes:.2f} 分钟"
    )

    print(
        "预计向量存储大小:",
        f"{estimated_storage_mb:.2f} MiB"
    )

    print(
        "向量维度:",
        embedding_dim
    )

    # 6. 保存测试结果
    print("\n[6] 保存性能测试结果...")

    output_path = (
        project_root
        / "results"
        / "embedding_cpu_benchmark.json"
    )

    report = {
        "model": "BAAI/bge-small-en-v1.5",
        "device": "cpu",
        "sample_size": len(texts),
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "total_chunks": total_count,
        "average_sample_length": average_length,
        "elapsed_seconds": elapsed,
        "chunks_per_second": speed,
        "estimated_total_minutes": estimated_minutes,
        "embedding_dimension": embedding_dim,
        "estimated_storage_mib": estimated_storage_mb
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("结果保存位置:", output_path)

    print("\n说明：")
    print("预计时间是根据128个样本推算的。")
    print("实际耗时可能受到CPU、内存和文本长度影响。")

    print("\n" + "=" * 60)
    print("Embedding CPU 性能测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()