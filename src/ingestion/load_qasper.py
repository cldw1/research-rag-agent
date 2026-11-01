
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 项目二：科研 RAG Agent
# P2-02：下载并查看 QASPER 数据集
# 修复新版 datasets 不支持旧版加载脚本的问题
# ==========================================


def main():
    # 1. 获取项目根目录
    project_root = Path(__file__).resolve().parents[2]

    # 2. 设置数据缓存目录
    cache_dir = project_root / "data" / "raw" / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("科研 RAG Agent - QASPER 数据集读取")
    print("=" * 60)

    # 3. 使用 Hugging Face 提供的 Parquet 数据文件
    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    data_files = {
        "train": f"{base_url}/train/0000.parquet",
        "validation": f"{base_url}/validation/0000.parquet",
        "test": f"{base_url}/test/0000.parquet",
    }

    # 4. 下载并加载数据集
    print("\n[1] 正在加载 QASPER 数据集...")
    print("首次运行需要联网下载，请耐心等待。")

    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        cache_dir=str(cache_dir)
    )

    print("\n数据集加载成功！")

    # 5. 查看数据集规模
    print("\n[2] 数据集规模")

    total_papers = 0

    for split_name, split_data in dataset.items():
        count = len(split_data)
        total_papers += count

        print(f"{split_name}: {count} 篇论文")

    print(f"总论文数量: {total_papers}")

    # 6. 查看数据字段
    print("\n[3] 数据字段")

    sample = dataset["train"][0]

    for key in sample.keys():
        print(f"- {key}")

    # 7. 查看第一篇论文
    print("\n[4] 第一篇论文的信息")

    print("论文 ID:", sample["id"])
    print("论文标题:", sample["title"])

    abstract = sample.get("abstract") or ""

    print("摘要前 300 个字符:")
    print(abstract[:300])

    # 8. 查看论文章节
    print("\n[5] 论文章节")

    full_text = sample["full_text"]
    section_names = full_text["section_name"]

    print("章节数量:", len(section_names))

    for index, section in enumerate(section_names[:5], start=1):
        print(f"{index}. {section}")

    # 9. 查看真实问题
    print("\n[6] 论文对应的真实问题")

    qas = sample["qas"]
    questions = qas["question"]

    print("该论文的问题数量:", len(questions))

    if questions:
        print("第一个问题:", questions[0])

    print("\n" + "=" * 60)
    print("QASPER 数据读取与初步检查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
