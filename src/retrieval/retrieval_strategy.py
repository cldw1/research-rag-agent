
from src.retrieval.section_aware_reranker import (
    constrained_section_rerank
)


SUPPORTED_STRATEGIES = {
    "hybrid",
    "section_aware"
}


def retrieve_with_strategy(
    retriever,
    query,
    paper_id,
    strategy="hybrid",
    top_k=5,
    candidate_k=50
):
    """
    统一检索入口。

    strategy:
        hybrid:
            原始 Hybrid 检索

        section_aware:
            Hybrid + 受约束章节重排序

    retriever 必须提供：
        search(query, paper_id, top_k, candidate_k)

    section_aware 还需要：
        paper_rows
        section_name 元数据

    当前默认保留原始 Hybrid。
    """

    # ======================================
    # 1. 参数校验
    # ======================================

    if strategy not in SUPPORTED_STRATEGIES:

        raise ValueError(
            f"不支持的检索策略：{strategy}"
        )

    if top_k < 1:
        raise ValueError(
            "top_k 必须大于等于1"
        )

    if candidate_k < 1:
        raise ValueError(
            "candidate_k 必须大于等于1"
        )

    if not paper_id:
        raise ValueError(
            "必须指定 paper_id"
        )

    # ======================================
    # 2. 原始 Hybrid 检索
    # ======================================

    baseline = retriever.search(
        query=query,
        paper_id=paper_id,
        top_k=top_k,
        candidate_k=max(
            candidate_k,
            top_k
        )
    )

    if not baseline:
        return []

    # ======================================
    # 3. 默认策略直接返回
    # ======================================

    if strategy == "hybrid":

        return [
            {
                **item,
                "rank": rank,
                "retrieval_strategy": "hybrid"
            }
            for rank, item in enumerate(
                baseline,
                start=1
            )
        ]

    # ======================================
    # 4. 获取论文全部候选数量
    # ======================================

    paper_rows = getattr(
        retriever,
        "paper_rows",
        None
    )

    if (
        not isinstance(paper_rows, dict)
        or paper_id not in paper_rows
    ):

        raise ValueError(
            "section_aware 需要支持 "
            "paper_rows 的 V3 检索器"
        )

    total_chunks = len(
        paper_rows[paper_id]
    )

    if total_chunks == 0:
        return []

    # ======================================
    # 5. 扩大候选池
    # ======================================

    candidates = retriever.search(
        query=query,
        paper_id=paper_id,
        top_k=total_chunks,
        candidate_k=total_chunks
    )

    if not candidates:
        return baseline

    # ======================================
    # 6. 检查章节元数据
    # ======================================

    for item in candidates:

        if not item.get("section_name"):

            raise ValueError(
                "候选 Chunk 缺少 section_name，"
                "请使用带章节信息的 V3 检索器"
            )

    # ======================================
    # 7. 调用第66步的重排序模块
    # ======================================

    reranked = constrained_section_rerank(
        query=query,
        baseline_results=baseline,
        candidates=candidates,
        top_k=top_k
    )

    return [
        {
            **item,
            "retrieval_strategy": "section_aware"
        }
        for item in reranked
    ]
