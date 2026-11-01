
from src.generation.context_builder import build_context


def build_strict_context(
    results,
    max_chars=8000
):
    """
    严格控制上下文字符长度。

    返回:
        context: 最终上下文字符串
        sources: 与上下文一致的引用来源
    """

    if max_chars <= 0:
        raise ValueError(
            "max_chars 必须大于 0"
        )

    if not results:
        return "", []

    # ======================================
    # 第一次：正常构建上下文
    # ======================================

    context, sources = build_context(
        results,
        max_chars=max_chars
    )

    if len(context) <= max_chars:
        return context, sources

    # ======================================
    # 第二阶段：逐步收紧内部预算
    # ======================================

    internal_budget = max_chars

    for _ in range(20):

        overflow = len(context) - max_chars

        # 多留 8 个字符，避免格式开销
        internal_budget -= overflow + 8

        if internal_budget <= 0:
            break

        context, sources = build_context(
            results,
            max_chars=internal_budget
        )

        if len(context) <= max_chars:
            return context, sources

    # ======================================
    # 最后保障：逐步减少候选 Chunk
    # ======================================

    for keep_count in range(
        len(results) - 1,
        0,
        -1
    ):

        context, sources = build_context(
            results[:keep_count],
            max_chars=max_chars
        )

        if len(context) <= max_chars:
            return context, sources

    # 如果连第一个 Chunk 都无法完整放入，
    # 不截断原文，避免产生残缺引用。
    return "", []
