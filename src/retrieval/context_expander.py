
from collections import defaultdict


def expand_anchors_first(
    retrieved,
    all_chunks,
    window=1
):
    """
    Anchor-First 邻居扩展：
    先保留原始检索结果，再补充相邻段落。
    """

    if window < 0:
        raise ValueError("window 不能小于 0")

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in all_chunks
    }

    paragraph_map = defaultdict(list)

    for chunk in all_chunks:

        key = (
            chunk["section_index"],
            chunk["paragraph_index"]
        )

        paragraph_map[key].append(chunk)

    for chunks in paragraph_map.values():

        chunks.sort(
            key=lambda x: x["piece_index"]
        )

    anchors = []
    neighbors = []
    seen_ids = set()

    # 第一阶段：优先保留所有原始 Anchor
    for result in retrieved:

        chunk_id = result["chunk_id"]

        if chunk_id not in chunk_lookup:
            raise ValueError(
                f"找不到 Anchor：{chunk_id}"
            )

        if chunk_id in seen_ids:
            continue

        record = result.copy()
        record["source_role"] = "retrieved"

        anchors.append(record)
        seen_ids.add(chunk_id)

    # 第二阶段：追加相邻段落
    for result in retrieved:

        anchor = chunk_lookup[
            result["chunk_id"]
        ]

        section_index = anchor["section_index"]
        paragraph_index = anchor["paragraph_index"]

        for offset in range(
            -window,
            window + 1
        ):

            if offset == 0:
                continue

            key = (
                section_index,
                paragraph_index + offset
            )

            for neighbor in paragraph_map.get(
                key,
                []
            ):

                chunk_id = neighbor["chunk_id"]

                if chunk_id in seen_ids:
                    continue

                record = neighbor.copy()
                record["source_role"] = "neighbor"

                neighbors.append(record)
                seen_ids.add(chunk_id)

    return anchors + neighbors
