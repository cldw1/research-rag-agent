
"""
Research RAG Pipeline

支持：
1. Top-5 Retrieval
2. Top-10 Retrieval
3. 原始 Anchor-First Expansion
4. 已验证 V3 + Section-Aware 的智能邻居扩展

默认行为保持兼容。
"""

import re

from src.retrieval.context_expander import (
    expand_anchors_first
)

from src.retrieval.smart_section_expander import (
    rank_section_neighbors
)

from src.generation.strict_context_builder import (
    build_strict_context
)

from src.generation.context_builder import (
    build_qa_prompt
)

from src.generation.answer_generator import (
    generate_answer
)


# ==========================================
# 智能扩展参数
# ==========================================

SMART_MAX_DISTANCE = 10
SMART_MAX_EXTRA_CHUNKS = 2


class ResearchRAGPipeline:

    def __init__(
        self,
        retriever,
        max_chars=8000
    ):

        self.retriever = retriever
        self.max_chars = max_chars

    # ======================================
    # 1. 是否启用智能章节扩展
    # ======================================

    def _use_smart_section_expansion(self):

        has_sections = bool(
            getattr(
                self.retriever,
                "has_verified_sections",
                False
            )
        )

        strategy = getattr(
            self.retriever,
            "strategy",
            "hybrid"
        )

        return (
            has_sections
            and strategy == "section_aware"
        )

    # ======================================
    # 2. 检查证据是否全部进入上下文
    # ======================================

    def _all_evidence_in_context(
        self,
        items
    ):

        context, sources = build_strict_context(
            items,
            max_chars=self.max_chars
        )

        if not context or not sources:
            return False

        if len(context) > self.max_chars:
            return False

        expected_ids = {
            item["chunk_id"]
            for item in items
        }

        included_ids = {
            item["chunk_id"]
            for item in sources
        }

        return (
            expected_ids == included_ids
        )

    # ======================================
    # 3. 智能邻居扩展
    # ======================================

    def _smart_expand(
        self,
        question,
        retrieved,
        all_chunks
    ):

        # ----------------------------------
        # A. 保留所有原始 Anchor
        # ----------------------------------

        anchors = []
        seen = set()

        for item in retrieved:

            chunk_id = item["chunk_id"]

            if chunk_id in seen:
                continue

            record = item.copy()

            record["source_role"] = (
                "retrieved"
            )

            anchors.append(record)

            seen.add(chunk_id)

        if not anchors:
            return []

        # ----------------------------------
        # B. 生成同章节邻居候选
        # ----------------------------------

        ranked_neighbors = (
            rank_section_neighbors(
                question=question,
                anchors=anchors,
                all_chunks=all_chunks,
                max_distance=SMART_MAX_DISTANCE
            )
        )

        print(
            "[SmartSection] 候选邻居数量：",
            len(ranked_neighbors)
        )

        # ----------------------------------
        # C. 原始证据先占用预算
        # ----------------------------------

        if not self._all_evidence_in_context(
            anchors
        ):

            print(
                "[SmartSection] 原始 Anchor "
                "已超出预算，不新增邻居。"
            )

            return anchors

        selected = list(anchors)

        added = []

        # ----------------------------------
        # D. 依照相关性尝试新增
        # ----------------------------------

        for neighbor in ranked_neighbors:

            if (
                len(added)
                >= SMART_MAX_EXTRA_CHUNKS
            ):
                break

            trial = selected + [neighbor]

            # 检查所有原始证据和新增证据
            # 是否都能在字符预算中完整保留
            if not self._all_evidence_in_context(
                trial
            ):
                continue

            selected.append(neighbor)

            added.append(neighbor)

        # ----------------------------------
        # E. 打印调试信息
        # ----------------------------------

        print(
            "[SmartSection] 原始 Anchor：",
            len(anchors)
        )

        print(
            "[SmartSection] 最终新增邻居：",
            len(added)
        )

        for item in added:

            print(
                "[SmartSection] 新增：",
                item["chunk_id"]
            )

            print(
                "  章节：",
                item["section_name"]
            )

            print(
                "  页面：",
                item.get("page_number")
            )

            print(
                "  得分：",
                item["neighbor_score"]
            )

            print(
                "  公式编号：",
                item["equation_numbers"]
            )

        return selected

    # ======================================
    # 4. 准备检索上下文
    # ======================================

    def prepare(
        self,
        question,
        paper_id,
        method="anchor_first"
    ):
        """
        检索 + 上下文构建 + Prompt 构建。

        不调用 DeepSeek。
        """

        if method not in (
            "top5",
            "top10",
            "anchor_first"
        ):

            raise ValueError(
                f"不支持的方法：{method}"
            )

        # ----------------------------------
        # A. 获取指定论文的所有 Chunk
        # ----------------------------------

        paper_data = (
            self.retriever.get_paper_data(
                paper_id
            )
        )

        all_chunks = paper_data["chunks"]

        # ----------------------------------
        # B. 检索 Top-10
        # ----------------------------------

        top10 = self.retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=10,
            candidate_k=50
        )

        top5 = top10[:5]

        expansion_mode = "none"

        # ----------------------------------
        # C. 选择上下文构建方式
        # ----------------------------------

        if method == "top5":

            candidates = top5

        elif method == "top10":

            candidates = top10

        else:

            if self._use_smart_section_expansion():

                candidates = self._smart_expand(
                    question=question,
                    retrieved=top5,
                    all_chunks=all_chunks
                )

                expansion_mode = (
                    "smart_section"
                )

            else:

                # QASPER、普通 PDF、Hybrid
                # 完全沿用之前的 Anchor-First
                candidates = expand_anchors_first(
                    retrieved=top5,
                    all_chunks=all_chunks,
                    window=1
                )

                expansion_mode = (
                    "anchor_first"
                )

        # ----------------------------------
        # D. 构建带编号的严格上下文
        # ----------------------------------

        context, sources = build_strict_context(
            candidates,
            max_chars=self.max_chars
        )

        if not context or not sources:

            raise RuntimeError(
                "无法构建有效的检索上下文"
            )

        if len(context) > self.max_chars:

            raise RuntimeError(
                "上下文超过字符预算"
            )

        # ----------------------------------
        # E. 禁止跨论文证据
        # ----------------------------------

        for source in sources:

            if (
                str(source["paper_id"])
                != str(paper_id)
            ):

                raise RuntimeError(
                    "发现跨论文检索结果"
                )

        # ----------------------------------
        # F. 构建问答 Prompt
        # ----------------------------------

        prompt = build_qa_prompt(
            question=question,
            context=context
        )

        source_ids = {
            item["chunk_id"]
            for item in sources
        }

        smart_neighbor_ids = [
            item["chunk_id"]
            for item in candidates
            if (
                item.get("neighbor_selection")
                == "smart_section"
                and item["chunk_id"] in source_ids
            )
        ]

        # ----------------------------------
        # G. 返回结果
        # ----------------------------------

        return {
            "paper_id": str(paper_id),
            "question": question,
            "method": method,
            "expansion_mode": expansion_mode,
            "smart_neighbor_ids": (
                smart_neighbor_ids
            ),
            "context_chars": len(context),
            "chunk_count": len(sources),
            "sources": sources,
            "context": context,
            "prompt": prompt
        }

    # ======================================
    # 5. 生成回答
    # ======================================

    def generate(self, prepared):
        """
        使用已经准备好的证据调用 DeepSeek。
        """

        answer = generate_answer(
            prepared["prompt"]
        )

        cited_numbers = sorted({
            int(number)
            for number in re.findall(
                r"\[(\d+)\]",
                answer
            )
        })

        source_map = {
            int(source["citation"]): source
            for source in prepared["sources"]
        }

        valid_citations = [
            source_map[number]
            for number in cited_numbers
            if number in source_map
        ]

        invalid_citations = [
            number
            for number in cited_numbers
            if number not in source_map
        ]

        return {
            **prepared,
            "answer": answer,
            "cited_numbers": cited_numbers,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations
        }

    # ======================================
    # 6. 完整 RAG 流程
    # ======================================

    def run(
        self,
        question,
        paper_id,
        method="anchor_first"
    ):

        prepared = self.prepare(
            question=question,
            paper_id=paper_id,
            method=method
        )

        return self.generate(
            prepared
        )
