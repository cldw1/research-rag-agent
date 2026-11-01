
from collections import defaultdict


class PaperOutlineTool:

    name = "get_paper_outline"

    description = (
        "Get the title and section outline "
        "of a specified research paper. "
        "Use this for questions about the "
        "paper's structure or section headings."
    )

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, paper_id):

        paper_id = str(paper_id)

        paper_data = self.retriever.get_paper_data(
            paper_id
        )

        chunks = paper_data["chunks"]

        if not chunks:
            raise ValueError(
                f"论文不存在或没有内容：{paper_id}"
            )

        verified = bool(getattr(self.retriever, "has_verified_sections", False))
        section_map = defaultdict(
            lambda: {
                "section_name": "",
                "chunk_count": 0
            }
        )

        for chunk in chunks:

            # In verified V3 metadata section_index is a page offset, so
            # multiple headings on one page must not overwrite each other.
            section_index = (
                chunk["section_name"] if verified else int(chunk["section_index"])
            )

            section = section_map[section_index]

            section["section_name"] = chunk.get(
                "section_name", ""
            )

            section["chunk_count"] += 1

        sections = []

        ordered_keys = list(section_map) if verified else sorted(section_map)
        for ordinal, section_index in enumerate(ordered_keys):

            info = section_map[section_index]

            sections.append({
                "section_index": ordinal if verified else section_index,
                "section_name": info["section_name"],
                "chunk_count": info["chunk_count"]
            })

        return {
            "tool_name": self.name,
            "paper_id": paper_id,
            "title": chunks[0].get("title", ""),
            "section_count": len(sections),
            "total_chunks": len(chunks),
            "sections": sections
        }
