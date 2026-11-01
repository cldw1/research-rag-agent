
class GlobalCitationRegistry:
    """
    Agent 全局引用注册表。

    同一篇论文中的相同 Chunk，
    无论被检索多少次，都复用同一个引用编号。
    """

    def __init__(self):

        self.chunk_to_number = {}
        self.number_to_source = {}

        self.next_number = 1
        self.duplicate_count = 0

    def register(self, sources):

        citation_map = {}
        new_sources = []
        reused_sources = []

        for source in sources:

            old_number = int(
                source["citation"]
            )

            if old_number in citation_map:
                raise ValueError(
                    "单次检索的引用编号重复"
                )

            paper_id = str(
                source["paper_id"]
            )

            chunk_id = str(
                source["chunk_id"]
            )

            # 同一个 Chunk 的唯一身份
            key = (
                paper_id,
                chunk_id
            )

            if key in self.chunk_to_number:

                # 重复 Chunk：复用已有编号
                global_number = (
                    self.chunk_to_number[key]
                )

                reused_sources.append({
                    **source,
                    "citation": global_number
                })

                self.duplicate_count += 1

            else:

                # 新 Chunk：分配新编号
                global_number = self.next_number

                self.next_number += 1

                updated_source = {
                    **source,
                    "citation": global_number
                }

                self.chunk_to_number[key] = (
                    global_number
                )

                self.number_to_source[
                    global_number
                ] = updated_source

                new_sources.append(
                    updated_source
                )

            citation_map[old_number] = (
                global_number
            )

        return {
            "citation_map": citation_map,
            "new_sources": new_sources,
            "reused_sources": reused_sources
        }

    def get_all_sources(self):

        return list(
            self.number_to_source.values()
        )

    def get_source(self, number):

        return self.number_to_source.get(
            int(number)
        )

    def get_unique_count(self):

        return len(
            self.number_to_source
        )
