
import re

import pymupdf


MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 80

MIN_CHARS = 450
TARGET_CHARS = 800
MAX_CHARS = 1000


# ==========================================
# 1. 文本规范化
# ==========================================

def normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def compact(text):
    return re.sub(
        r"\s+",
        "",
        str(text)
    )


# ==========================================
# 2. 提取 PDF 各页文本块
# ==========================================

def extract_pdf_pages(pdf_bytes):

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF 不能超过 20 MB")

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("不是有效的 PDF 文件")

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as doc:

        if doc.needs_pass:
            raise ValueError("暂不支持加密 PDF")

        if len(doc) > MAX_PAGES:
            raise ValueError("PDF 不能超过 80 页")

        for page in doc:

            blocks = page.get_text(
                "blocks",
                sort=True
            )

            texts = []

            for block in blocks:

                if len(block) < 7 or block[6] != 0:
                    continue

                text = str(block[4]).strip()

                if normalize(text):
                    texts.append(text)

            pages.append(texts)

    return pages


# ==========================================
# 3. 识别明显的章节标题
# ==========================================

def is_section_heading(line):

    text = normalize(line)

    if not 7 <= len(text) <= 110:
        return False

    # 常见的独立章节标题
    known_headings = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "methodology",
        "experiments",
        "experimental results",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements"
    }

    if text.casefold() in known_headings:
        return True

    # 尝试识别：
    # 5.4 HMC in Practice and Theory
    # 5.4.2 Tuning HMC
    # 2 Related Work
    numbered_heading = re.fullmatch(
        r"\d+(?:\.\d+){0,4}\.?\s+\S.{3,95}",
        text
    )

    if numbered_heading:

        # 减少将普通句子误判为标题
        if text.endswith((".", "?", "!")):
            return False

        return True

    return False


# ==========================================
# 4. 安全切分超长文本
# ==========================================

def split_long_text(text, limit=MAX_CHARS):

    words = normalize(text).split()

    pieces = []
    current = ""

    for word in words:

        if len(word) > limit:

            if current:
                pieces.append(current)
                current = ""

            for i in range(0, len(word), limit):
                pieces.append(
                    word[i:i + limit]
                )

            continue

        candidate = (
            current + " " + word
            if current
            else word
        )

        if len(candidate) > limit:

            if current:
                pieces.append(current)

            current = word

        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


# ==========================================
# 5. V3 分块
# ==========================================

def build_v3_chunks(
    pages,
    paper_id,
    title
):

    chunks = []

    # 当前识别到的章节
    current_section = None

    for page_index, blocks in enumerate(pages):

        buffer = []
        buffer_section = current_section
        chunk_index = 0

        # ----------------------------------
        # 保存当前 Chunk
        # ----------------------------------

        def flush():

            nonlocal buffer
            nonlocal chunk_index

            if not buffer:
                return

            text = " ".join(buffer)

            section_name = (
                buffer_section
                if buffer_section
                else f"Page {page_index + 1}"
            )

            chunk_id = (
                f"{paper_id}_v3"
                f"_s{page_index:03d}"
                f"_p{chunk_index:03d}"
                "_c000"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": title,
                "section_name": section_name,
                "section_index": page_index,
                "paragraph_index": chunk_index,
                "piece_index": 0,
                "page_number": page_index + 1,
                "text": text,
                "char_count": len(text),
                "is_reference": (
                    section_name.casefold()
                    in {"references", "bibliography"}
                )
            })

            chunk_index += 1
            buffer = []

        # ----------------------------------
        # 添加文本
        # ----------------------------------

        def add_text(text):

            pieces = split_long_text(
                text,
                limit=MAX_CHARS
            )

            for piece in pieces:

                current_length = len(
                    " ".join(buffer)
                )

                candidate = (
                    " ".join(buffer + [piece])
                )

                # 接近目标长度时，优先分块
                should_flush = (
                    buffer
                    and (
                        len(candidate) > MAX_CHARS
                        or (
                            current_length >= MIN_CHARS
                            and len(candidate) > TARGET_CHARS
                        )
                    )
                )

                if should_flush:
                    flush()

                buffer.append(piece)

        # ----------------------------------
        # 遍历当前页面的所有文本块
        # ----------------------------------

        for block in blocks:

            body_lines = []

            # 保留原 PDF 行边界用于标题识别
            for raw_line in block.splitlines():

                line = normalize(raw_line)

                if not line:
                    continue

                if is_section_heading(line):

                    # 先保存标题前的正文
                    if body_lines:

                        add_text(
                            " ".join(body_lines)
                        )

                        body_lines = []

                    # 标题前的 Chunk 单独结束
                    flush()

                    current_section = line
                    buffer_section = current_section

                    # 标题本身加入新 Chunk
                    add_text(line)

                else:

                    body_lines.append(line)

            if body_lines:

                add_text(
                    " ".join(body_lines)
                )

        # 每页结束必须保存
        flush()

    return chunks
