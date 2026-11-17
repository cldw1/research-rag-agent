"""Numeric citation syntax shared by answer finalization and optional retries."""

import re


def extract_citation_numbers(answer: str) -> list[int]:
    """Read [1], [1, 2] and [1 2]; this does not establish semantic support."""
    return sorted({int(number) for group in re.findall(r'\[([\d,\s]+)\]', answer)
                   for number in re.findall(r'\d+', group)})
