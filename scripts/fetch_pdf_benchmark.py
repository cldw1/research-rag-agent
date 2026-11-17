"""Download only the pinned public PDFs in the benchmark manifest."""

import argparse
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_benchmark import load_dataset, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/pdf_qa/pilot.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pdf_benchmark/sources")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    args.sources.mkdir(parents=True, exist_ok=True)
    for paper in dataset["papers"]:
        path = args.sources / paper["filename"]
        if path.exists():
            if sha256(path.read_bytes()) != paper["pdf_sha256"]:
                raise ValueError(f"Existing PDF differs; refusing to overwrite: {path.name}")
            print(f"Verified existing PDF: {path.name}")
            continue
        url = "https://arxiv.org/pdf/" + paper["arxiv_version"]
        # No arbitrary URLs or local-network destinations from annotation data.
        import re
        if not re.fullmatch(r"\d{4}\.\d{4,5}v\d+", paper["arxiv_version"]) or paper["source_url"] != url:
            raise ValueError("Expected an explicit arXiv revision URL")
        request = urllib.request.Request(url, headers={"User-Agent": "Research-RAG-Agent/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read(20*1024*1024+1)
        if (len(content) > 20*1024*1024 or not content.startswith(b"%PDF")
                or sha256(content) != paper["pdf_sha256"]):
            raise ValueError(f"Downloaded PDF identity differs: {path.name}; do not silently replace labels")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=args.sources, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic exclusive publication; never replace an existing original.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified: {path.name}")


if __name__ == "__main__":
    main()
