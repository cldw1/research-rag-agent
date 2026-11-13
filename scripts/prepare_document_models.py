"""Download only this application's ONNX layout, table and RapidOCR models."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "copilot_models" / "docling")
    parser.add_argument("--language", default="eng+chi_sim")
    parser.add_argument("--formula", action="store_true", help="Also download the extra code/formula model")
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Keep all task downloads in the requested data directory, without changing
    # system environment variables or the existing BGE cache.
    os.environ["HF_HOME"] = str(target.parent / "download_cache")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.object_detection_engine_options import ObjectDetectionEngineType
    from docling.utils.model_downloader import download_hf_model, download_models
    from src.ingestion.docling_parser import rapidocr_language
    from src.ingestion.parsers import ParseOptions, parser_dependencies

    spec = PdfPipelineOptions().layout_options.model_spec
    override = spec.engine_overrides[ObjectDetectionEngineType.ONNXRUNTIME]
    repo, revision = override.repo_id or spec.repo_id, override.revision or spec.revision
    print("Preparing local ONNX layout model…", flush=True)
    download_hf_model(repo_id=repo, revision=revision, local_dir=target / repo.replace("/", "--"), progress=True)
    print("Preparing table and multilingual ONNX OCR models…", flush=True)
    download_models(output_dir=target, with_layout=False, with_tableformer=True,
        with_code_formula=args.formula, with_picture_classifier=False,
        with_rapidocr=True, rapidocr_models=["onnxruntime:" + rapidocr_language(args.language)], progress=True)
    files = []
    for path in sorted(target.rglob("*")):
        relative = path.relative_to(target)
        if not path.is_file() or ".cache" in relative.parts or path.name == "model_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"format": 1, "layout_repo": repo, "layout_revision_requested": revision,
        "ocr_language": rapidocr_language(args.language), "formula": args.formula,
        "dependencies": parser_dependencies(ParseOptions(backend="docling"), "pdf"), "files": files}
    (target / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ready", "files": len(files), "bytes": sum(item["size"] for item in files)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError):
        print("模型准备未完成，请检查 requirements-docling.txt、网络及模型目录；已下载文件保留供重试。", file=sys.stderr)
        raise SystemExit(1) from None
