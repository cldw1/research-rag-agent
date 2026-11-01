
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


# ==========================================
# 1. 项目根目录
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "results"
    / "repository_audit.json"
)


# ==========================================
# 2. 关键文件
# ==========================================

IMPORTANT_FILES = [
    "app.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "src/generation/rag_pipeline.py",
    "src/retrieval/retrieval_strategy.py",
    "src/retrieval/strategy_proxy.py",
    "src/retrieval/smart_section_expander.py",
]


# ==========================================
# 3. 判断文件是否不宜直接上传
# ==========================================

def classify_path(relative_path):

    path = Path(relative_path)

    parts = {
        part.lower()
        for part in path.parts
    }

    name = path.name.lower()

    # 环境变量、密钥等
    if (
        name == ".env"
        or (
            name.startswith(".env.")
            and name != ".env.example"
        )
        or name.endswith(
            (".pem", ".p12", ".pfx", ".key")
        )
    ):
        return "sensitive"

    # 虚拟环境、缓存、IDE 配置
    if parts & {
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".idea"
    }:
        return "generated"

    if (
        "models" in parts
        and "embedding_cache" in parts
    ):
        return "model_cache"

    if name.endswith(
        (".pyc", ".pyo")
    ):
        return "generated"

    # PDF、压缩包、大型数据需要单独检查
    if name.endswith(
        (".pdf", ".zip", ".7z", ".rar")
    ):
        return "review"

    # 第69步的本地代码备份
    if name == "app_before_step69.py":
        return "backup"

    return None


# ==========================================
# 4. 执行 Git 命令
# ==========================================

def run_git(*arguments):

    try:

        process = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            timeout=30
        )

        stdout = process.stdout.decode(
            "utf-8",
            errors="replace"
        )

        stderr = process.stderr.decode(
            "utf-8",
            errors="replace"
        )

        return (
            process.returncode,
            stdout,
            stderr
        )

    except FileNotFoundError:

        return (
            -1,
            "",
            "系统没有找到 Git 命令"
        )

    except Exception as exc:

        return (
            -1,
            "",
            str(exc)
        )


# ==========================================
# 5. 扫描 Git 已跟踪的文件
# ==========================================

def get_tracked_files():

    code, output, error = run_git(
        "ls-files",
        "-z"
    )

    if code != 0:
        return [], error

    files = [
        name
        for name in output.split("\0")
        if name
    ]

    return files, None


# ==========================================
# 6. 扫描疑似硬编码密钥
# ==========================================

SECRET_PATTERNS = {
    "OpenAI-style token":
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),

    "GitHub token":
        re.compile(r"gh[pousr]_[A-Za-z0-9_]{25,}"),

    "HuggingFace token":
        re.compile(r"hf_[A-Za-z0-9]{25,}"),

    "Private key":
    re.compile(
        r"(?m)^\s*[-]{5}BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY[-]{5}\s*$"
    )
}


def scan_possible_secrets(tracked_files):

    findings = []

    text_extensions = {
        ".py", ".md", ".txt",
        ".json", ".yaml", ".yml",
        ".toml", ".ini", ".cfg",
        ".env", ".sh", ".ps1"
    }

    for relative in tracked_files:

        path = ROOT / relative

        if not path.is_file():
            continue

        if path.stat().st_size > 2_000_000:
            continue

        if (
            path.suffix.lower()
            not in text_extensions
            and path.name not in {
                ".env",
                ".gitignore"
            }
        ):
            continue

        try:

            content = path.read_text(
                encoding="utf-8",
                errors="replace"
            )

        except Exception:
            continue

        for name, pattern in SECRET_PATTERNS.items():

            matches = list(
                pattern.finditer(content)
            )

            for match in matches[:5]:

                line_number = (
                    content.count(
                        "\n",
                        0,
                        match.start()
                    ) + 1
                )

                # 不输出实际密钥内容
                findings.append({
                    "file": relative,
                    "line": line_number,
                    "type": name
                })

    return findings


# ==========================================
# 7. 找出大文件
# ==========================================

def find_large_files(tracked_files):

    large_files = []

    threshold = 10 * 1024 * 1024

    for relative in tracked_files:

        path = ROOT / relative

        if not path.is_file():
            continue

        size = path.stat().st_size

        if size >= threshold:

            large_files.append({
                "file": relative,
                "size_mb": round(
                    size / 1024 / 1024,
                    2
                )
            })

    return sorted(
        large_files,
        key=lambda x: -x["size_mb"]
    )


# ==========================================
# 8. 检查目录
# ==========================================

def check_directories():

    result = {}

    for name in [
        "src",
        "scripts",
        "results",
        "data",
        "models",
        "docs",
        "tests"
    ]:

        path = ROOT / name

        result[name] = path.is_dir()

    return result


# ==========================================
# 9. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("Research RAG Agent - GitHub 仓库检查")
    print("=" * 65)

    # --------------------------------------
    # A. 检查关键文件
    # --------------------------------------

    important_files = {
        name: (ROOT / name).is_file()
        for name in IMPORTANT_FILES
    }

    print("\n[1] 关键文件")

    for name, exists in important_files.items():

        status = "OK" if exists else "MISSING"

        print(
            f"[{status}] {name}"
        )

    # --------------------------------------
    # B. Git 状态
    # --------------------------------------

    code, output, error = run_git(
        "rev-parse",
        "--is-inside-work-tree"
    )

    git_initialized = (
        code == 0
        and output.strip() == "true"
    )

    tracked_files = []

    if git_initialized:

        tracked_files, git_error = (
            get_tracked_files()
        )

    else:

        git_error = error or (
            "当前项目尚未初始化 Git"
        )

    print("\n[2] Git 状态")

    print(
        "已初始化 Git：",
        git_initialized
    )

    print(
        "已跟踪文件数量：",
        len(tracked_files)
    )

    # --------------------------------------
    # C. 危险文件
    # --------------------------------------

    risky = []

    for relative in tracked_files:

        category = classify_path(
            relative
        )

        if category:

            risky.append({
                "file": relative,
                "category": category
            })

    print("\n[3] 需要检查的已跟踪文件")

    if risky:

        for item in risky[:30]:

            print(
                f"[{item['category']}] "
                f"{item['file']}"
            )

    else:

        print("没有发现明显不宜跟踪的文件路径")

    # --------------------------------------
    # D. 大文件
    # --------------------------------------

    large_files = find_large_files(
        tracked_files
    )

    print("\n[4] 大文件")

    if not large_files:

        print(
            "没有发现超过10MB的已跟踪文件"
        )

    for item in large_files:

        print(
            item["file"],
            item["size_mb"],
            "MB"
        )

    # --------------------------------------
    # E. 疑似密钥
    # --------------------------------------

    possible_secrets = scan_possible_secrets(
        tracked_files
    )

    print("\n[5] 疑似密钥")

    print(
        "疑似位置数量：",
        len(possible_secrets)
    )

    for item in possible_secrets[:20]:

        print(
            item["file"],
            "第",
            item["line"],
            "行：",
            item["type"]
        )

    # --------------------------------------
    # F. 生成报告
    # --------------------------------------

    report = {
        "project": "Research RAG Agent",
        "timestamp": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "git_initialized": git_initialized,
        "git_error": git_error,
        "tracked_file_count": len(
            tracked_files
        ),
        "important_files": important_files,
        "directories": check_directories(),
        "tracked_risky_files": risky,
        "large_tracked_files": large_files,
        "possible_secrets": possible_secrets,
        "notes": [
            "This audit does not change files.",
            "Only Git-tracked files are scanned for token patterns.",
            "No automated scan guarantees absence of secrets.",
            "Ignored files may still exist locally.",
            "Gitignore does not remove already committed secrets."
        ]
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("检查完成")
    print("=" * 65)

    print(
        "报告路径：",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
