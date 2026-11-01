
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MAX_SIZE_MB = 10

TEXT_EXTENSIONS = {
    ".py", ".json", ".md", ".txt",
    ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".ps1", ".sh"
}

SECRET_PATTERNS = {
    "疑似 API Key": re.compile(
        r"sk-[A-Za-z0-9_-]{20,}"
    ),
    "疑似 GitHub Token": re.compile(
        r"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"
    ),
    "疑似 HuggingFace Token": re.compile(
        r"hf_[A-Za-z0-9]{25,}"
    ),
    "私钥": re.compile(
    r"(?m)^\s*[-]{5}BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY[-]{5}\s*$"
    )
}

ASSIGNMENT_PATTERN = re.compile(
    r"""(?i)\b(?:DEEPSEEK_API_KEY|OPENAI_API_KEY|API_KEY|ACCESS_TOKEN|SECRET_KEY)\b\s*[:=]\s*['"]([^'"\n]+)['"]"""
)

PLACEHOLDERS = (
    "your_", "example", "dummy",
    "placeholder", "replace",
    "changeme", "test_", "os.getenv"
)


def get_candidate_files():

    process = subprocess.run(
        [
            "git", "ls-files",
            "--cached", "--others",
            "--exclude-standard", "-z"
        ],
        cwd=ROOT,
        capture_output=True
    )

    if process.returncode != 0:
        raise RuntimeError(
            process.stderr.decode(
                "utf-8", errors="replace"
            )
        )

    names = process.stdout.decode(
        "utf-8", errors="replace"
    ).split("\0")

    return [
        ROOT / name
        for name in names
        if name
    ]


def main():

    print("=" * 60)
    print("Research RAG Agent - GitHub 安全检查")
    print("=" * 60)

    files = get_candidate_files()

    issues = []

    print(f"\n准备检查的文件：{len(files)} 个")

    for path in files:

        relative = path.relative_to(ROOT).as_posix()

        if not path.is_file():
            continue

        if path.is_symlink():
            issues.append((relative, "符号链接需要人工检查"))
            continue

        name = path.name.lower()

        if name == ".env" or (
            name.startswith(".env.")
            and name != ".env.example"
        ):
            issues.append((relative, "敏感环境文件"))
            continue

        size_mb = (
            path.stat().st_size / 1024 / 1024
        )

        if size_mb > MAX_SIZE_MB:
            issues.append((
                relative,
                f"文件较大：{size_mb:.2f} MB"
            ))

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        if size_mb > 2:
            continue

        content = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        for label, pattern in SECRET_PATTERNS.items():

            for match in pattern.finditer(content):

                line = content.count(
                    "\n", 0, match.start()
                ) + 1

                issues.append((
                    relative,
                    f"第 {line} 行：{label}"
                ))

        for match in ASSIGNMENT_PATTERN.finditer(content):

            value = match.group(1).strip()

            if any(
                word in value.lower()
                for word in PLACEHOLDERS
            ):
                continue

            line = content.count(
                "\n", 0, match.start()
            ) + 1

            issues.append((
                relative,
                f"第 {line} 行：疑似硬编码密钥"
            ))

    print("\n检查结果：")

    if issues:

        for filename, problem in issues:

            print(f"[检查] {filename}")
            print(f"       {problem}")

        print(
            "\n发现需要人工核对的文件，"
            "请暂时不要提交。"
        )

    else:

        print("未发现上述规则能识别的明显风险。")
        print("仍需人工检查，不能保证不存在其他秘密信息。")

    print("\n检查结束。")


if __name__ == "__main__":
    main()
