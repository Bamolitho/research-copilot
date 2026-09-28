from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / f"{filename}.md"

    return path.read_text(encoding="utf-8")
