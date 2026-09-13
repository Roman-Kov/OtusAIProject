# src/rag_kb/indexing/loaders.py
from pathlib import Path

from langchain_community.document_loaders import TextLoader

from rag_kb.types import LoadedDoc

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml"}

DOC_TYPE_BY_EXT = {
    ".md": "markdown",
    ".txt": "text",
    ".py": "python",
    ".js": "js",
    ".ts": "ts",
    ".json": "json",
    ".yaml": "yaml",
}


def scan_folder(folder: Path, pattern: str = "**/*") -> list[Path]:
    """Найти все поддерживаемые файлы в папке по glob-паттерну."""
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_file(path: Path) -> LoadedDoc:
    """Прочитать поддерживаемый файл как текст (через TextLoader из LangChain)."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format {ext} is not supported: {path}")
    text = TextLoader(str(path), encoding="utf-8").load()[0].page_content
    return LoadedDoc(path=path, text=text, doc_type=DOC_TYPE_BY_EXT[ext])
