# src/rag_kb/indexing/chunking.py
import hashlib

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag_kb.types import Chunk, LoadedDoc

LANGUAGE_BY_TYPE = {
    "python": Language.PYTHON,
    "js": Language.JS,
    "ts": Language.TS,
    "markdown": Language.MARKDOWN,
}


def _splitter(doc_type: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    lang = LANGUAGE_BY_TYPE.get(doc_type)
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_document(doc: LoadedDoc, chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    """Разбить документ на чанки: для кода — по границам функций/классов,
    для текста — по абзацам."""
    parts = _splitter(doc.doc_type, chunk_size, chunk_overlap).split_text(doc.text)
    source = str(doc.path)
    chunks = []
    for i, text in enumerate(parts):
        chunk_id = hashlib.sha1(f"{source}:{i}".encode()).hexdigest()
        chunks.append(Chunk(id=chunk_id, text=text, metadata={
            "source": source,
            "chunk_index": i,
            "total_chunks": len(parts),
            "doc_type": doc.doc_type,
        }))
    return chunks
