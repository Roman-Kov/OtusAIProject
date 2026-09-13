# tests/unit/test_chunking.py
from rag_kb.indexing.chunking import split_document
from rag_kb.types import LoadedDoc


def make_doc(text: str, doc_type: str = "text") -> LoadedDoc:
    return LoadedDoc(path="fake.md", text=text, doc_type=doc_type)


def test_long_text_split_with_overlap():
    text = "слово " * 600  # ~3.6k символов
    chunks = split_document(make_doc(text), chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["total_chunks"] == len(chunks)
        assert c.metadata["source"] == "fake.md"
        assert len(c.text) <= 600  # допуск на разделители


def test_short_text_single_chunk():
    chunks = split_document(make_doc("короткий текст"))
    assert len(chunks) == 1
    assert chunks[0].text == "короткий текст"


def test_python_split_prefers_function_boundaries():
    code = "def a():\n    return 1\n\n\n" + "x = 2\n" * 200
    chunks = split_document(make_doc(code, doc_type="python"), chunk_size=400, chunk_overlap=0)
    assert len(chunks) > 1
    assert "def a():" in chunks[0].text


def test_chunk_id_deterministic():
    a = split_document(make_doc("один и тот же текст"))
    b = split_document(make_doc("один и тот же текст"))
    assert a[0].id == b[0].id
