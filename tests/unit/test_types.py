# tests/unit/test_types.py
from rag_kb.types import Chunk, IndexReport, IndexStats, LoadedDoc


def test_dataclasses_hold_values(tmp_path):
    doc = LoadedDoc(path=tmp_path / "a.md", text="hello", doc_type="markdown")
    chunk = Chunk(id="x", text="hello", metadata={"source": "a.md", "chunk_index": 0,
                                                  "total_chunks": 1, "doc_type": "markdown"})
    report = IndexReport(files=1, chunks=1, seconds=0.1, errors=[])
    stats = IndexStats(files=1, chunks=1, last_indexed_at=None)
    assert (doc.text, chunk.id, report.files, stats.chunks) == ("hello", "x", 1, 1)
