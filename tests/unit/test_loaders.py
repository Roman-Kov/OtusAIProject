# tests/unit/test_loaders.py
import pytest

from rag_kb.indexing.loaders import load_file, scan_folder


def test_scan_folder_filters_by_extension(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("x=1", encoding="utf-8")
    (tmp_path / "c.exe").write_bytes(b"\x00")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.yaml").write_text("k: v", encoding="utf-8")
    found = scan_folder(tmp_path, "**/*")
    names = sorted(p.name for p in found)
    assert names == ["a.md", "b.py", "d.yaml"]


def test_scan_folder_respects_pattern(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("x", encoding="utf-8")
    assert [p.name for p in scan_folder(tmp_path, "*.md")] == ["a.md"]


@pytest.mark.parametrize("name,text,doc_type", [
    ("a.md", "# Заголовок", "markdown"),
    ("b.txt", "текст", "text"),
    ("c.py", "x = 1", "python"),
    ("d.js", "const x = 1", "js"),
    ("e.ts", "const x: number = 1", "ts"),
    ("f.json", '{"k": 1}', "json"),
    ("g.yaml", "k: v", "yaml"),
])
def test_load_file_supported_formats(tmp_path, name, text, doc_type):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    doc = load_file(p)
    assert doc.doc_type == doc_type
    assert doc.path == p


def test_load_file_rejects_unsupported(tmp_path):
    p = tmp_path / "a.exe"
    p.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="not supported"):
        load_file(p)
