# tests/test_sample_docs.py
from pathlib import Path

import pytest

SAMPLE_DOCS = Path(__file__).parents[1] / "sample_docs"

VERIFICATION_FACTS = [
    "Трон Пепла", "Пепельные Пустоши", "Кальдера", "Исольда", "127 лет", "812 году",
    "Лавовый Дракон", "40-63", "Обсидианный голем", "47 золота", "Пепельная завеса", "25%",
    "Магнус Чёрный Молот", "897 году", "Битва у Трёх Кратеров", "14 октября 903",
    "Сердце Кальдеры", "Регалии Пепла", "Пепельной войны 889-891", "Лиара Ветрокрылая",
    "Пепельный проход", "907 году", "7 миссий", "12 408", "Ночь Обсидиана", "30 ноября",
]


def _all_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in SAMPLE_DOCS.rglob("*") if p.is_file()]
    return "\n".join(parts)


def test_total_size_at_least_500kb():
    total = sum(p.stat().st_size for p in SAMPLE_DOCS.rglob("*") if p.is_file())
    assert total >= 500 * 1024


@pytest.mark.parametrize("fact", VERIFICATION_FACTS)
def test_verification_fact_present(fact):
    assert fact in _all_text()
