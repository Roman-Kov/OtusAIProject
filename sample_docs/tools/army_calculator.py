# -*- coding: utf-8 -*-
"""Калькулятор армий Кальдеры — внутренняя утилита команды дополнения «Трон Пепла».

Модуль объединяет ежедневные задачи балансовой группы: сколько стоит найм и
содержание армии, каковы её суммарное здоровье и ожидаемый урон, чем
заканчивается стычка двух армий по упрощённой модели боя, и во сколько город
обойдётся ремонт обсидиановых големов. Числа существ вшиты в модуль и сверены
с машинной выгрузкой ``sample_docs/data/creature-stats.json`` (сборка 0.9.4);
источник истины — ``sample_docs/balance/creature-stats.md``. При расхождении
верьте выгрузке, а этот файл правьте: он рабочая утилита, а не таблица
канона. Кальдера не прощает рассинхронизации чисел, ибо «город, который
обманывает в счёте, обманывает и в погоде».

Использование из командной строки::

    python sample_docs/tools/army_calculator.py --help
    python sample_docs/tools/army_calculator.py list
    python sample_docs/tools/army_calculator.py list --tier 7
    python sample_docs/tools/army_calculator.py creature lavovyi_drakon
    python sample_docs/tools/army_calculator.py stats --army "molot:10,kleimenyi_golem:3"
    python sample_docs/tools/army_calculator.py cost --army "lavovyi_drakon:1" --weeks 4
    python sample_docs/tools/army_calculator.py battle --attacker "molot:20" --defender "ugolnitsa:25" --seed 903
    python sample_docs/tools/army_calculator.py repair --creature obsidianovyi_golem --hp 30 --orn
    python sample_docs/tools/army_calculator.py economy

Использование как библиотеки::

    from army_calculator import CREATURES, army_stats, recruit_cost, parse_army_spec

    army = parse_army_spec("molot:10,pepelnaia_garpiia:3")
    print(army_stats(army))
    print(recruit_cost(army).effective_gold)

Соглашения, зашитые в модуль (см. раздел «Методика» balance/creature-stats.md):

* урон пишется цифрами через дефис — «40-63», а не словесной формой;
* клейменый слиток обсидиана стоит 250 золота в городе и 180 у эрафийских
  торговцев; эффективная цена драконов считается по городскому курсу;
* ремонт големов — 47 золота за единицу здоровья, скидка специализации
  Орна Тлеющего Круга округляется вверх: 37,6 → 38;
* приросты пирамиды (14, 9, 7, 6, 4, 3, 1 и 2 у Академии) неизменны с
  Гарнизонного устава 908 года и здесь только читаются.

Коды возврата CLI: 0 — успех, 1 — ошибка данных (неизвестное существо,
кривая спецификация армии), 2 — ошибка аргументов командной строки.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "APP_VERSION",
    "INGOT_CITY_RATE_GOLD",
    "INGOT_ERATHIAN_RATE_GOLD",
    "GOLEM_REPAIR_GOLD_PER_HP",
    "ORN_REPAIR_GOLD_PER_HP",
    "CreatureStats",
    "CostSummary",
    "ArmyStats",
    "StackState",
    "BattleReport",
    "CREATURES",
    "creature",
    "find_creature",
    "effective_cost",
    "damage_multiplier",
    "stack_damage",
    "parse_army_spec",
    "army_stats",
    "army_power",
    "recruit_cost",
    "golem_repair_cost",
    "simulate_skirmish",
    "weekly_pyramid_cost",
    "creature_card",
    "render_table",
    "load_creatures",
    "main",
]

__version__ = "0.9.4"

APP_VERSION = __version__

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "creature-stats.json"

INGOT_CITY_RATE_GOLD = 250
INGOT_ERATHIAN_RATE_GOLD = 180
GOLEM_REPAIR_GOLD_PER_HP = 47
GOLEM_REPAIR_UNCHANGED_SINCE = "0.7.3"
ORN_REPAIR_DISCOUNT = 0.2
ORN_REPAIR_GOLD_PER_HP = 38
ORN_REPAIR_NOTE = (
    "47 × 0,8 = 37,6; округление вверх — «лучше переплатить меди, чем недоплатить камню»"
)

DAMAGE_MULTIPLIER_STEP = 0.05
DEFENSE_MULTIPLIER_STEP = 0.025
MAX_DAMAGE_MULTIPLIER = 4.0
MIN_DAMAGE_MULTIPLIER = 0.25

HARPY_DODGE_CHANCE = 0.25
DODGING_CREATURES = frozenset({"pepelnaia_garpiia", "garpiia_zavesnitsa"})

UPGRADE_CORRIDOR = (0.30, 0.45)
WEEKLY_HEADS_TOTAL = 46
WEEKLY_GOLD_TOTAL = 7770

GOLEM_IDS = frozenset({"obsidianovyi_golem", "kleimenyi_golem"})

DWELLING_ORDER: Tuple[str, ...] = (
    "Кресальный навес",
    "Городская кошара",
    "Егерский питомник",
    "Вербовочный двор",
    "Големная печь",
    "Гнездовой навес",
    "Академия огня",
    "Горн зова",
)


@dataclass(frozen=True)
class CreatureStats:
    """Немутуемая карточка существа пирамиды Кальдеры.

    Атрибуты повторяют поля выгрузки ``creature-stats.json`` один в один,
    чтобы сверка модуля с JSON выполнялась простым сравнением словарей.
    Урон хранится и числами (``damage_min``/``damage_max``), и канонической
    строкой отображения ``damage_display`` — с дефисом, как требует
    правило дефисов балансовой группы.

    Attributes:
        id: Латинский идентификатор из JSON (например, ``lavovyi_drakon``).
        name: Русское имя существа (например, «Лавовый Дракон»).
        tier: Уровень пирамиды: целое 1-7 или строка ``"особое"`` для Академии.
        dwelling: Название жилища города.
        attack: Базовая атака без героя и артефактов.
        defense: Базовая защита без героя и артефактов.
        damage_min: Минимальный урон одной головы.
        damage_max: Максимальный урон одной головы.
        damage_display: Каноническая запись урона через дефис («40-63»).
        health: Здоровье одной головы.
        speed: Скорость в бою.
        growth: Недельный прирост жилища по уставу 908 года.
        cost_gold: Цена найма в золоте.
        cost_ingots: Цена найма в клейменых слитках (``None``, кроме драконов).
        effective_cost_gold: Эффективная цена в золоте по городскому курсу слитка.
        ai_value: Внутренний вес отряда для боевого искателя ИИ.
        upgrades_to: Идентификатор апгрейда, если есть.
        upgrade_of: Идентификатор базового существа, если это апгрейд.
        special: Короткая сводка ключевой особенности.
        special_details: Подробности особенности отдельными строками.

    Examples:
        >>> dragon = CreatureStats(
        ...     id="lavovyi_drakon", name="Лавовый Дракон", tier=7,
        ...     dwelling="Горн зова", attack=18, defense=18,
        ...     damage_min=40, damage_max=63, damage_display="40-63",
        ...     health=220, speed=9, growth=1, cost_gold=2400,
        ...     ai_value=4900, cost_ingots=1, effective_cost_gold=2650,
        ... )
        >>> dragon.damage_display
        '40-63'
    """

    id: str
    name: str
    tier: Union[int, str]
    dwelling: str
    attack: int
    defense: int
    damage_min: int
    damage_max: int
    damage_display: str
    health: int
    speed: int
    growth: int
    cost_gold: int
    ai_value: int
    cost_ingots: Optional[int] = None
    effective_cost_gold: Optional[int] = None
    upgrades_to: Optional[str] = None
    upgrade_of: Optional[str] = None
    special: str = ""
    special_details: Tuple[str, ...] = ()

    @property
    def average_damage(self) -> float:
        """Средний урон одной головы: арифметическое середины коридора."""
        return (self.damage_min + self.damage_max) / 2

    def effective_cost(self) -> int:
        """Возвращает эффективную цену найма в золоте по городскому курсу.

        Для существ без слитков равна ``cost_gold``; для драконов —
        ``cost_gold + cost_ingots × 250``. Двойная бухгалтерия дракона
        (сезонный контракт за 72 слитка) — кампанейский слой и сюда не входит.
        """
        if self.effective_cost_gold is not None:
            return self.effective_cost_gold
        ingots = self.cost_ingots or 0
        return self.cost_gold + ingots * INGOT_CITY_RATE_GOLD


CREATURES: Dict[str, CreatureStats] = {
    "iskrovik": CreatureStats(
        id="iskrovik",
        name="Искровик",
        tier=1,
        dwelling="Кресальный навес",
        attack=2,
        defense=1,
        damage_min=1,
        damage_max=2,
        damage_display="1-2",
        health=4,
        speed=7,
        growth=14,
        cost_gold=25,
        ai_value=18,
        upgrades_to="kresalo",
        special="поджиг; сухой рой — прогноз бурь",
        special_details=(
            "поджиг: подожжённый обоз, осадный щит или отряд получает урон огнём и штраф до конца боя",
            "сухой рой: за сутки до пепельной бури искровики покидают квадрат — егеря читают это как прогноз",
        ),
    ),
    "kresalo": CreatureStats(
        id="kresalo",
        name="Кресало",
        tier=1,
        dwelling="Кресальный навес",
        attack=4,
        defense=2,
        damage_min=2,
        damage_max=3,
        damage_display="2-3",
        health=6,
        speed=8,
        growth=14,
        cost_gold=40,
        ai_value=35,
        upgrade_of="iskrovik",
        special="удар жаром; не боится воды",
        special_details=(
            "удар жаром: подожжённый отряд теряет 1 защиты до конца боя",
            "подожжённых нельзя складывать — один отряд горит один раз",
        ),
    ),
    "ugolnitsa": CreatureStats(
        id="ugolnitsa",
        name="Угольница",
        tier=2,
        dwelling="Городская кошара",
        attack=5,
        defense=4,
        damage_min=3,
        damage_max=7,
        damage_display="3-7",
        health=20,
        speed=5,
        growth=9,
        cost_gold=70,
        ai_value=95,
        upgrades_to="zharovaia_ugolnitsa",
        special="живой таран; полевое молоко",
        special_details=(
            "урон 3-7 — самый широкий разброс уровня: таран бьёт то краем, то серединой",
            "первая линия Угольниц работает стеной для Молотов",
        ),
    ),
    "zharovaia_ugolnitsa": CreatureStats(
        id="zharovaia_ugolnitsa",
        name="Жаровая угольница",
        tier=2,
        dwelling="Городская кошара",
        attack=7,
        defense=6,
        damage_min=5,
        damage_max=9,
        damage_display="5-9",
        health=24,
        speed=6,
        growth=9,
        cost_gold=110,
        ai_value=140,
        upgrade_of="ugolnitsa",
        special="таран-загривок; ворота с первого удара",
        special_details=(
            "таранный удар по воротам и осадным сооружениям засчитывается с первого удара",
            "в полевых боях +2 к защите против стрелков",
        ),
    ),
    "pepelnyi_volk": CreatureStats(
        id="pepelnyi_volk",
        name="Пепельный волк",
        tier=3,
        dwelling="Егерский питомник",
        attack=6,
        defense=3,
        damage_min=3,
        damage_max=5,
        damage_display="3-5",
        health=15,
        speed=8,
        growth=7,
        cost_gold=120,
        ai_value=130,
        upgrades_to="dymnyi_volk",
        special="разведка; помнит лица",
        special_details=(
            "память лиц: +2 урона отряду, который уже бил его хозяина-героя в этом месяце",
            "бонус не стакается ни с чем и не действует на отряды под чужими аурами",
        ),
    ),
    "dymnyi_volk": CreatureStats(
        id="dymnyi_volk",
        name="Дымный волк",
        tier=3,
        dwelling="Егерский питомник",
        attack=8,
        defense=4,
        damage_min=4,
        damage_max=6,
        damage_display="4-6",
        health=18,
        speed=9,
        growth=7,
        cost_gold=160,
        ai_value=170,
        upgrade_of="pepelnyi_volk",
        special="«не теряет след»; ведёт сквозь фронты",
        special_details=(
            "отряд со стеком дымных волков движется по пепельным фронтам без штрафа",
            "видит на одну клетку дальше в пепельной мгле",
        ),
    ),
    "molot": CreatureStats(
        id="molot",
        name="Молот",
        tier=4,
        dwelling="Вербовочный двор",
        attack=7,
        defense=9,
        damage_min=6,
        damage_max=9,
        damage_display="6-9",
        health=30,
        speed=5,
        growth=6,
        cost_gold=150,
        ai_value=220,
        upgrades_to="chernyi_molot",
        special="секирный строй; «один удар вместо трёх»",
        special_details=(
            "+1 к урону за каждый ход в строю без перемещения, максимум +3",
            "урон 6-9 взят у Мечника дословно — «секирная школа» эталона",
        ),
    ),
    "chernyi_molot": CreatureStats(
        id="chernyi_molot",
        name="Чёрный молот",
        tier=4,
        dwelling="Вербовочный двор",
        attack=9,
        defense=11,
        damage_min=8,
        damage_max=12,
        damage_display="8-12",
        health=38,
        speed=6,
        growth=6,
        cost_gold=200,
        ai_value=320,
        upgrade_of="molot",
        special="право первого удара",
        special_details=(
            "первый удар срабатывает, только если отряд не двигался в первый ход боя",
            "двигающийся Чёрный молот — просто дорогой Молот; стоящий — стена, которая бьёт первой",
        ),
    ),
    "obsidianovyi_golem": CreatureStats(
        id="obsidianovyi_golem",
        name="Обсидиановый голем",
        tier=5,
        dwelling="Големная печь",
        attack=9,
        defense=12,
        damage_min=10,
        damage_max=14,
        damage_display="10-14",
        health=50,
        speed=5,
        growth=4,
        cost_gold=275,
        ai_value=350,
        upgrades_to="kleimenyi_golem",
        special="ремонт: 47 золота за ХП; «полустекло»: -25% урона от магии",
        special_details=(
            "полустекло: полустеклянная фактура гасит 25% урона от заклинаний",
            "ремонт: 47 золота за единицу здоровья, цена не менялась с 0.7.3",
            "не лечится заклинаниями и не воскрешается — чинится за золото по факту боя",
        ),
    ),
    "kleimenyi_golem": CreatureStats(
        id="kleimenyi_golem",
        name="Клеймёный голем",
        tier=5,
        dwelling="Големная печь",
        attack=11,
        defense=14,
        damage_min=12,
        damage_max=16,
        damage_display="12-16",
        health=60,
        speed=5,
        growth=4,
        cost_gold=375,
        ai_value=480,
        upgrade_of="obsidianovyi_golem",
        special="то же, что базовый; «привычка строя»",
        special_details=(
            "привычка строя: линия с молотами получает +1 к защите",
            "сохраняет ремонт 47 золота за ХП и «полустекло» -25% урона от магии",
        ),
    ),
    "pepelnaia_garpiia": CreatureStats(
        id="pepelnaia_garpiia",
        name="Пепельная гарпия",
        tier=6,
        dwelling="Гнездовой навес",
        attack=10,
        defense=8,
        damage_min=6,
        damage_max=10,
        damage_display="6-10",
        health=25,
        speed=11,
        growth=3,
        cost_gold=350,
        ai_value=460,
        upgrades_to="garpiia_zavesnitsa",
        special="«Пепельная завеса»: 25% уклонения",
        special_details=(
            "Пепельная завеса: 25% уклонения от всех атак (в 0.7.0 было 20%)",
            "жёсткий потолок численности: три головы в неделю, и ни одной больше",
        ),
    ),
    "garpiia_zavesnitsa": CreatureStats(
        id="garpiia_zavesnitsa",
        name="Гарпия-завесница",
        tier=6,
        dwelling="Гнездовой навес",
        attack=12,
        defense=10,
        damage_min=8,
        damage_max=12,
        damage_display="8-12",
        health=30,
        speed=12,
        growth=3,
        cost_gold=500,
        ai_value=640,
        upgrade_of="pepelnaia_garpiia",
        special="завеса 25%; налёт без ответного удара",
        special_details=(
            "налёт без ответного удара: бьёт из завесы и в завесу уходит",
            "завеса накрывает соседние ряды",
        ),
    ),
    "fitil": CreatureStats(
        id="fitil",
        name="Фитиль",
        tier="особое",
        dwelling="Академия огня",
        attack=2,
        defense=4,
        damage_min=8,
        damage_max=12,
        damage_display="8-12",
        health=20,
        speed=6,
        growth=2,
        cost_gold=250,
        ai_value=290,
        upgrades_to="plamennyi",
        special="дальний огненный удар",
        special_details=(
            "инверсия Гога: статы атакующего почти нулевые, статы удара — полные",
            "прирост 2 не складывается ни с какими бонусами города",
        ),
    ),
    "plamennyi": CreatureStats(
        id="plamennyi",
        name="Пламенный",
        tier="особое",
        dwelling="Академия огня",
        attack=4,
        defense=6,
        damage_min=12,
        damage_max=18,
        damage_display="12-18",
        health=28,
        speed=7,
        growth=2,
        cost_gold=400,
        ai_value=460,
        upgrade_of="fitil",
        special="дальний удар; «подсветка» ночью",
        special_details=(
            "подсветка: союзники под его светом не теряют точности в пепельную ночь",
            "специализация Исольды: на 12-м уровне огненный залп Фитиля получает +20% урона",
        ),
    ),
    "lavovyi_drakon": CreatureStats(
        id="lavovyi_drakon",
        name="Лавовый Дракон",
        tier=7,
        dwelling="Горн зова",
        attack=18,
        defense=18,
        damage_min=40,
        damage_max=63,
        damage_display="40-63",
        health=220,
        speed=9,
        growth=1,
        cost_gold=2400,
        ai_value=4900,
        cost_ingots=1,
        effective_cost_gold=2650,
        upgrades_to="drakon_glubinnogo_zhara",
        special="иммунитет к огню; таран",
        special_details=(
            "иммунитет к огню полный; против остальных школ уязвим полностью",
            "таран без огненного дыхания: стихия дракона — ворота, стены и аргументы",
            "двойная бухгалтерия: городской найм 2400 золота + 1 слиток, эффективно 2650",
        ),
    ),
    "drakon_glubinnogo_zhara": CreatureStats(
        id="drakon_glubinnogo_zhara",
        name="Дракон глубинного жара",
        tier=7,
        dwelling="Горн зова",
        attack=21,
        defense=21,
        damage_min=45,
        damage_max=70,
        damage_display="45-70",
        health=275,
        speed=10,
        growth=1,
        cost_gold=3200,
        ai_value=6700,
        cost_ingots=2,
        effective_cost_gold=3700,
        upgrade_of="lavovyi_drakon",
        special="иммунитет к огню; жаркая чешуя",
        special_details=(
            "жаркая чешуя: всякий, кто ударяет его в ближнем бою, получает 5 урона за удар",
            "запрет найма двух драконов в одну неделю в мультиплеере 0.9.4",
        ),
    ),
}


@dataclass(frozen=True)
class CostSummary:
    """Итог расчёта стоимости: золото, слитки и эффективное золото.

    Attributes:
        gold: Сумма цен найма в золоте.
        ingots: Сумма цен найма в клейменых слитках (только драконы).
        effective_gold: Всё вместе, пересчитанное по городскому курсу 250.
    """

    gold: int
    ingots: int
    effective_gold: int


@dataclass(frozen=True)
class ArmyStats:
    """Сводные боевые показатели армии против заданной защиты.

    Attributes:
        heads: Общее число голов.
        total_health: Суммарное здоровье армии.
        min_damage: Суммарный минимальный урон за раунд с учётом множителя.
        max_damage: Суммарный максимальный урон за раунд с учётом множителя.
        avg_damage: Ожидаемый (средний) урон за раунд.
        power: Сумма оценок ИИ — «армейская мощь» по внутренней шкале.
        best_speed: Скорость самого быстрого отряда.
        stacks: Число отрядов в армии.
    """

    heads: int
    total_health: int
    min_damage: float
    max_damage: float
    avg_damage: float
    power: int
    best_speed: int
    stacks: int


@dataclass
class StackState:
    """Состояние одного отряда в симуляции стычки.

    Отряд хранится как пул здоровья: ``total_health`` убывает от урона, а
    число живых голов выводится делением с округлением вверх. Такой учёт
    проще гексагональной логики оригинала и не претендует на неё: симуляция
    отвечает на вопрос «кто остаётся стоять», а не «как именно это выглядело».

    Attributes:
        creature: Карточка существа.
        initial_count: Число голов на начало боя.
        total_health: Текущий пул здоровья отряда.
    """

    creature: CreatureStats
    initial_count: int
    total_health: int

    def alive_count(self) -> int:
        """Возвращает число живых голов (деление пула здоровья с округлением вверх)."""
        if self.total_health <= 0:
            return 0
        return -(-self.total_health // self.creature.health)

    @property
    def losses(self) -> int:
        """Потери отряда в головах от начала боя до текущего момента."""
        return self.initial_count - self.alive_count()


@dataclass
class BattleReport:
    """Отчёт о симулированной стычке двух армий.

    Attributes:
        winner: ``"attacker"``, ``"defender"`` или ``"draw"``.
        rounds: Число сыгранных раундов.
        attacker_losses: Потери атакующих в головах по идентификаторам существ.
        defender_losses: Потери защищающихся в головах по идентификаторам.
        log: Построчный журнал боя на русском языке.
    """

    winner: str
    rounds: int
    attacker_losses: Dict[str, int] = field(default_factory=dict)
    defender_losses: Dict[str, int] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)


def creature(creature_id: str) -> CreatureStats:
    """Возвращает карточку существа по идентификатору.

    Args:
        creature_id: Латинский идентификатор из JSON, например ``"molot"``.

    Returns:
        Немутуемая карточка :class:`CreatureStats`.

    Raises:
        KeyError: Если идентификатор неизвестен; в сообщении перечислены все
            допустимые идентификаторы, чтобы ошибка читалась у ночного костра
            без открытия файла.

    Examples:
        >>> creature("iskrovik").name
        'Искровик'
    """
    try:
        return CREATURES[creature_id]
    except KeyError:
        known = ", ".join(sorted(CREATURES))
        raise KeyError(f"Неизвестное существо: {creature_id!r}. Допустимо: {known}") from None


def find_creature(query: str) -> CreatureStats:
    """Ищет существо по идентификатору или русскому имени без учёта регистра.

    Поиск по имени удобен в быстрых прикидках («Молот:10»), но каноническим
    ключом остаётся латинский идентификатор: имена с «ё» и дефисами — плохая
    основа для CLI.

    Args:
        query: Идентификатор (``"molot"``) или имя (``"чёрный молот"``).

    Returns:
        Карточка существа.

    Raises:
        ValueError: Если ничего не найдено.
    """
    lowered = query.strip().lower().replace("ё", "е")
    for stats in CREATURES.values():
        if stats.id == query.strip() or stats.name.lower().replace("ё", "е") == lowered:
            return stats
    raise ValueError(f"Существо не найдено: {query!r}. Смотрите `python army_calculator.py list`.")


def damage_multiplier(attack: int, defense: int) -> float:
    """Вычисляет множитель урона по разнице атаки и защиты.

    Внутреннее соглашение дополнения, близкое к классической формуле: каждый
    пункт перевеса атаки над защитой даёт +5% урона (потолок ×4.0), каждый
    пункт перевеса защиты — −2.5% (пол ×0.25). Формула описана здесь, чтобы
    все утилиты команды считали одинаково; изменение формулы без правки
    этого docstring карается сверкой с балансом.

    Args:
        attack: Атака атакующего отряда (без героя).
        defense: Защита цели.

    Returns:
        Множитель урона в диапазоне [0.25; 4.0].

    Examples:
        >>> round(damage_multiplier(18, 18), 3)
        1.0
        >>> damage_multiplier(5, 1)
        1.2
        >>> damage_multiplier(1, 5)
        0.9
    """
    diff = attack - defense
    if diff >= 0:
        return min(1.0 + DAMAGE_MULTIPLIER_STEP * diff, MAX_DAMAGE_MULTIPLIER)
    return max(1.0 - DEFENSE_MULTIPLIER_STEP * (-diff), MIN_DAMAGE_MULTIPLIER)


def stack_damage(
    stats: CreatureStats, count: int, target_defense: int
) -> Tuple[float, float, float]:
    """Считает урон отряда из ``count`` голов против защиты ``target_defense``.

    Возвращаются минимальный, средний и максимальный урон за один раунд с
    учётом множителя :func:`damage_multiplier`. Особенности (секирный строй,
    поджиг, память лиц) не моделируются — это экономика головы, а не тактика.

    Args:
        stats: Карточка существа.
        count: Число голов в отряде.
        target_defense: Защита цели для расчёта множителя.

    Returns:
        Кортеж ``(мин, среднее, макс)``.

    Raises:
        ValueError: Если ``count`` отрицателен.
    """
    if count < 0:
        raise ValueError("Численность отряда не может быть отрицательной")
    multiplier = damage_multiplier(stats.attack, target_defense)
    low = stats.damage_min * count * multiplier
    mid = stats.average_damage * count * multiplier
    high = stats.damage_max * count * multiplier
    return low, mid, high


def parse_army_spec(spec: str) -> Dict[str, int]:
    """Разбирает текстовую спецификацию армии в словарь ``{id: count}``.

    Формат — отряды через запятую, идентификатор и численность через двоеточие:
    ``"molot:10,kleimenyi_golem:3,lavovyi_drakon:1"``. Пробелы вокруг частей
    игнорируются; вместо латинского идентификатора можно передать русское имя
    («Молот:10»), которое будет разрешён через :func:`find_creature`.

    Args:
        spec: Спецификация армии одной строкой.

    Returns:
        Словарь ``{идентификатор: численность}`` с нормализованными ключами.

    Raises:
        ValueError: При пустой спецификации, плохом формате или неизвестном
            существе (сообщение подсказывает команду ``list``).

    Examples:
        >>> parse_army_spec("molot:2, lavovyi_drakon:1")
        {'molot': 2, 'lavovyi_drakon': 1}
    """
    if not spec or not spec.strip():
        raise ValueError("Пустая спецификация армии; формат: id:число,id:число")
    army: Dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Непонятный отряд: {chunk!r} (нужен формат id:число)")
        raw_id, _, raw_count = chunk.rpartition(":")
        try:
            count = int(raw_count.strip())
        except ValueError:
            raise ValueError(f"Численность должна быть целым числом: {chunk!r}") from None
        if count <= 0:
            raise ValueError(f"Численность должна быть положительной: {chunk!r}")
        stats = find_creature(raw_id)
        army[stats.id] = count
    if not army:
        raise ValueError("В спецификации не нашлось ни одного отряда")
    return army


def _army_counts(army: Mapping[str, int]) -> List[Tuple[CreatureStats, int]]:
    """Приводит словарь армии к списку пар (карточка, численность).

    Внутренний помощник: единая точка валидации для функций, которым нужна
    и карточка существа, и число голов.
    """
    pairs: List[Tuple[CreatureStats, int]] = []
    for key, raw_count in army.items():
        stats = find_creature(str(key))
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"Численность должна быть положительной: {key}")
        pairs.append((stats, count))
    return pairs


def army_stats(army: Mapping[str, int], target_defense: int = 10) -> ArmyStats:
    """Считает сводные показатели армии против цели с заданной защитой.

    Функция отвечает на три вопроса интенданта: сколько в армии здоровья
    (насколько долго она стоит), сколько урона она кладёт за раунд (насколько
    быстро кончается противник) и сколько она стоит оценке ИИ (кого боевой
    искатель сочтёт лакомой целью). Защита цели по умолчанию 10 — усреднённая
    защита эталонных существ середины пирамиды.

    Args:
        army: Словарь ``{идентификатор: численность}`` (см. :func:`parse_army_spec`).
        target_defense: Защита цели для расчёта множителя урона.

    Returns:
        Немутуемая сводка :class:`ArmyStats`.

    Examples:
        >>> report = army_stats({"molot": 10})
        >>> report.heads
        10
        >>> report.total_health
        300
    """
    pairs = _army_counts(army)
    heads = sum(count for _, count in pairs)
    total_health = sum(stats.health * count for stats, count in pairs)
    low = mid = high = 0.0
    power = 0
    best_speed = 0
    for stats, count in pairs:
        stack_low, stack_mid, stack_high = stack_damage(stats, count, target_defense)
        low += stack_low
        mid += stack_mid
        high += stack_high
        power += stats.ai_value * count
        best_speed = max(best_speed, stats.speed)
    return ArmyStats(
        heads=heads,
        total_health=total_health,
        min_damage=round(low, 1),
        max_damage=round(high, 1),
        avg_damage=round(mid, 1),
        power=power,
        best_speed=best_speed,
        stacks=len(pairs),
    )


def army_power(army: Mapping[str, int]) -> int:
    """Возвращает «армейскую мощь» — сумму оценок ИИ по всем головам.

    Оценка ИИ (`ai_value` из выгрузки) — внутренний вес отряда для боевого
    искателя; сумма по армии даёт быстрый масштаб для сравнения армий в
    сценарных тестах. Мощь не заменяет симуляцию: она не знает ни позиций,
    ни синергий вроде «голем и волк».

    Args:
        army: Словарь ``{идентификатор: численность}``.

    Returns:
        Целочисленная мощь армии.
    """
    return sum(stats.ai_value * count for stats, count in _army_counts(army))


def recruit_cost(army: Mapping[str, int]) -> CostSummary:
    """Считает стоимость разового найма армии в золоте и слитках.

    Слитки учитываются только у драконов (Лавовый Дракон — 1 слиток, Дракон
    глубинного жара — 2) и пересчитываются в эффективное золото по
    городскому курсу 250. Эрафийский курс 180 здесь не используется: найм
    идёт в городе, а не у чужих торговцев.

    Args:
        army: Словарь ``{идентификатор: численность}``.

    Returns:
        :class:`CostSummary` с тремя суммами.

    Examples:
        >>> recruit_cost({"lavovyi_drakon": 1}).effective_gold
        2650
        >>> recruit_cost({"molot": 10}).gold
        1500
    """
    gold = 0
    ingots = 0
    for stats, count in _army_counts(army):
        gold += stats.cost_gold * count
        ingots += (stats.cost_ingots or 0) * count
    return CostSummary(
        gold=gold, ingots=ingots, effective_gold=gold + ingots * INGOT_CITY_RATE_GOLD
    )


def golem_repair_cost(hp_lost: int, orn_discount: bool = False) -> int:
    """Считает стоимость ремонта големов в золоте за потерянные здоровья.

    Тариф — 47 золота за единицу здоровья, неизменен со сборки 0.7.3 и
    трогать его запрещено протоколом 57. Скидка специализации Орна Тлеющего
    Круга — минус 20% с округлением вверх: 47 × 0,8 = 37,6 → 38 золота за
    ХП. Это единственная скидка игры, округляемая вверх, и единственная, за
    которую казна говорит «спасибо».

    Args:
        hp_lost: Суммарно потерянные здоровья после боя.
        orn_discount: Взять ли тариф Орна (38 золота за ХП) вместо базового.

    Returns:
        Итоговая стоимость ремонта целым числом золота.

    Raises:
        ValueError: Если ``hp_lost`` отрицателен.

    Examples:
        >>> golem_repair_cost(50)
        2350
        >>> golem_repair_cost(50, orn_discount=True)
        1900
        >>> golem_repair_cost(1, orn_discount=True)
        38
    """
    if hp_lost < 0:
        raise ValueError("Потерянные здоровья не могут быть отрицательными")
    if orn_discount:
        return math.ceil(hp_lost * ORN_REPAIR_GOLD_PER_HP)
    return hp_lost * GOLEM_REPAIR_GOLD_PER_HP


def weekly_pyramid_cost() -> List[Dict[str, Any]]:
    """Строит таблицу недельной экономики пирамиды по базовым постройкам.

    Для каждого жилища берётся базовое (не апгрейдное) существо, его прирост
    по уставу 908 года умножается на цену найма. Итог — 46 голов и 7770
    золота плюс 1 слиток за дракона; доли считаются от золота и округляются
    до десятых. Игрок должен знать в воскресенье, что он купит в понедельник,
    и эта таблица — то самое расписание.

    Returns:
        Список словарей с ключами ``dwelling``, ``heads_per_week``,
        ``gold_per_week``, ``extra`` и ``share_percent``.
    """
    rows: List[Dict[str, Any]] = []
    bases = [s for s in CREATURES.values() if s.upgrade_of is None]
    by_dwelling: Dict[str, CreatureStats] = {s.dwelling: s for s in bases}
    total_gold = 0
    total_heads = 0
    for dwelling in DWELLING_ORDER:
        stats = by_dwelling[dwelling]
        gold = stats.cost_gold * stats.growth
        ingots = (stats.cost_ingots or 0) * stats.growth
        total_gold += gold
        total_heads += stats.growth
        rows.append(
            {
                "dwelling": dwelling,
                "heads_per_week": stats.growth,
                "gold_per_week": gold,
                "extra": f"{ingots} слиток" if ingots else "",
            }
        )
    for row in rows:
        row["share_percent"] = round(row["gold_per_week"] / total_gold * 100, 1)
    rows.append(
        {
            "dwelling": "Итого",
            "heads_per_week": total_heads,
            "gold_per_week": total_gold,
            "extra": "1 слиток",
            "share_percent": 100.0,
        }
    )
    return rows


def simulate_skirmish(
    attacker_army: Mapping[str, int],
    defender_army: Mapping[str, int],
    seed: Optional[int] = None,
    max_rounds: int = 100,
) -> BattleReport:
    """Симулирует упрощённую стычку двух армий без героя, магии и позиций.

    Модель честна в своей простоте: раундами, в порядке убывания скорости,
    каждый отряд бьёт первый живой отряд противника средним уроном с
    множителем :func:`damage_multiplier`; гарпии с вероятностью 25% уходят от
    удара («Пепельная завеса» — уклонение видно всем). Урон снимается с
    пула здоровья отряда, погибшие считаются делением с округлением вверх.
    Результат воспроизводим при фиксированном ``seed``.

    Что модель сознательно не умеет: огонь Фитиля на дистанции (здесь он
    дерётся в лоб), секирный строй Молотов, ответный урон чешуи, поджиг,
    мораль и удача. Для вопросов «стоит ли лезть» этого довольно; для
    вопросов «как именно лезть» существует плейтест.

    Args:
        attacker_army: Словарь армии атакующих.
        defender_army: Словарь армии защищающихся.
        seed: Зерно генератора случайных чисел для уклонений гарпий.
        max_rounds: Предохранитель по числу раундов (по умолчанию 100).

    Returns:
        :class:`BattleReport` с победителем, раундами, потерями и журналом.

    Examples:
        >>> report = simulate_skirmish({"molot": 20}, {"ugolnitsa": 25}, seed=903)
        >>> report.winner in {"attacker", "defender", "draw"}
        True
    """
    rng = random.Random(seed)
    log: List[str] = []
    attackers = [StackState(s, n, s.health * n) for s, n in _army_counts(attacker_army)]
    defenders = [StackState(s, n, s.health * n) for s, n in _army_counts(defender_army)]
    attackers.sort(key=lambda st: st.creature.speed, reverse=True)
    defenders.sort(key=lambda st: st.creature.speed, reverse=True)

    def side_alive(stacks: Sequence[StackState]) -> bool:
        return any(st.alive_count() > 0 for st in stacks)

    round_no = 0
    for round_no in range(1, max_rounds + 1):
        log.append(f"— Раунд {round_no} —")
        for side_name, acting, opposing in (
            ("Атакующие", attackers, defenders),
            ("Защитники", defenders, attackers),
        ):
            for stack in acting:
                if stack.alive_count() <= 0 or not side_alive(opposing):
                    continue
                target = next(st for st in opposing if st.alive_count() > 0)
                if target.creature.id in DODGING_CREATURES and rng.random() < HARPY_DODGE_CHANCE:
                    log.append(
                        f"{side_name}: {stack.creature.name} x{stack.alive_count()} бьёт по "
                        f"{target.creature.name} — Пепельная завеса, уклонение"
                    )
                    continue
                multiplier = damage_multiplier(stack.creature.attack, target.creature.defense)
                dealt = int(round(stack.creature.average_damage * stack.alive_count() * multiplier))
                before = target.alive_count()
                target.total_health = max(0, target.total_health - dealt)
                killed = before - target.alive_count()
                wound_note = "" if killed else " (без убитых)"
                log.append(
                    f"{side_name}: {stack.creature.name} x{stack.alive_count()} → "
                    f"{target.creature.name}: {dealt} урона, {killed} погибло{wound_note}"
                )
        if not side_alive(defenders) or not side_alive(attackers):
            break

    attacker_alive = side_alive(attackers)
    defender_alive = side_alive(defenders)
    if attacker_alive and not defender_alive:
        winner = "attacker"
    elif defender_alive and not attacker_alive:
        winner = "defender"
    else:
        winner = "draw"

    attacker_losses = {st.creature.id: st.losses for st in attackers if st.losses}
    defender_losses = {st.creature.id: st.losses for st in defenders if st.losses}
    return BattleReport(
        winner=winner,
        rounds=round_no,
        attacker_losses=attacker_losses,
        defender_losses=defender_losses,
        log=log,
    )


def load_creatures(path: Union[str, Path]) -> Dict[str, CreatureStats]:
    """Загружает карточки существ из машинной выгрузки JSON.

    Полезно для сверки встроенного словаря с выгрузкой: утилита читает
    ``creature-stats.json`` и строит те же :class:`CreatureStats`. Файл
    фауны (`hireable: false`) и эталоны в загрузку не входят — пирамида
    найма Кальдеры состоит из шестнадцати боевых существ.

    Args:
        path: Путь к JSON-файлу выгрузки.

    Returns:
        Словарь ``{идентификатор: карточка}``.

    Raises:
        ValueError: Если в файле нет списка ``creatures``.
        OSError: Если файл не читается.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    creatures_raw = data.get("creatures")
    if not isinstance(creatures_raw, list):
        raise ValueError(f"В {path} нет списка 'creatures' — это не выгрузка статов")
    loaded: Dict[str, CreatureStats] = {}
    for entry in creatures_raw:
        stats = CreatureStats(
            id=entry["id"],
            name=entry["name"],
            tier=entry["tier"],
            dwelling=entry["dwelling"],
            attack=entry["attack"],
            defense=entry["defense"],
            damage_min=entry["damage_min"],
            damage_max=entry["damage_max"],
            damage_display=entry["damage_display"],
            health=entry["health"],
            speed=entry["speed"],
            growth=entry["growth"],
            cost_gold=entry["cost_gold"],
            ai_value=entry["ai_value"],
            cost_ingots=entry.get("cost_ingots"),
            effective_cost_gold=entry.get("effective_cost_gold"),
            upgrades_to=entry.get("upgrades_to"),
            upgrade_of=entry.get("upgrade_of"),
            special=entry.get("special", ""),
            special_details=tuple(entry.get("special_details", ())),
        )
        loaded[stats.id] = stats
    return loaded


def render_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool] = ()
) -> str:
    """Формирует моноширинную таблицу с выравниванием колонок.

    Числовые колонки выравниваются вправо, текстовые — влево; ширина
    берётся по самой длинной ячейке колонки. Утилита без внешних
    зависимостей: таблицы читаются в терминале, в журналах и в отчётах
    сборки одинаково.

    Args:
        headers: Заголовки колонок.
        rows: Строки таблицы (строковое представление значений).
        numeric: Флаги «колонка числовая» по индексам колонок.

    Returns:
        Готовая таблица одной строкой (с переводами строк).
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    numeric = tuple(numeric) + (False,) * (len(headers) - len(numeric))

    def fmt_row(row: Sequence[str]) -> str:
        cells = []
        for idx, cell in enumerate(row):
            cell = str(cell)
            if numeric[idx]:
                cells.append(cell.rjust(widths[idx]))
            else:
                cells.append(cell.ljust(widths[idx]))
        return "  ".join(cells).rstrip()

    lines = [fmt_row(headers)]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def creature_card(creature_id: str) -> str:
    """Собирает читаемую карточку существа для команды ``creature``.

    Args:
        creature_id: Идентификатор или русское имя существа.

    Returns:
        Многострочная карточка: статы, цены с двойной бухгалтерией слитка,
        особенность и её подробности.

    Raises:
        ValueError: Если существо не найдено.
    """
    stats = find_creature(creature_id)
    tier = stats.tier if isinstance(stats.tier, int) else f"{stats.tier} (вне пирамиды)"
    lines = [
        f"{stats.name}  (id: {stats.id})",
        f"  Уровень: {tier}   Жилище: {stats.dwelling}",
        f"  Атака {stats.attack}   Защита {stats.defense}   Урон {stats.damage_display}   "
        f"Здоровье {stats.health}   Скорость {stats.speed}",
        f"  Прирост: {stats.growth}/нед   Оценка ИИ: {stats.ai_value}",
        f"  Найм: {stats.cost_gold} золота"
        + (f" + {stats.cost_ingots} слиток(а)" if stats.cost_ingots else ""),
        f"  Эффективная цена: {stats.effective_cost()} золота (курс слитка {INGOT_CITY_RATE_GOLD})",
    ]
    if stats.upgrades_to:
        lines.append(f"  Апгрейд: {creature(stats.upgrades_to).name}")
    if stats.special:
        lines.append(f"  Особенность: {stats.special}")
    for detail in stats.special_details:
        lines.append(f"    • {detail}")
    return "\n".join(lines)


def _force_utf8_stdio() -> None:
    """Переводит стандартные потоки в UTF-8 для стабильного вывода кириллицы.

    На Windows при перенаправлении вывода Python может выбрать кодировку
    локали, в которой кириллица живёт плохо. Повторная настройка потоков в
    UTF-8 делает ``--help`` и отчёты одинаковыми в консоли, в пайпе и в
    журнале сборки.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _apply_data_override(path: Optional[str]) -> None:
    """Подменяет встроенные карточки данными из JSON, если передан ``--data``.

    Сверка и подгонка: утилита может работать и по вшитым числам сборки
    0.9.4, и по свежей выгрузке. После подмены все команды считают по новым
    данным, о чём печатается строка-примечание.
    """
    if not path:
        return
    loaded = load_creatures(path)
    CREATURES.clear()
    CREATURES.update(loaded)
    print(f"[данные] загружено {len(loaded)} существ из {path}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Собирает CLI-парсер утилиты с подкомандами.

    Подкоманды: ``list`` (таблица существ), ``creature`` (карточка),
    ``cost`` (стоимость найма), ``stats`` (сводка армии), ``battle``
    (симуляция стычки), ``repair`` (ремонт големов) и ``economy``
    (недельная пирамида). Общая опция подкоманд ``--data`` подменяет
    встроенные числа выгрузкой JSON.

    Returns:
        Настроенный :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="army_calculator.py",
        description="Калькулятор армий Кальдеры: найм, содержание, урон и стычки "
        "по числам сборки «Трона Пепла» 0.9.4.",
        epilog="Источник чисел: sample_docs/data/creature-stats.json. "
        "Балансовые обоснования: sample_docs/balance/creature-stats.md.",
    )
    parser.add_argument("--version", action="version", version=f"army_calculator.py {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="команда")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data",
        metavar="JSON",
        help="взять статы из выгрузки JSON вместо вшитых чисел сборки",
    )

    list_parser = subparsers.add_parser(
        "list", parents=[common], help="таблица всех существ пирамиды и Академии"
    )
    list_parser.add_argument("--tier", metavar="УР", help="фильтр по уровню: 1-7 или особое")

    subparsers.add_parser(
        "creature", parents=[common], help="карточка существа по идентификатору или имени"
    ).add_argument("target", metavar="СУЩЕСТВО", help="id или русское имя, например molot")

    cost_parser = subparsers.add_parser("cost", parents=[common], help="стоимость найма армии")
    cost_parser.add_argument(
        "--army", required=True, metavar="СПЕЦ", help='армия: "molot:10,lavovyi_drakon:1"'
    )
    cost_parser.add_argument(
        "--weeks", type=int, default=1, metavar="N", help="число недель найма (по умолчанию 1)"
    )
    cost_parser.add_argument("--json", action="store_true", help="вывести итог машиночитаемым JSON")

    stats_parser = subparsers.add_parser(
        "stats", parents=[common], help="сводные боевые показатели армии"
    )
    stats_parser.add_argument(
        "--army", required=True, metavar="СПЕЦ", help='армия: "molot:10,pepelnaia_garpiia:3"'
    )
    stats_parser.add_argument(
        "--defense", type=int, default=10, metavar="N", help="защита цели (по умолчанию 10)"
    )
    stats_parser.add_argument(
        "--json", action="store_true", help="вывести итог машиночитаемым JSON"
    )

    battle_parser = subparsers.add_parser(
        "battle", parents=[common], help="симуляция стычки двух армий"
    )
    battle_parser.add_argument("--attacker", required=True, metavar="СПЕЦ", help="армия атакующих")
    battle_parser.add_argument(
        "--defender", required=True, metavar="СПЕЦ", help="армия защищающихся"
    )
    battle_parser.add_argument(
        "--seed", type=int, default=None, metavar="N", help="зерно ГПЧ для уклонений гарпий"
    )
    battle_parser.add_argument(
        "--rounds", type=int, default=100, metavar="N", help="потолок раундов (по умолчанию 100)"
    )
    battle_parser.add_argument("--log", action="store_true", help="напечатать полный журнал боя")
    battle_parser.add_argument(
        "--json", action="store_true", help="вывести отчёт машиночитаемым JSON"
    )

    repair_parser = subparsers.add_parser(
        "repair", parents=[common], help="стоимость ремонта големов"
    )
    repair_parser.add_argument(
        "--creature",
        dest="creature_id",
        default="obsidianovyi_golem",
        metavar="ID",
        help="голем (по умолчанию obsidianovyi_golem)",
    )
    repair_parser.add_argument(
        "--hp", required=True, type=int, metavar="N", help="сколько здоровья потеряно"
    )
    repair_parser.add_argument(
        "--orn", action="store_true", help="тариф Орна Тлеющего Круга: 38 золота за ХП"
    )
    repair_parser.add_argument(
        "--json", action="store_true", help="вывести итог машиночитаемым JSON"
    )

    subparsers.add_parser("economy", help="недельная экономика пирамиды (7770 золота + 1 слиток)")
    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    """Выполняет подкоманду ``list``: таблица существ с фильтром по уровню."""
    rows: List[List[str]] = []
    for stats in CREATURES.values():
        tier = str(stats.tier)
        if args.tier and tier != args.tier:
            continue
        price = str(stats.cost_gold) + (f" + {stats.cost_ingots} сл." if stats.cost_ingots else "")
        rows.append(
            [
                stats.name,
                tier,
                str(stats.attack),
                str(stats.defense),
                stats.damage_display,
                str(stats.health),
                str(stats.speed),
                str(stats.growth),
                price,
                str(stats.ai_value),
            ]
        )
    print(
        render_table(
            ["Существо", "Ур.", "Атк", "Зщт", "Урон", "ХП", "Скр", "Прир", "Цена", "ИИ"],
            rows,
            numeric=(False, True, True, True, False, True, True, True, False, True),
        )
    )
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    """Выполняет подкоманду ``cost``: найм армии, при желании на N недель."""
    army = parse_army_spec(args.army)
    one_shot = recruit_cost(army)
    summary = CostSummary(
        gold=one_shot.gold * args.weeks,
        ingots=one_shot.ingots * args.weeks,
        effective_gold=one_shot.effective_gold * args.weeks,
    )
    if args.json:
        print(
            json.dumps(
                {"army": army, "weeks": args.weeks, **vars(summary)}, ensure_ascii=False, indent=2
            )
        )
        return 0
    print(f"Армия: {args.army}")
    print(f"Недель найма: {args.weeks}")
    print(f"Золото: {summary.gold}")
    print(f"Слитки: {summary.ingots}")
    print(f"Эффективное золото (курс {INGOT_CITY_RATE_GOLD}): {summary.effective_gold}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Выполняет подкоманду ``stats``: сводные показатели армии."""
    army = parse_army_spec(args.army)
    report = army_stats(army, target_defense=args.defense)
    if args.json:
        print(json.dumps({"army": army, **vars(report)}, ensure_ascii=False, indent=2))
        return 0
    print(f"Армия: {args.army} (отрядов: {report.stacks})")
    print(f"Голов: {report.heads}   Суммарное здоровье: {report.total_health}")
    print(
        f"Урон за раунд против защиты {args.defense}: "
        f"{report.min_damage} — {report.max_damage} (в среднем {report.avg_damage})"
    )
    print(f"Мощь (оценка ИИ): {report.power}   Лучшая скорость: {report.best_speed}")
    return 0


def _cmd_battle(args: argparse.Namespace) -> int:
    """Выполняет подкоманду ``battle``: симуляция стычки и отчёт о ней."""
    attacker = parse_army_spec(args.attacker)
    defender = parse_army_spec(args.defender)
    report = simulate_skirmish(attacker, defender, seed=args.seed, max_rounds=args.rounds)
    verdict = {
        "attacker": "победа атакующих",
        "defender": "победа защитников",
        "draw": "взаимное истребление",
    }[report.winner]
    if args.json:
        print(
            json.dumps(
                {
                    "attacker": attacker,
                    "defender": defender,
                    "seed": args.seed,
                    "winner": report.winner,
                    "rounds": report.rounds,
                    "attacker_losses": report.attacker_losses,
                    "defender_losses": report.defender_losses,
                    "log": report.log,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"Атакующие: {args.attacker}")
    print(f"Защитники: {args.defender}")
    print(f"Итог: {verdict} за {report.rounds} раундов (зерно {args.seed}).")
    for label, losses in (
        ("атакующие", report.attacker_losses),
        ("защитники", report.defender_losses),
    ):
        if losses:
            pretty = ", ".join(f"{CREATURES[cid].name} x{n}" for cid, n in losses.items())
            print(f"Потери {label}: {pretty}")
    if args.log:
        print("\n".join(report.log))
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    """Выполняет подкоманду ``repair``: смета ремонта големов."""
    stats = find_creature(args.creature_id)
    if stats.id not in GOLEM_IDS:
        print(
            f"[внимание] {stats.name} — не голем; считаю по тарифу для сверки.",
            file=sys.stderr,
        )
    price = golem_repair_cost(args.hp, orn_discount=args.orn)
    if args.json:
        print(
            json.dumps(
                {
                    "creature": stats.id,
                    "hp_lost": args.hp,
                    "rate_gold_per_hp": ORN_REPAIR_GOLD_PER_HP
                    if args.orn
                    else GOLEM_REPAIR_GOLD_PER_HP,
                    "total_gold": price,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    rate = ORN_REPAIR_GOLD_PER_HP if args.orn else GOLEM_REPAIR_GOLD_PER_HP
    print(f"Существо: {stats.name}")
    print(f"Потеряно здоровья: {args.hp}")
    print(f"Тариф: {rate} золота за ХП" + (" (Орн Тлеющий Круг)" if args.orn else ""))
    print(f"Итого к уплате: {price} золота")
    if args.orn:
        print(f"Примечание: {ORN_REPAIR_NOTE}")
    return 0


def _cmd_economy() -> int:
    """Выполняет подкоманду ``economy``: недельная таблица пирамиды."""
    rows = weekly_pyramid_cost()
    table_rows = [
        [
            r["dwelling"],
            str(r["heads_per_week"]),
            str(r["gold_per_week"]),
            r.get("extra", ""),
            f"{r['share_percent']:.1f}%",
        ]
        for r in rows
    ]
    print(
        render_table(
            ["Жилище", "Голов/нед", "Золото/нед", "Слитки", "Доля"],
            table_rows,
            numeric=(False, True, True, False, False),
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа CLI; разбирает аргументы и выполняет подкоманду.

    Args:
        argv: Аргументы командной строки (по умолчанию — ``sys.argv[1:]``).

    Returns:
        Код возврата: 0 — успех, 1 — ошибка данных, 2 — ошибка аргументов
        (аргпартс завершает процесс сам).

    Examples:
        >>> main(["stats", "--army", "molot:1"]) in (0, 1)
        True
    """
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _apply_data_override(args.data)
            return _cmd_list(args)
        if args.command == "creature":
            _apply_data_override(args.data)
            print(creature_card(args.target))
            return 0
        if args.command == "cost":
            _apply_data_override(args.data)
            return _cmd_cost(args)
        if args.command == "stats":
            _apply_data_override(args.data)
            return _cmd_stats(args)
        if args.command == "battle":
            _apply_data_override(args.data)
            return _cmd_battle(args)
        if args.command == "repair":
            _apply_data_override(args.data)
            return _cmd_repair(args)
        if args.command == "economy":
            return _cmd_economy()
    except (ValueError, KeyError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
