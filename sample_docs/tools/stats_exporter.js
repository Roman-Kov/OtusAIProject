/**
 * stats_exporter.js — экспортёр статов существ Кальдеры во внутренние форматы.
 *
 * Внутренняя утилита команды дополнения «Трон Пепла»: читает машинную
 * выгрузку `sample_docs/data/creature-stats.json` (сборка 0.9.4) и строит
 * по ней CSV для табличных сверок, Markdown-таблицы для вики базы знаний и
 * производные показатели (урон на голову, здоровье за золото, доля
 * недельной пирамиды) для балансовой группы. Источник истины —
 * `sample_docs/balance/creature-stats.md`; если выгрузка расходится с
 * таблицами, правьте конфигурацию и перегенерируйте выгрузку, а не наоборот.
 *
 * Использование из командной строки (Node.js без внешних зависимостей):
 *
 *     node tools/stats_exporter.js --help
 *     node tools/stats_exporter.js --format csv
 *     node tools/stats_exporter.js --format md --tier 7
 *     node tools/stats_exporter.js --format json --sort ai_value
 *     node tools/stats_exporter.js --input ../data/creature-stats.json --out table.csv
 *
 * Использование как модуля (CommonJS):
 *
 *     const exporter = require('./stats_exporter.js');
 *     const doc = exporter.loadStats('../data/creature-stats.json');
 *     const rows = exporter.flattenCreatures(doc);
 *     console.log(exporter.toMarkdownTable(rows, exporter.STATS_COLUMNS));
 *
 * Соглашения, зашитые в утилиту (раздел «Методика» balance/creature-stats.md):
 * урон пишется цифрами через дефис («40-63»), проценты — цифрами со знаком,
 * клейменый слиток стоит 250 золота в городе и 180 у эрафийских торговцев,
 * ремонт големов — 47 золота за ХП. Всё это константы модуля, а не магия.
 *
 * @module stats_exporter
 * @version 0.9.4
 * @license внутренняя утилита команды «Трон Пепла»; распространение вне команды не предусмотрено
 */

'use strict';

const fs = require('fs');
const path = require('path');

/**
 * Курс клейменого слитка внутри города; по нему считаются effective-цены драконов.
 * @type {number}
 */
const INGOT_CITY_RATE_GOLD = 250;

/**
 * Курс слитка у эрафийских торговцев: обратная продажа идёт с потерей трети.
 * @type {number}
 */
const INGOT_ERATHIAN_RATE_GOLD = 180;

/**
 * Тариф ремонта големов: 47 золота за единицу здоровья, не менялся с 0.7.3.
 * @type {number}
 */
const GOLEM_REPAIR_GOLD_PER_HP = 47;

/** Путь к выгрузке по умолчанию — рядом с утилитой, в ../data/. */
const DEFAULT_INPUT = path.join(__dirname, '..', 'data', 'creature-stats.json');

/** Дельта апгрейда обязана укладываться в коридор 30-45% силы отряда. */
const UPGRADE_CORRIDOR = { min: 0.3, max: 0.45 };

/** Жесткий потолок прироста гарпий: три головы в неделю, «и ни одной больше». */
const HARPY_WEEKLY_CAP = 3;

/** Итог недельной пирамиды при базовых постройках: 46 голов. */
const WEEKLY_HEADS_TOTAL = 46;

/** Итог недельной пирамиды: 7770 золота плюс 1 слиток за дракона. */
const WEEKLY_GOLD_TOTAL = 7770;

/**
 * @typedef {Object} CreatureRow
 * @property {string} id            Латинский идентификатор (например, lavovyi_drakon).
 * @property {string} name          Русское имя существа (например, «Лавовый Дракон»).
 * @property {(number|string)} tier Уровень пирамиды: 1-7 или строка «особое».
 * @property {string} dwelling      Жилище города.
 * @property {number} attack        Базовая атака.
 * @property {number} defense       Базовая защита.
 * @property {number} damage_min    Минимальный урон одной головы.
 * @property {number} damage_max    Максимальный урон одной головы.
 * @property {string} damage_display Каноническая запись урона через дефис.
 * @property {number} health        Здоровье одной головы.
 * @property {number} speed         Скорость.
 * @property {number} growth        Недельный прирост жилища.
 * @property {number} cost_gold     Цена найма в золоте.
 * @property {?number} cost_ingots  Цена в клейменых слитках (только драконы).
 * @property {number} effective_cost_gold Эффективная цена по городскому курсу слитка.
 * @property {number} ai_value      Вес отряда для боевого искателя ИИ.
 * @property {string} special       Короткая сводка ключевой особенности.
 */

/**
 * @typedef {Object} ColumnSpec
 * @property {string} key    Поле строки для выборки значения.
 * @property {string} header Заголовок колонки в таблицах.
 * @property {boolean} [numeric] Числовая ли колонка (для выравнивания в Markdown).
 */

/** Колонки сводной таблицы статов — аналог раздела 2.1 balance/creature-stats.md. */
const STATS_COLUMNS = [
  { key: 'name', header: 'Существо' },
  { key: 'tier', header: 'Ур.' },
  { key: 'attack', header: 'Атака', numeric: true },
  { key: 'defense', header: 'Защита', numeric: true },
  { key: 'damage_display', header: 'Урон' },
  { key: 'health', header: 'Здоровье', numeric: true },
  { key: 'speed', header: 'Скорость', numeric: true },
  { key: 'growth', header: 'Прирост', numeric: true },
  { key: 'cost_gold', header: 'Цена', numeric: true },
  { key: 'ai_value', header: 'Оценка ИИ', numeric: true },
];

/** Колонки экономической таблицы: что даёт голова за своё золото. */
const ECONOMY_COLUMNS = [
  { key: 'name', header: 'Существо' },
  { key: 'effective_cost_gold', header: 'Эфф. цена', numeric: true },
  { key: 'avg_damage', header: 'Ср. урон', numeric: true },
  { key: 'dpr_per_1000', header: 'Урон/1000 зол.', numeric: true },
  { key: 'hp_per_1000', header: 'ХП/1000 зол.', numeric: true },
  { key: 'weekly_gold', header: 'Золото/нед', numeric: true },
  { key: 'weekly_share', header: 'Доля пирамиды', numeric: true },
];

/**
 * Загружает и разбирает выгрузку статов.
 *
 * Функция синхронная намеренно: утилита командной строки читает один файл
 * на один запуск, и асинхронность здесь — обещания без причины.
 *
 * @param {string} [filePath=DEFAULT_INPUT] Путь к creature-stats.json.
 * @returns {Object} Разобранный JSON-документ выгрузки.
 * @throws {Error} Если файл не найден или JSON не разбирается; в сообщении
 *   подсказывается путь по умолчанию, чтобы новый член команды не искал.
 */
function loadStats(filePath) {
  const target = filePath || DEFAULT_INPUT;
  if (!fs.existsSync(target)) {
    throw new Error(`Файл выгрузки не найден: ${target} (по умолчанию: ${DEFAULT_INPUT})`);
  }
  return JSON.parse(fs.readFileSync(target, 'utf8'));
}

/**
 * Выбирает список существ из выгрузки в плоский массив строк.
 *
 * @param {Object} doc Документ выгрузки (результат loadStats).
 * @returns {CreatureRow[]} Массив карточек существ как есть, без сортировки.
 * @throws {Error} Если в документе нет массива creatures.
 */
function flattenCreatures(doc) {
  if (!doc || !Array.isArray(doc.creatures)) {
    throw new Error('В документе нет массива creatures — это не выгрузка статов Кальдеры');
  }
  return doc.creatures.slice();
}

/**
 * Форматирует целое число с разделителем тысяч (пробел по русской традиции).
 *
 * @param {number} value Число для форматирования.
 * @returns {string} Строка вида «2 400»; отрицательные значения не предусмотрены сметой.
 */
function formatNumber(value) {
  return String(Math.round(value)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/**
 * Форматирует коридор урона с дефисом — правило дефисов балансовой группы.
 *
 * @param {number} min Минимальный урон.
 * @param {number} max Максимальный урон.
 * @returns {string} Строка вида «40-63»; словесная форма не находится поиском.
 */
function formatRange(min, max) {
  return `${min}-${max}`;
}

/**
 * Считает эффективную цену существа в золоте по городскому курсу слитка.
 *
 * Для существ без слитков равна цене в золоте; для Лавового Дракона
 * (2400 + 1×250) — 2650, для Дракона глубинного жара (3200 + 2×250) — 3700.
 *
 * @param {CreatureRow} creature Карточка существа.
 * @returns {number} Эффективная цена в золоте.
 */
function effectiveCostGold(creature) {
  const ingots = creature.cost_ingots || 0;
  return creature.cost_gold + ingots * INGOT_CITY_RATE_GOLD;
}

/**
 * Считает производные показатели существа для экономической таблицы.
 *
 * Добавляет к карточке: средний урон головы, урон за раунд на 1000 золота,
 * здоровье на 1000 золота, недельную стоимость прироста и долю недельной
 * пирамиды. Доля считается от {@link WEEKLY_GOLD_TOTAL} — 7770 золота, и в
 * сумме по базовым существам даёт сто процентов с точностью до округления.
 *
 * @param {CreatureRow} creature Карточка существа.
 * @returns {Object} Копия карточки с полями avgDamage, dprPer1000, hpPer1000,
 *   weeklyGold и weeklyShare (в процентах, один знак после запятой).
 */
function computeDerived(creature) {
  const effective = effectiveCostGold(creature);
  const avgDamage = (creature.damage_min + creature.damage_max) / 2;
  const dprPer1000 = effective > 0 ? (avgDamage / effective) * 1000 : 0;
  const hpPer1000 = effective > 0 ? (creature.health / effective) * 1000 : 0;
  const weeklyGold = creature.cost_gold * creature.growth;
  const weeklyShare = Math.round((weeklyGold / WEEKLY_GOLD_TOTAL) * 1000) / 10;
  return Object.assign({}, creature, {
    effective_cost_gold: effective,
    avg_damage: Math.round(avgDamage * 10) / 10,
    dpr_per_1000: Math.round(dprPer1000 * 10) / 10,
    hp_per_1000: Math.round(hpPer1000 * 10) / 10,
    weekly_gold: weeklyGold,
    weekly_share: weeklyShare,
  });
}

/**
 * Экранирует значение ячейки для CSV.
 *
 * Кавычки удваиваются, точки с запятой и переводы строк заставляют поле
 * взять себя в кавычки. Разделитель — точка с запятой: русские локали Excel
 * ждут именно её, а балансовая группа живёт в Excel.
 *
 * @param {*} value Значение ячейки (приводится к строке).
 * @returns {string} Безопасное для CSV представление.
 */
function escapeCsvValue(value) {
  const text = value === null || value === undefined ? '' : String(value);
  if (/[";\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

/**
 * Строит CSV-текст из строк по списку колонок.
 *
 * @param {Object[]} rows Строки данных.
 * @param {ColumnSpec[]} columns Спецификации колонок.
 * @returns {string} CSV с заголовком и строками, разделитель «;».
 */
function toCSV(rows, columns) {
  const header = columns.map((col) => escapeCsvValue(col.header)).join(';');
  const lines = rows.map((row) =>
    columns.map((col) => escapeCsvValue(row[col.key])).join(';')
  );
  return [header].concat(lines).join('\r\n');
}

/**
 * Строит Markdown-таблицу из строк по списку колонок.
 *
 * Числовые колонки выравниваются вправо (`---:`), текстовые — влево; это
 * же правило использует генератор таблиц раздела balance/, чтобы вики базы
 * выглядела одинаково вне зависимости от источника таблицы.
 *
 * @param {Object[]} rows Строки данных.
 * @param {ColumnSpec[]} columns Спецификации колонок.
 * @returns {string} Markdown-таблица с шапкой и разделителем.
 */
function toMarkdownTable(rows, columns) {
  const header = `| ${columns.map((col) => col.header).join(' | ')} |`;
  const divider = `| ${columns.map((col) => (col.numeric ? '---:' : '---')).join(' | ')} |`;
  const body = rows.map((row) => {
    const cells = columns.map((col) => {
      const value = row[col.key];
      return value === null || value === undefined ? '—' : String(value);
    });
    return `| ${cells.join(' | ')} |`;
  });
  return [header, divider].concat(body).join('\n');
}

/**
 * Сортирует карточки существ по значению поля.
 *
 * Поддерживаемые поля: числовые статы, ai_value, effective_cost_gold и
 * name (по алфавиту). Порядок — убывание по умолчанию: балансовую группу
 * интересует сначала сильное и дорогое.
 *
 * @param {CreatureRow[]} creatures Массив карточек.
 * @param {string} key Поле сортировки.
 * @param {boolean} [descending=true] Убывание (по умолчанию) или возрастание.
 * @returns {CreatureRow[]} Новый отсортированный массив.
 */
function sortBy(creatures, key, descending) {
  const desc = descending === undefined ? true : descending;
  const factor = desc ? -1 : 1;
  return creatures.slice().sort((a, b) => {
    const left = a[key];
    const right = b[key];
    if (typeof left === 'string' || typeof right === 'string') {
      return String(left).localeCompare(String(right), 'ru') * factor;
    }
    return ((left || 0) - (right || 0)) * factor;
  });
}

/**
 * Фильтрует карточки по уровню пирамиды.
 *
 * @param {CreatureRow[]} creatures Массив карточек.
 * @param {(number|string)} tier Уровень: 1-7, 0/«все» — без фильтра, «особое» — Академия.
 * @returns {CreatureRow[]} Отфильтрованная копия массива.
 */
function filterByTier(creatures, tier) {
  if (tier === undefined || tier === null || tier === 0 || tier === 'все') {
    return creatures.slice();
  }
  const wanted = typeof tier === 'string' && /^\d+$/.test(tier) ? Number(tier) : tier;
  return creatures.filter((row) => row.tier === wanted);
}

/**
 * Собирает полный Markdown-отчёт по выгрузке: шапка, таблица статов,
 * экономическая таблица и справочник курсов.
 *
 * Отчёт сверен по разделам 2.1 и 2.3 balance/creature-stats.md и пригоден
 * для вставки в вики базы знаний без правок.
 *
 * @param {Object} doc Документ выгрузки.
 * @param {CreatureRow[]} [creatures] Подготовленные строки (по умолчанию — все).
 * @returns {string} Markdown-документ одной строкой.
 */
function renderMarkdownReport(doc, creatures) {
  const rows = (creatures || flattenCreatures(doc)).map(computeDerived);
  const statsTable = toMarkdownTable(rows, STATS_COLUMNS);
  const economyTable = toMarkdownTable(sortBy(rows, 'effective_cost_gold'), ECONOMY_COLUMNS);
  return [
    `# Выгрузка статов существ Кальдеры (сборка ${doc.version})`,
    '',
    `Источник истины: balance/creature-stats.md; выгрузка сгенерирована ${doc.generated}.`,
    'Урон пишется цифрами через дефис; эффективные цены — по городскому курсу слитка.',
    '',
    '## Сводная таблица статов',
    '',
    statsTable,
    '',
    '## Экономика пирамиды: что даёт голова за своё золото',
    '',
    economyTable,
    '',
    '## Справочник тарифов',
    '',
    `- Клейменый слиток: ${INGOT_CITY_RATE_GOLD} золота в городе, ${INGOT_ERATHIAN_RATE_GOLD} у эрафийских торговцев.`,
    `- Ремонт големов: ${GOLEM_REPAIR_GOLD_PER_HP} золота за ХП, без изменений с 0.7.3.`,
    `- Коридор дельты апгрейда: ${UPGRADE_CORRIDOR.min * 100}-${UPGRADE_CORRIDOR.max * 100}%.`,
    `- Потолок прироста гарпий: ${HARPY_WEEKLY_CAP} головы в неделю, не поднимается постройками.`,
    `- Итог недельной пирамиды: ${formatNumber(WEEKLY_HEADS_TOTAL)} голов, ${formatNumber(WEEKLY_GOLD_TOTAL)} золота и 1 слиток.`,
    '',
  ].join('\n');
}

/**
 * Печатает справку по ключам командной строки.
 *
 * @returns {string} Текст справки на русском языке.
 */
function usage() {
  return [
    'Использование: node stats_exporter.js [опции]',
    '',
    'Опции:',
    '  --input <путь>   файл выгрузки JSON (по умолчанию ../data/creature-stats.json)',
    '  --format <вид>   table | csv | md | json (по умолчанию table)',
    '  --tier <уровень> фильтр по уровню: 1-7, особое, все',
    '  --sort <поле>    сортировка: ai_value, health, cost_gold, name ...',
    '  --columns <набор> stats | economy (набор колонок для csv/md)',
    '  --out <путь>     записать результат в файл вместо печати',
    '  --help           эта справка',
    '',
    'Примеры:',
    '  node tools/stats_exporter.js --format csv --tier 7',
    '  node tools/stats_exporter.js --format md --sort ai_value --out pyramid.md',
  ].join('\n');
}

/**
 * Разбирает аргументы командной строки в объект опций.
 *
 * Ручной разбор вместо библиотек: утилита не должна тянуть зависимости ради
 * семи флагов. Неизвестные флаги игнорируются с предупреждением в stderr —
 * ночные скрипты сборки не должны падать из-за опечатки в cron-строке.
 *
 * @param {string[]} argv Аргументы без пути к скрипту.
 * @returns {Object} Опции: input, format, tier, sort, columns, out, help.
 */
function parseArgs(argv) {
  const options = { input: null, format: 'table', tier: null, sort: null, columns: 'stats', out: null, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case '--input':
        options.input = argv[++i] || null;
        break;
      case '--format':
        options.format = (argv[++i] || 'table').toLowerCase();
        break;
      case '--tier':
        options.tier = argv[++i] || null;
        break;
      case '--sort':
        options.sort = argv[++i] || null;
        break;
      case '--columns':
        options.columns = (argv[++i] || 'stats').toLowerCase();
        break;
      case '--out':
        options.out = argv[++i] || null;
        break;
      case '--help':
      case '-h':
        options.help = true;
        break;
      default:
        process.stderr.write(`[внимание] неизвестный аргумент проигнорирован: ${arg}\n`);
        break;
    }
  }
  if (options.tier !== null && /^\d+$/.test(options.tier)) {
    options.tier = Number(options.tier);
  }
  return options;
}

/**
 * Главная функция CLI: читает выгрузку, готовит строки и печатает результат.
 *
 * @param {string[]} argv Аргументы командной строки.
 * @returns {number} Код возврата: 0 — успех, 1 — ошибка данных или опций.
 */
function main(argv) {
  const options = parseArgs(argv);
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }
  let doc;
  try {
    doc = loadStats(options.input);
  } catch (error) {
    process.stderr.write(`Ошибка: ${error.message}\n`);
    return 1;
  }
  let rows = flattenCreatures(doc);
  rows = filterByTier(rows, options.tier);
  if (options.sort) {
    const descending = options.sort !== 'name';
    rows = sortBy(rows, options.sort, descending);
  }

  let output;
  switch (options.format) {
    case 'csv':
      output = toCSV(rows.map(computeDerived), options.columns === 'economy' ? ECONOMY_COLUMNS : STATS_COLUMNS);
      break;
    case 'md':
      output = toMarkdownTable(rows.map(computeDerived), options.columns === 'economy' ? ECONOMY_COLUMNS : STATS_COLUMNS);
      break;
    case 'json':
      output = JSON.stringify(rows.map(computeDerived), null, 2);
      break;
    case 'table':
      output = renderMarkdownReport(doc, rows);
      break;
    default:
      process.stderr.write(`Неизвестный формат: ${options.format} (допустимо table, csv, md, json)\n`);
      return 1;
  }

  if (options.out) {
    fs.writeFileSync(options.out, `${output}\n`, 'utf8');
    process.stdout.write(`Записано: ${options.out} (${rows.length} существ)\n`);
  } else {
    process.stdout.write(`${output}\n`);
  }
  return 0;
}

/** Экспорт утилит для использования как модуля и запуск CLI при прямом вызове. */
module.exports = {
  INGOT_CITY_RATE_GOLD,
  INGOT_ERATHIAN_RATE_GOLD,
  GOLEM_REPAIR_GOLD_PER_HP,
  UPGRADE_CORRIDOR,
  HARPY_WEEKLY_CAP,
  WEEKLY_HEADS_TOTAL,
  WEEKLY_GOLD_TOTAL,
  DEFAULT_INPUT,
  STATS_COLUMNS,
  ECONOMY_COLUMNS,
  loadStats,
  flattenCreatures,
  formatNumber,
  formatRange,
  effectiveCostGold,
  computeDerived,
  escapeCsvValue,
  toCSV,
  toMarkdownTable,
  sortBy,
  filterByTier,
  renderMarkdownReport,
  usage,
  parseArgs,
  main,
};

if (require.main === module) {
  process.exit(main(process.argv.slice(2)));
}
