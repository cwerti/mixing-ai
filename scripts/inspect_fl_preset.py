"""
Печатает цепочки плагинов по вставкам микшера из .flp/.fst и декодированные параметры
стоковых плагинов (Parametric EQ 2, Soundgoodizer, Blood Overdrive, Stereo Enhancer, VST в Fruity Wrapper).

Пример:
    python scripts/inspect_fl_preset.py data/raw/presets/che/*.fst
    python scripts/inspect_fl_preset.py --all-inserts project.flp

Файлы читаются только как байты; содержимое не исполняется.
"""
import argparse
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

# Настройка кодировки консоли Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Добавление корня проекта в пути импорта Python
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from app.fl import FLParseError, PluginSlot, load_project
from app.fl.plugins import eq2

SKIP_FIELDS = {"raw", "records", "state"}


def describe(slot: PluginSlot) -> str:
    if slot.decode_error:
        return f"[decode error] {slot.decode_error}"
    decoded = slot.decoded
    if decoded is None:
        return f"[raw {len(slot.data)} bytes, not decoded]"
    if isinstance(decoded, eq2.ParametricEQ2):
        return eq2.summary(decoded)
    if slot.wrapper is not None:
        w = slot.wrapper
        return f"VST {w.name!r} by {w.vendor!r} at {w.path!r}, state {len(w.state)} bytes"
    if is_dataclass(decoded):
        parts = []
        for f in fields(decoded):
            if f.name in SKIP_FIELDS:
                continue
            value = getattr(decoded, f.name)
            parts.append(f"{f.name}={value:.3f}" if isinstance(value, float) else f"{f.name}={value}")
        return ", ".join(parts)
    return repr(decoded)


def inspect(path: Path, all_inserts: bool) -> int:
    try:
        project = load_project(path)
    except (FLParseError, OSError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    print(f"\n=== {path.name} (format {project.format}, {len(project.file.events)} events) ===")
    for offset, note in project.file.quirks:
        print(f"  quirk @ {offset}: {note}")
    for insert in project.inserts if all_inserts else project.used_inserts():
        print(f"insert {insert.index} {insert.name or ''}".rstrip())
        for slot in insert.slots:
            label = slot.display_name if slot.display_name and slot.display_name != slot.name else ""
            print(f"  slot {slot.index}: {slot.name}{f' ({label})' if label else ''}")
            print(f"      {describe(slot)}")
    for slot in project.other_plugins:
        print(f"outside mixer: {slot.name} — {describe(slot)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--all-inserts", action="store_true", help="показывать и пустые вставки")
    args = parser.parse_args()
    return max(inspect(p, args.all_inserts) for p in args.files)


if __name__ == "__main__":
    sys.exit(main())
