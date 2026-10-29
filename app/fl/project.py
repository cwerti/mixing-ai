"""
Вставки микшера и цепочки плагинов из .flp проекта или .fst пресета вставки.

Порядок событий одной вставки (проверено на FL 20 .flp и FL 26 .fst):
    149 цвет, 204 имя  -> идут ПЕРЕД 236 той вставки, которую описывают;
    236 флаги          -> открывает вставку (первая в .flp — мастер, индекс 0);
    201 внутреннее имя плагина, 203 отображаемое имя, 213 данные плагина, 98 индекс слота
                       -> 98 закрывает слот (98 без предшествующего 201 — пустой слот);
    235/165/166/154/147 -> маршрутизация, 147 закрывает вставку.

В .fst пресете одной вставки есть ровно один 236, поэтому его индекс 0 — не реальный номер
вставки в проекте. Плагины вне слотов микшера (генераторы каналов, пресет одного плагина)
попадают в ``FLProject.other_plugins``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.fl.events import Event, FLFile, read_fl_file
from app.fl.plugins import PluginDecodeError, decode_plugin
from app.fl.plugins.wrapper import INTERNAL_NAME as WRAPPER_NAME
from app.fl.plugins.wrapper import WrapperInfo

EV_SLOT_INDEX = 98
EV_INSERT_COLOR = 149
EV_INSERT_OUTPUT = 147
EV_PLUGIN_INTERNAL_NAME = 201
EV_PLUGIN_DISPLAY_NAME = 203
EV_INSERT_NAME = 204
EV_PLUGIN_DATA = 213
EV_INSERT_FLAGS = 236


@dataclass
class PluginSlot:
    """Плагин в слоте: имена, сырые данные события 213 и результат декодера (или None)."""

    internal_name: str
    display_name: str | None = None
    data: bytes = b""
    index: int | None = None
    decoded: Any | None = None
    decode_error: str | None = None

    @property
    def name(self) -> str:
        """Имя для показа: имя VST внутри Fruity Wrapper, иначе внутреннее имя."""
        if self.vst_name:
            return self.vst_name
        return self.internal_name

    @property
    def wrapper(self) -> WrapperInfo | None:
        return self.decoded if isinstance(self.decoded, WrapperInfo) else None

    @property
    def vst_name(self) -> str | None:
        return self.wrapper.name if self.wrapper else None

    @property
    def vst_path(self) -> str | None:
        return self.wrapper.path if self.wrapper else None


@dataclass
class MixerInsert:
    index: int
    name: str | None = None
    color: int | None = None
    slots: list[PluginSlot] = field(default_factory=list)

    @property
    def chain(self) -> list[str]:
        """Внутренние имена плагинов по порядку слотов, VST в виде ``Fruity Wrapper(<имя>)``."""
        return [f"{s.internal_name}({s.vst_name})" if s.vst_name else s.internal_name for s in self.slots]


@dataclass
class FLProject:
    file: FLFile
    inserts: list[MixerInsert]
    other_plugins: list[PluginSlot]

    @property
    def format(self) -> int:
        return self.file.format

    def insert_by_name(self, name: str) -> MixerInsert:
        for insert in self.inserts:
            if insert.name == name:
                return insert
        raise KeyError(name)

    def used_inserts(self) -> list[MixerInsert]:
        """Вставки с именем или хотя бы одним плагином."""
        return [i for i in self.inserts if i.name or i.slots]


def _decode_slot(slot: PluginSlot) -> PluginSlot:
    try:
        slot.decoded = decode_plugin(slot.internal_name, slot.data)
    except PluginDecodeError as exc:
        slot.decode_error = str(exc)
    return slot


def build_project(fl_file: FLFile) -> FLProject:
    """Собирает вставки и слоты из уже разобранного потока событий."""
    inserts: list[MixerInsert] = []
    others: list[PluginSlot] = []
    pending_name: str | None = None
    pending_color: int | None = None
    current: MixerInsert | None = None
    plugin: PluginSlot | None = None

    def flush_unslotted() -> None:
        # Плагин, не закрытый событием 98, не принадлежит слоту микшера.
        if plugin is not None and plugin.internal_name:
            others.append(_decode_slot(plugin))

    event: Event
    for event in fl_file.events:
        eid = event.id
        if eid == EV_INSERT_NAME:
            pending_name = event.text
        elif eid == EV_INSERT_COLOR:
            pending_color = int(event.value)  # type: ignore[arg-type]
        elif eid == EV_INSERT_FLAGS:
            flush_unslotted()
            plugin = None
            current = MixerInsert(index=len(inserts), name=pending_name, color=pending_color)
            inserts.append(current)
            pending_name = pending_color = None
        elif eid == EV_PLUGIN_INTERNAL_NAME:
            flush_unslotted()
            plugin = PluginSlot(internal_name=event.text)
        elif eid == EV_PLUGIN_DISPLAY_NAME and plugin is not None:
            plugin.display_name = event.text
        elif eid == EV_PLUGIN_DATA and plugin is not None:
            plugin.data = event.data
        elif eid == EV_SLOT_INDEX:
            if plugin is not None and current is not None and plugin.internal_name:
                plugin.index = int(event.value)  # type: ignore[arg-type]
                current.slots.append(_decode_slot(plugin))
            plugin = None
        elif eid == EV_INSERT_OUTPUT:
            flush_unslotted()
            plugin = None
            current = None
    flush_unslotted()
    return FLProject(file=fl_file, inserts=inserts, other_plugins=others)


def load_project(path: str | Path) -> FLProject:
    """Читает .flp/.fst и возвращает вставки с цепочками плагинов."""
    return build_project(read_fl_file(path))
