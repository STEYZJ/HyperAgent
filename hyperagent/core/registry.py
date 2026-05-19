"""Small typed registry used for pluggable components."""

from typing import Dict, Generic, Iterable, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name-to-object registry with explicit errors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, item: T, *, replace: bool = False) -> None:
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError(f"{self.name} registry key cannot be empty")
        if normalized in self._items and not replace:
            raise KeyError(f"{self.name} registry already contains '{normalized}'")
        self._items[normalized] = item

    def get(self, key: str) -> T:
        normalized = key.strip().lower()
        if normalized not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(
                f"{self.name} registry does not contain '{normalized}'. "
                f"Available: {available}"
            )
        return self._items[normalized]

    def has(self, key: str) -> bool:
        return key.strip().lower() in self._items

    def keys(self) -> Iterable[str]:
        return tuple(sorted(self._items))

    def first(self) -> Optional[T]:
        for key in self.keys():
            return self._items[key]
        return None

