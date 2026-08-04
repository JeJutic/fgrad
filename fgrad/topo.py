from typing import Callable


def topo_sort[T](x: T, children: Callable[[T], tuple[T, ...]]) -> list[T]:
    res = []
    visited = set()

    def build(v: T) -> None:
        if v not in visited:
            visited.add(v)
            for child in children(v):
                build(child)
            res.append(v)

    build(x)
    return list(reversed(res))