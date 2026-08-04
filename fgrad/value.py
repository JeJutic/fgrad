from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final

from fgrad.topo import topo_sort


@dataclass(eq=False, repr=False)
class Value:
    data: float
    children: Final[tuple[Value, ...]]

    # could be moved to a separate class Const
    @classmethod
    def from_data(cls, data: float) -> Value:
        return cls(data, ())

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value.from_data(other)
        return Add.construct(self, other)

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value.from_data(other)
        return Mult.construct(self, other)

    def __pow__(self, power: int | float) -> Value:
        return Pow.construct(self, power)

    def relu(self) -> Value:
        return Relu.construct(self)

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | float) -> Value:
        return other + (-self)

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other ** (-1)

    def __rtruediv__(self, other: Value | float) -> Value:
        return other * self ** (-1)

    def __repr__(self):
        return f"Value({self.data})"


@dataclass(eq=False, repr=False)
class Add(Value):
    a: Final[Value]
    b: Final[Value]

    @classmethod
    def construct(cls, a: Value, b: Value) -> Add:
        return cls(a.data + b.data, (a, b), a, b)


@dataclass(eq=False, repr=False)
class Mult(Value):
    a: Final[Value]
    b: Final[Value]

    @classmethod
    def construct(cls, a: Value, b: Value) -> Mult:
        return cls(a.data * b.data, (a, b), a, b)


@dataclass(eq=False, repr=False)
class Pow(Value):
    a: Final[Value]
    power: Final[float | int]

    @classmethod
    def construct(cls, a: Value, power: float | int) -> Pow:
        return cls(a.data ** power, (a,), a, power)


@dataclass(eq=False, repr=False)
class Relu(Value):
    a: Final[Value]

    @classmethod
    def construct(cls, a: Value) -> Relu:
        return cls(a.data if a.data > 0 else 0.0, (a,), a)


def backward(y: Value) -> dict[Value, float]:
    grads: dict[Value, float] = defaultdict(float)
    grads[y] = 1.0

    def _backward(u: Value) -> None:
        match u:
            case Add(_, _, a, b):
                grads[a] += grads[u]
                grads[b] += grads[u]
            case Mult(_, _, a, b):
                grads[a] += b.data * grads[u]
                grads[b] += a.data * grads[u]
            case Pow(_, _, a, power):
                grads[a] += power * a.data ** (power - 1) * grads[u]
            case Relu(_, _, a):
                grads[a] += grads[u] if u.data > 0.0 else 0

    for v in topo_sort(y, lambda u: u.children):
        _backward(v)

    return grads
