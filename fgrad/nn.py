import random
from dataclasses import dataclass
from typing import Final

from fgrad.value import Value


class Module:
    def parameters(self) -> list[Value]:
        return []


@dataclass
class Neuron(Module):
    w: Final[list[Value]]
    b: Final[Value]
    nonlin: bool = True

    @classmethod
    def uniform(cls, nin: int, nonlin: bool = True):
        return cls([Value.from_data(random.uniform(-1, 1)) for _ in range(nin)], Value.from_data(0.0), nonlin)

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


@dataclass
class Layer(Module):
    neurons: Final[list[Neuron]]

    @classmethod
    def uniform(cls, nin: int, nout: int, nonlin: bool = True):
        return cls([Neuron.uniform(nin, nonlin=nonlin) for _ in range(nout)])

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


@dataclass
class MLP(Module):
    layers: Final[list[Layer]]

    @classmethod
    def uniform(cls, nin: int, nouts: list[int]):
        szs = [nin] + nouts
        return cls([Layer.uniform(
            nin,
            nout,
            nonlin=i != len(nouts) - 1
        ) for ((i, nin), nout) in zip(enumerate(szs), szs[1:])])

    def __call__(self, x: list[Value]) -> list[Value]:
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for l in self.layers for p in l.parameters()]