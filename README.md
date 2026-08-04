# fgrad

![woof!](puppy.jpg)

A variation on theme _what if [karpathy/micrograd](https://github.com/karpathy/micrograd)_ was written
by a functional programmer.

## Key differences

1. Operation-specific immutable classes via `@final` and `@dataclass`
    It allows a greater separation of concerns: now we don't need to define
    chain rule for the operation at the time it is being constructed.
    We can define it in a separate function which pattern matches on known operations.
    We could also write `eval` function instead of computing
    expression result on construction.
2. Gradient computation does not change existing variables
    Even though mutability is still used under the hood (dict), one gradient computation
    won't intervene with another. Hence, you do not need to use `.zero_grad()`
3. Moved topological sort to a separate file
4. Changed puppy photo
5. Changed `random.seed(1337)` to `random.seed(42)` for better performance

## AI usage report

Local autocomplete and web search only.