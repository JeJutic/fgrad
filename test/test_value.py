from fgrad.value import Value, backward
import micrograd.engine as micrograd


def test_sanity_check():

    x = Value.from_data(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    grads = backward(y)
    xfg, yfg = x, y

    x = micrograd.Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    assert yfg.data == ymg.data
    assert grads[xfg] == xmg.grad

def test_more_ops():

    a = Value.from_data(-4.0)
    b = Value.from_data(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    grads = backward(g)
    afg, bfg, gfg = a, b, g

    a = micrograd.Value(-4.0)
    b = micrograd.Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gfg.data - gmg.data) < tol
    # backward pass went well
    assert abs(grads[afg] - amg.grad) < tol
    assert abs(grads[bfg] - bmg.grad) < tol