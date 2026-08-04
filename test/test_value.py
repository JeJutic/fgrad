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