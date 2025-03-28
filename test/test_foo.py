import torch
from trade_rl.foo import bar


def test_bar():
    torch.testing.assert_close(bar(), torch.Tensor([1337]))
