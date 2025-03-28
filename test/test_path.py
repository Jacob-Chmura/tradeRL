from trade_rl.util.path import get_run_id


def test_get_run_id_default_args():
    run_id = get_run_id()

    assert isinstance(run_id, str)
    assert run_id.startswith('tradeRL/')
    assert run_id.count('/') == 1
    assert run_id.count('_') == 2


def test_get_run_id():
    run_id = get_run_id('mock')
    assert run_id.startswith('mock/')
    assert run_id.count('/') == 1
    assert run_id.count('_') == 2
