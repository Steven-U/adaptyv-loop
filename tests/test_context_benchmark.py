from backtest.run_context_benchmark import run_benchmark


def test_context_benchmark_adapts_faster_than_context_free():
    summary = run_benchmark(80)
    contextual = summary["context+recency"]
    baseline = summary["context-free"]

    assert contextual["stale_tests"] < baseline["stale_tests"]
    assert contextual["binders"] > baseline["binders"]
    assert contextual["recovery_round"] < baseline["recovery_round"]
