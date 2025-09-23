from metrics import MetricsCollector


def test_metrics_increment_and_snapshot():
    mc = MetricsCollector()
    mc.increment("test_counter")
    mc.increment("test_counter", 2)
    mc.timing("test_timer", 50)
    snap = mc.snapshot()
    assert snap["counters"]["test_counter"] == 3
    assert "test_timer" in snap["timers"]


