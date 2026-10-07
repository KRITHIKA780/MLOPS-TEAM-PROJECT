import pandas as pd

from src.ab_testing import analyse, assign_variant, min_sample_size, two_proportion_ztest


def test_assignment_is_deterministic_and_balanced():
    assert all(assign_variant(u) == assign_variant(u) for u in range(100))
    share_b = sum(assign_variant(u) == "B" for u in range(5000)) / 5000
    assert 0.45 < share_b < 0.55


def test_ztest_detects_clear_difference():
    _, p = two_proportion_ztest(300, 1000, 200, 1000)
    assert p < 0.001
    _, p = two_proportion_ztest(101, 1000, 100, 1000)
    assert p > 0.5


def test_decision_requires_significance_and_sample():
    big = pd.DataFrame({"variant": ["A", "B"], "impressions": [200000, 200000], "clicks": [10000, 11000]})
    assert analyse(big)["decision"] == "promote_B"
    tiny = pd.DataFrame({"variant": ["A", "B"], "impressions": [100, 100], "clicks": [5, 9]})
    assert analyse(tiny)["decision"] != "promote_B"
    assert min_sample_size(0.05) > 1000
