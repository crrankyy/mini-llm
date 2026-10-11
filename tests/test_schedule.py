import pytest

from mini_llm.training.schedule import learning_rate_at

TOTAL, PEAK, WARMUP, FINAL = 100, 1.0, 10, 0.1


def rate(update, warmup=WARMUP):
    return learning_rate_at(update, TOTAL, PEAK, warmup, FINAL)


def test_the_rate_rises_in_equal_steps_and_the_last_warmup_update_is_made_at_the_peak():
    assert [rate(update) for update in range(WARMUP)] == pytest.approx([0.1 * n for n in range(1, WARMUP + 1)])
    # the first update is not made at a rate of zero, which would waste it
    assert rate(0) > 0


def test_the_rate_falls_along_a_cosine_from_the_peak_to_the_final_value():
    assert rate(WARMUP) == pytest.approx(PEAK)
    assert rate((WARMUP + TOTAL) // 2) == pytest.approx((PEAK + FINAL) / 2)

    falling = [rate(update) for update in range(WARMUP, TOTAL)]

    assert falling == sorted(falling, reverse=True)
    assert FINAL < falling[-1] < FINAL + 0.01


def test_the_rate_stays_at_the_final_value_once_the_planned_updates_are_made():
    assert rate(TOTAL) == rate(TOTAL + 1000) == FINAL


def test_without_a_warmup_the_first_update_is_made_at_the_peak():
    assert rate(0, warmup=0) == pytest.approx(PEAK)
