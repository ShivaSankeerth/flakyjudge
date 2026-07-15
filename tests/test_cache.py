import pytest

from flakyjudge.cache import BudgetExceededError, RequestKey, ResponseCache


def make_key(**overrides) -> RequestKey:
    defaults = dict(
        provider="openai-compat",
        model="gpt-4o-2024-11-20",
        system="sys",
        prompt="Query: q\n\nResponse: r\n\nUnit Test: t",
        temperature=0.0,
        max_tokens=8,
        logprobs=True,
        repeat_idx=0,
    )
    defaults.update(overrides)
    return RequestKey(**defaults)


@pytest.fixture
def cache(tmp_path):
    c = ResponseCache(tmp_path / "cache.db")
    yield c
    c.close()


class TestRoundTrip:
    def test_miss_then_hit(self, cache):
        key = make_key()
        assert cache.get(key) is None
        response = {"choices": [{"message": {"content": "4"}}]}
        cache.put(key, response, input_tokens=750, output_tokens=1, cost_usd=0.002)
        assert cache.get(key) == response

    def test_persists_across_connections(self, tmp_path):
        path = tmp_path / "cache.db"
        first = ResponseCache(path)
        first.put(make_key(), {"ok": True}, 10, 1, 0.001)
        first.close()
        second = ResponseCache(path)
        assert second.get(make_key()) == {"ok": True}
        second.close()


class TestKeyIdentity:
    def test_repeat_idx_distinguishes_resamples(self, cache):
        cache.put(make_key(repeat_idx=0), {"score": "4"}, 10, 1, 0.001)
        assert cache.get(make_key(repeat_idx=1)) is None

    def test_any_field_change_changes_key(self):
        base = make_key().digest()
        assert make_key(prompt="other").digest() != base
        assert make_key(temperature=1.0).digest() != base
        assert make_key(model="gpt-4o-mini-2024-07-18").digest() != base

    def test_digest_is_deterministic(self):
        assert make_key().digest() == make_key().digest()


class TestSpendLedger:
    def test_total_and_per_model(self, cache):
        cache.put(make_key(repeat_idx=0), {}, 10, 1, 0.5)
        cache.put(make_key(repeat_idx=1), {}, 10, 1, 0.25)
        cache.put(make_key(model="other", repeat_idx=0), {}, 10, 1, 1.0)
        assert cache.total_spend() == pytest.approx(1.75)
        assert cache.spend_by_model()["other"] == pytest.approx(1.0)

    def test_budget_hard_stop(self, cache, monkeypatch):
        monkeypatch.setenv("FLAKYJUDGE_MAX_SPEND_USD", "1.0")
        cache.put(make_key(), {}, 10, 1, 2.0)
        with pytest.raises(BudgetExceededError):
            cache.check_budget()

    def test_budget_ok_under_limit(self, cache, monkeypatch):
        monkeypatch.setenv("FLAKYJUDGE_MAX_SPEND_USD", "50.0")
        cache.put(make_key(), {}, 10, 1, 0.01)
        cache.check_budget()
