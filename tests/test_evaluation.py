import pytest

from cd8_subtyper import io
from cd8_subtyper.evaluation import TISCH_EXPECTED_STATE, chance_hit_at_k, hit_at_k


@pytest.fixture(scope="module")
def states():
    return io.load_subtype_states()


def test_every_tisch_label_maps_to_a_real_state(states):
    assert set(TISCH_EXPECTED_STATE.values()) <= set(states.values())


def test_hit_at_k(states):
    ranked = ["CD8.c06(GZMK+ Tem)", "CD8.c11(GZMK+ Tex)", "CD8.c01(Tn)"]
    assert not hit_at_k(ranked, "CD8Tex", states, 1)
    assert hit_at_k(ranked, "CD8Tex", states, 2)
    assert hit_at_k(ranked, "CD8Tem", states, 1)


def test_chance(states):
    assert chance_hit_at_k("CD8Tex", states, 1) == pytest.approx(4 / 17)  # four exhausted subtypes
    assert chance_hit_at_k("CD8Tn", states, 1) == pytest.approx(1 / 17)
    assert chance_hit_at_k("CD8Tn", states, 3) == pytest.approx(3 / 17)
