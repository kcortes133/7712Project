import pytest

from cd8_subtyper.similarity import jaccard, marker_overlap, rank_subtypes


@pytest.mark.parametrize("a, b, expected", [
    ({"A", "B"}, {"A", "B"}, 1.0),
    ({"A"}, {"B"}, 0.0),
    ({"A", "B", "C"}, {"B", "C", "D"}, 0.5),
    (set(), set(), 0.0),
    (["A", "A", "B"], ["B"], 0.5),  # duplicates in lists are ignored
])
def test_jaccard(a, b, expected):
    assert jaccard(a, b) == pytest.approx(expected)


def test_jaccard_is_symmetric():
    assert jaccard({"A", "B"}, {"B", "C", "D"}) == jaccard({"B", "C", "D"}, {"A", "B"})


def test_rank_subtypes_orders_best_first():
    ref = {"x": {"A", "B"}, "y": {"A", "B", "C"}, "z": {"Q"}}
    ranking = rank_subtypes({"A", "B", "C"}, ref)
    assert [name for name, _ in ranking] == ["y", "x", "z"]
    assert ranking[0][1] == 1.0


def test_marker_overlap():
    markers = {"exhaustion": ["PDCD1", "TOX", "HAVCR2"], "naive": ["SELL"]}
    assert marker_overlap({"TOX", "PDCD1", "GZMK"}, markers) == {"exhaustion": ["PDCD1", "TOX"], "naive": []}
