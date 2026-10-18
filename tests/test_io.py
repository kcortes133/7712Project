from cd8_subtyper import io


def test_reference_has_17_subtypes(ref_df):
    assert ref_df["Cluster"].nunique() == 17
    assert (ref_df["ES"] >= 0.15).all()
    assert ref_df["SigFreq"].between(0, 1).all()


def test_every_subtype_has_a_state(ref_df):
    states = io.load_subtype_states()
    assert set(states) == set(ref_df["Cluster"])
    assert set(states.values()) == {"naive", "memory", "effector_memory", "effector", "exhausted", "other"}


def test_bundled_datasets():
    names = io.available_datasets()
    assert len(names) == 6
    for name in names:
        df = io.load_tisch_clusters(io.find_tisch_file(name))
        assert list(df.columns) == ["Cluster", "Gene", "ES"]
        assert io.list_cd8_clusters(df), name


def test_marker_sets():
    sets = io.load_marker_sets()
    assert "PDCD1" in sets["Exhaustion marker"]
