import pandas as pd

from spaq_iqa.splitting import assert_group_safe, assign_groups, build_groups, materialize_manifest


def test_duplicate_component_never_crosses_splits() -> None:
    metadata = pd.DataFrame(
        {
            "Image name": [f"{index:05d}.jpg" for index in range(30)],
            "MOS": [float(index) for index in range(30)],
        }
    )
    edges = [("00000.jpg", "00001.jpg"), ("00001.jpg", "00002.jpg")]
    groups = build_groups(metadata, edges)
    manifest = materialize_manifest(groups, assign_groups(groups, seed=7))
    assert_group_safe(manifest, edges)
    component = manifest[manifest["image_name"].isin(["00000.jpg", "00001.jpg", "00002.jpg"])]
    assert component["group_id"].nunique() == 1
    assert component["split"].nunique() == 1


def test_split_is_deterministic() -> None:
    metadata = pd.DataFrame(
        {"Image name": [f"{index:05d}.jpg" for index in range(100)], "MOS": range(100)}
    )
    groups = build_groups(metadata, [])
    first = materialize_manifest(groups, assign_groups(groups, seed=20261109))
    second = materialize_manifest(groups, assign_groups(groups, seed=20261109))
    pd.testing.assert_frame_equal(first, second)
