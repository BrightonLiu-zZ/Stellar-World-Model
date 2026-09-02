"""Tests for the C1/C2 supervised baselines (roadmap D20 / Y13b).

The load-bearing ones are the FAIRNESS tests, not the shape tests. C1's whole protocol is "identical
splits and populations to the F1 scorecard", and the cheapest way for that to be quietly false is for
the supervised population to drift from the one the mu caches were built on. So the split-identity
tests compare against the actual F1 mu caches rather than against the manifest's own numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from swm.data.labelled import StarBags, task_targets
from swm.exp.gen_supervised import expand_runs, pilot_runs, render
from swm.models.supervised import SupervisedNet, pool_bags
from swm.train.supervised import metric_floor, score

repo_root = Path(__file__).resolve().parents[3]
MANIFEST_PATH = repo_root / "experiments" / "configs" / "c1c2_supervised_baselines.yaml"
SUBSET_MU = repo_root / "experiments" / "exp08_menu_channel" / "subset_mu_cache" / "hann0p3_fbwd_s0.npz"
POOL_MU = repo_root / "experiments" / "exp08_menu_channel" / "mu_cache" / "hann0p3_fbwd_s0.npz"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ------------------------------------------------------------------------------------ bag pooling
def test_pool_bags_is_the_per_star_mean():
    """The star score of D20: mean over the star's window outputs, with ragged bags."""
    window_out = torch.tensor([1.0, 3.0, 10.0, -2.0, 0.0, 2.0])
    star_index = torch.tensor([0, 0, 1, 2, 2, 2])
    pooled = pool_bags(window_out, star_index, 3)
    assert torch.allclose(pooled, torch.tensor([2.0, 10.0, 0.0]))


def test_pool_bags_backpropagates_into_every_window():
    """A featureless window must be able to receive gradient, which is why pooling is in-graph."""
    window_out = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
    star_index = torch.tensor([0, 0, 1])
    pool_bags(window_out, star_index, 2).sum().backward()
    assert torch.allclose(window_out.grad, torch.tensor([0.5, 0.5, 1.0]))


def test_star_bags_gather_and_subset_preserve_grouping():
    windows = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    bags = StarBags(np.array([11, 22, 33]), windows, np.array([2, 5, 3]))
    picked, bag_ids = bags.gather(np.array([2, 0]))
    assert picked.shape == (5, 4)
    assert bag_ids.tolist() == [0, 0, 0, 1, 1]
    assert np.allclose(picked[0], windows[7])
    kept = bags.subset(np.array([False, True, True]))
    assert kept.tics.tolist() == [22, 33]
    assert len(kept.windows) == 8


# ------------------------------------------------------------------------------------- the two arms
@pytest.mark.parametrize("trunk,kwargs", [("conv", {"enc_channels": [32, 64, 128, 256], "kernel_size": 5}),
                                          ("dense", {"hidden": [256, 256]})])
def test_both_arms_score_one_value_per_window(trunk, kwargs):
    net = SupervisedNet(trunk=trunk, window=256, z_dim=128, dropout=0.2, **kwargs)
    out = net(torch.randn(7, 256, 1))
    assert out.shape == (7,)


def test_conv_arm_reuses_the_shipped_encoder_and_carries_no_posterior():
    """The trunk must BE the Encoder's stack, and there must be no dead logvar head."""
    net = SupervisedNet(trunk="conv", window=256, z_dim=128, dropout=0.2,
                        enc_channels=[32, 64, 128, 256], kernel_size=5)
    assert net.flat_dim == 256 * 16 # window 256 through four stride-2 pools
    assert net.to_z.in_features == 4096 and net.to_z.out_features == 128
    names = []
    for name, _ in net.named_parameters():
        names.append(name)
    assert not any("logvar" in name for name in names), "the supervised arm has no posterior"


def test_the_two_arms_differ_only_in_the_trunk():
    conv = SupervisedNet("conv", 256, 128, 0.2, enc_channels=[32, 64, 128, 256], kernel_size=5)
    dense = SupervisedNet("dense", 256, 128, 0.2, hidden=[256, 256])
    assert conv.head.in_features == dense.head.in_features == 128
    assert conv.to_z.out_features == dense.to_z.out_features == 128
    assert float(conv.dropout.p) == float(dense.dropout.p)


# ------------------------------------------------------------------------------------------ metrics
def test_metric_floors_are_metric_native():
    y = np.array([0, 0, 0, 1])
    assert metric_floor("detection", y) == pytest.approx(0.25) # prevalence = a random ranker's PR-AUC
    assert metric_floor("contrastive", y) == 0.5
    assert metric_floor("regression", y) == 0.0


def test_score_dispatches_on_task_shape():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert score("detection", y, perfect) == pytest.approx(1.0)
    assert score("contrastive", y, perfect) == pytest.approx(1.0)
    assert score("regression", np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------- manifest
def test_manifest_covers_the_eleven_scorecard_tasks(manifest):
    names = []
    for task in manifest["tasks"]:
        names.append(task["name"])
    assert len(names) == 11 and len(set(names)) == 11
    menu = {"numax_hon", "rotation_period", "osc_giant", "solar_like_osc", "rgb_vs_heb", "ijspeert", "flare"}
    v1 = {"pulsating", "eb", "rotation", "transit"}
    assert set(names) == menu | v1
    assert "ijspeert_excl_villanova" not in names # ADR-0010: one probe per physical quantity


def test_manifest_target_transforms_match_the_probe(manifest):
    """numax is regressed in log10 and rotation_period linearly, exactly as new_task_scorecard does."""
    transforms = {}
    for task in manifest["tasks"]:
        if task["shape"] == "regression":
            transforms[task["name"]] = task["target_transform"]
    assert transforms == {"numax_hon": "log10", "rotation_period": "linear"}


def test_queue_expansion_is_arm_major_and_complete(manifest):
    runs = expand_runs(manifest)
    assert len(runs) == 2 * 11 * 3
    assert len(set(runs)) == len(runs)
    first_arm = manifest["queue"]["arm_order"][0]
    assert runs[0] == (first_arm, manifest["queue"]["order"][0], 0)
    assert runs[32][0] == first_arm and runs[33][0] == manifest["queue"]["arm_order"][1]


def test_pilot_is_one_arm_one_seed_every_task(manifest):
    pilot = pilot_runs(manifest)
    assert len(pilot) == 11
    assert {arm for arm, _, _ in pilot} == {"conv_supervised"}
    assert {seed for _, _, seed in pilot} == {0}
    assert set(pilot).issubset(set(expand_runs(manifest)))


def test_generated_runner_is_ascii_and_resumable(manifest):
    text = render(manifest, MANIFEST_PATH)
    text.encode("ascii") # PS 5.1 mis-parses non-ASCII; fail here rather than in the user's terminal
    assert "DO NOT EDIT" in text
    assert "DONE.txt" in text and "PilotOnly" in text and "MaxHours" in text


# ------------------------------------------------------------- fairness: the populations must match F1
@pytest.mark.skipif(not SUBSET_MU.exists(), reason="F1 v1 mu cache not present")
def test_v1_population_matches_the_f1_mu_cache(manifest):
    """The v1 star set and order must equal the cache F1's published numbers were scored on."""
    from swm.data.labelled import load_bags
    cache = np.load(SUBSET_MU, allow_pickle=False)
    for split in ["train", "test"]:
        bags = load_bags("v1", split, 256, repo_root / "experiments" / "c1c2_supervised" / "data_cache")
        assert bags.tics.tolist() == cache[f"{split}_tics"].tolist()
        assert bags.counts.tolist() == cache[f"{split}_counts"].tolist()


@pytest.mark.skipif(not POOL_MU.exists(), reason="F1 pool mu cache not present")
def test_manifest_pool_counts_match_the_f1_mu_cache(manifest):
    """Cheap version of the same check for the pool: counts only, no npz replay in the test suite."""
    cache = np.load(POOL_MU, allow_pickle=False)
    by_name = {}
    for task in manifest["tasks"]:
        by_name[task["name"]] = task
    for split in ["train", "val", "test"]:
        assert by_name["osc_giant"]["n"][split] == len(cache[f"{split}_tics"])


POOL_BAG_CACHE = repo_root / "experiments" / "c1c2_supervised" / "data_cache" / "pool_test_w256.npz"


@pytest.mark.skipif(not (POOL_MU.exists() and POOL_BAG_CACHE.exists()),
                    reason="pool bag cache not built yet (12 min npz replay); skipped rather than paid here")
def test_pool_population_matches_the_f1_mu_cache_star_for_star():
    """The strong form of the pool check, once the bag cache exists: same stars, same order, same bags."""
    from swm.data.labelled import load_bags
    cache = np.load(POOL_MU, allow_pickle=False)
    bags = load_bags("pool", "test", 256, POOL_BAG_CACHE.parent)
    assert bags.tics.tolist() == cache["test_tics"].tolist()
    assert bags.counts.tolist() == cache["test_counts"].tolist()


def test_v1_manifest_counts_match_the_packed_index(manifest):
    import pandas as pd
    packed = repo_root / "experiments" / "exp01_window256_seq16" / "packed"
    by_name = {}
    for task in manifest["tasks"]:
        by_name[task["name"]] = task
    for split in ["train", "val", "test"]:
        index = pd.read_parquet(packed / f"{split}_index.parquet")
        assert by_name["eb"]["n"][split] == index["tic_id"].nunique()


def test_every_task_resolves_to_the_f1_test_population(manifest):
    """The footing check: all 11 keep masks reproduce F1's published n_test and n_test_pos.

    This is the cheapest place a fairness break would show up -- a renamed label column, a keep mask
    that stopped matching the probe's, a catalog regenerated underneath us. It runs on the test split
    only (the label frame is the expensive part, not the split) and needs no bag cache.
    """
    import pandas as pd
    from swm.data.labelled import star_label_frame, task_targets
    labels = star_label_frame()
    packed = pd.read_parquet(repo_root / "experiments" / "exp01_window256_seq16" / "packed"
                             / "test_index.parquet")
    v1_tics = np.array(sorted(packed["tic_id"].unique()), dtype=np.int64)
    pool = pd.read_parquet(repo_root / "processed" / "subset" / "new_task_pool.parquet")
    pool_tics = np.array(sorted(pool.loc[pool["split"] == "test", "tic_id"].astype(int)), dtype=np.int64)

    for task in manifest["tasks"]:
        tics = v1_tics if task["population"] == "v1" else pool_tics
        y, keep = task_targets(task, tics, labels)
        assert int(keep.sum()) == task["n"]["test"], f"{task['name']}: kept population moved"
        assert np.isfinite(y[keep]).all(), f"{task['name']}: a kept star has a non-finite target"
        if task["shape"] != "regression":
            assert int(y[keep].sum()) == task["n"]["pos_test"], f"{task['name']}: positive count moved"


# -------------------------------------------------------------------------------------- keep masks
def test_task_targets_keep_kinds(manifest):
    import pandas as pd
    tics = np.array([1, 2, 3, 4])
    labels = pd.DataFrame({"eb": [1.0, 0.0, np.nan, 1.0],
                           "numax_hon": [10.0, np.nan, 100.0, np.nan],
                           "rotation_period": [1.0, 7.0, 3.0, 2.0],
                           "rotation": [1, 1, 0, 1]}, index=tics)

    y, keep = task_targets({"name": "eb", "keep": "all", "label": {"column": "eb"}}, tics, labels)
    assert keep.all() and y.tolist() == [1.0, 0.0, 0.0, 1.0] # absent detection label reads as 0

    y, keep = task_targets({"name": "numax_hon", "keep": "labelled", "target_transform": "log10",
                            "label": {"column": "numax_hon"}}, tics, labels)
    assert keep.tolist() == [True, False, True, False]
    assert y[keep].tolist() == [1.0, 2.0] # log10 applied, matching REGRESSION's log_target

    y, keep = task_targets({"name": "rotation_period", "keep": "tars_rotator_le5d",
                            "target_transform": "linear", "label": {"column": "rotation_period"}},
                           tics, labels)
    assert keep.tolist() == [True, False, False, True] # >5 d dropped, non-rotator dropped
    assert y[keep].tolist() == [1.0, 2.0]
