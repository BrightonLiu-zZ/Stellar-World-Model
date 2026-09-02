"""exp10 cells: the three complementarity levers, composed and driven the way the runner drives them.

exp10 adds the first knobs in this project that put DATA other than flux into the training loop, so the
failure modes are new. Three of them are pinned here, edge by edge:

  1. INERTNESS. `decoder_cond_dim=0` and `decorr_weight=0.0` must leave exp00-09 bit-identical, because
     every reused arm in the eval fan (the exp07 hann0p3 reference, the untrained arms, the exp08 menu
     caches) was trained under the old code path and is compared against these cells directly.
  2. GRADIENT REACH. Both new terms exist to change what the ENCODER stores. A conditioned decoder or a
     penalty that never propagates back through mu would produce a plausible run of the wrong cell, and
     nothing downstream would notice -- the same class of silent miss as exp09's nested-knob traps.
  3. FEATURES JOIN. A star whose feature row is missing trains against the standardized zero vector,
     which is a legal value (the train mean), not a crash. The count must therefore be exact.

The rationale for testing this way rather than smoking the GPU queue is unchanged from wave 6/7:
composing `+experiment=exp10/<cell>` and driving `run_epoch` exercises Hydra resolution, the manifest
--> config contract, the features join and the loss path, but NOT the 100-epoch schedule, checkpoint
resume, or W&B. That gap is disclosed in the handoff rather than papered over.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

import swm
from swm.data.dataset import SeqWindowDataset
from swm.models import WorldModel
from swm.train.loop import run_epoch
from swm.train.losses import decorr_loss

CONFIG_DIR = Path(swm.__file__).resolve().parent / "configs"
FEATURES = Path(swm.__file__).resolve().parents[2] / "experiments" / "exp10_features" / "subset_features25.parquet"

# (cell, decoder_cond_dim, decorr_weight, dyn_mode, lambda_dyn) -- the manifest's intent, written out
# independently here so a manifest typo cannot silently agree with itself.
CELLS = [
    ("exp10_cond_dec", 25, 0.0, "fwd_bwd", 60),
    ("exp10_decorr", 0, 5.0, "fwd_bwd", 60),
    ("exp10_multistep", 0, 0.0, "multistep", 20),
]
N_FEATURES = 25 # the exp10 feature table's width; decoder_cond_dim must match it exactly


def compose_cell(cell: str):
    """Resolve one cell the way `python -m swm.train +experiment=exp10/<cell>` does."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=[f"+experiment=exp10/{cell}", "variant=B", "seed=0"])


def tiny_model(cfg, z_dim: int = 8, window: int = 32) -> WorldModel:
    """A 2-stage stand-in with the cell's real dyn_mode and conditioning width, small enough to run on CPU."""
    torch.manual_seed(0)
    return WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=z_dim, window=window,
                      gru_hidden=16, gru_layers=1, dyn_mode=cfg.model.dyn_mode,
                      decoder_cond_dim=int(cfg.model.decoder_cond_dim))


def synthetic_loader(n_batches: int = 2, bsz: int = 2, seq_len: int = 4, window: int = 32,
                     with_feats: bool = False) -> list:
    """Stand in for the packed DataLoader: (x) batches, or (x, feats) when a cell consumes features."""
    torch.manual_seed(1)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(bsz, seq_len, window, 1)
        if with_feats:
            batches.append((x, torch.randn(bsz, N_FEATURES)))
        else:
            batches.append(x)
    return batches


@pytest.mark.parametrize("cell,cond_dim,decorr,dyn_mode,lambda_dyn", CELLS)
def test_cell_resolves_to_its_intended_knobs(cell, cond_dim, decorr, dyn_mode, lambda_dyn):
    cfg = compose_cell(cell)
    assert int(cfg.model.decoder_cond_dim) == cond_dim
    assert float(cfg.train.decorr_weight) == decorr
    assert cfg.model.dyn_mode == dyn_mode and float(cfg.train.lambda_dyn) == lambda_dyn
    # the shipping hann0p3 recipe must be intact underneath, or the cell is not a one-knob delta on it
    aux = cfg.train.recon_aux
    assert aux.type == "combined" and float(aux.weight) == 0.3 and aux.psd_window == "hann"
    assert aux.spectral_floor is None
    assert float(cfg.train.free_bits) == 0.0 # D-E10.8: match the reference exactly
    assert float(cfg.train.impulse_penalty_weight) == 0.0
    assert float(cfg.train.augment.hf_noise_sigma) == 0.0
    assert cfg.data.window == 256 and cfg.data.seq_len == 16
    assert bool(cfg.train.track_recon_only_best) # D-E10.9: best_recon_only is the primary read


@pytest.mark.parametrize("cell,cond_dim,decorr,dyn_mode,lambda_dyn", CELLS)
def test_a_feature_consuming_cell_points_at_the_feature_table(cell, cond_dim, decorr, dyn_mode, lambda_dyn):
    """Either consumer set means the path must be set; neither set means it must stay null (E4 is untouched)."""
    cfg = compose_cell(cell)
    consumes = cond_dim > 0 or decorr > 0.0
    if consumes:
        assert str(cfg.train.features_path).endswith("subset_features25.parquet")
    else:
        assert cfg.train.features_path is None


@pytest.mark.parametrize("cell,cond_dim,decorr,dyn_mode,lambda_dyn", CELLS)
def test_cell_trains_one_epoch(cell, cond_dim, decorr, dyn_mode, lambda_dyn):
    """One real `run_epoch` per cell -- the loss path must run, not merely resolve."""
    cfg = compose_cell(cell)
    cfg.train.amp = False # autocast('cuda') is a no-op on CPU
    cfg.data.window = 32
    cfg.model.z_dim = 8 # run_epoch sizes its KL accumulator from cfg.model.z_dim
    model = tiny_model(cfg)
    loader = synthetic_loader(with_feats=cond_dim > 0 or decorr > 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = run_epoch(model, loader, None, scaler, cfg, beta=0.1, device="cpu", train=False)
    assert torch.isfinite(torch.tensor(out["total"])), f"{cell}: non-finite loss"
    assert out["aux"] > 0.0, f"{cell}: the aux term is silent"
    if decorr > 0.0:
        assert out["decorr"] > 0.0, f"{cell}: the decorrelation term is silent"
    else:
        assert out["decorr"] == 0.0, f"{cell}: decorr must be exactly zero when its weight is zero"


@pytest.mark.parametrize("cell,cond_dim,decorr,dyn_mode,lambda_dyn", CELLS)
def test_the_total_loss_is_exactly_its_documented_terms(cell, cond_dim, decorr, dyn_mode, lambda_dyn):
    """The epoch total must equal recon + w*aux + w_imp*imp + beta*KL + lambda*dyn + w_decorr*decorr.

    Every term is a per-batch mean averaged over batches, so the identity is linear and exact. This is
    what pins "no new term leaked into the objective": for cells with decorr_weight 0 it reproduces the
    exp09 expression term for term, and for E2 it pins the new term's weight as the only difference.
    """
    cfg = compose_cell(cell)
    cfg.train.amp = False
    cfg.data.window = 32
    cfg.model.z_dim = 8
    model = tiny_model(cfg)
    loader = synthetic_loader(with_feats=cond_dim > 0 or decorr > 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    out = run_epoch(model, loader, None, scaler, cfg, beta=0.1, device="cpu", train=False)
    expected = (out["recon"] + float(cfg.train.recon_aux.weight) * out["aux"]
                + float(cfg.train.impulse_penalty_weight) * out["imp"]
                + 0.1 * out["kl_loss"] + float(cfg.train.lambda_dyn) * out["dyn"]
                + float(cfg.train.decorr_weight) * out["decorr"])
    assert out["total"] == pytest.approx(expected, rel=1e-6)


def test_decoder_cond_dim_zero_is_the_exp09_model_bit_for_bit():
    """cond_dim=0 must not perturb parameter shapes, init draws, or forward values.

    The whole eval fan pairs exp10 cells against arms trained before this knob existed, so a default
    that shifted the RNG stream or widened a layer would invalidate every paired comparison at once.
    """
    torch.manual_seed(0)
    reference = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                           gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd")
    torch.manual_seed(0)
    explicit = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                          gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd", decoder_cond_dim=0)
    ref_state, new_state = reference.state_dict(), explicit.state_dict()
    assert list(ref_state.keys()) == list(new_state.keys())
    for key in ref_state:
        assert torch.equal(ref_state[key], new_state[key]), f"init diverged at {key}"
    x = torch.randn(2, 4, 32, 1)
    torch.manual_seed(3)
    ref_out = reference(x)
    torch.manual_seed(3)
    new_out = explicit(x)
    for key in ("recon", "mu_seq", "pred_next"):
        assert torch.equal(ref_out[key], new_out[key]), f"forward diverged on {key}"


def test_conditioning_widens_only_the_first_decoder_layer():
    """D-E10.7 draws the line at the conv stacks: the fc may widen, the architecture may not move."""
    torch.manual_seed(0)
    plain = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd")
    torch.manual_seed(0)
    conditioned = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                             gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd", decoder_cond_dim=N_FEATURES)
    assert plain.decoder.fc.in_features == 8
    assert conditioned.decoder.fc.in_features == 8 + N_FEATURES
    for key, value in plain.state_dict().items():
        if key == "decoder.fc.weight":
            continue
        assert conditioned.state_dict()[key].shape == value.shape, f"{key} changed shape"
    for key, value in plain.encoder.state_dict().items():
        assert torch.equal(conditioned.encoder.state_dict()[key], value), f"encoder moved at {key}"


def test_conditioning_gradient_reaches_the_encoder():
    """The conditioned decoder must still backpropagate into mu, or E1 is a decoder-only experiment."""
    torch.manual_seed(0)
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd", decoder_cond_dim=N_FEATURES)
    x = torch.randn(2, 4, 32, 1)
    feats = torch.randn(2, N_FEATURES)
    out = model(x, feats)
    out["recon"].pow(2).mean().backward()
    reached = False
    for parameter in model.encoder.parameters():
        if parameter.grad is not None and parameter.grad.abs().sum() > 0:
            reached = True
    assert reached, "no encoder parameter received gradient through the conditioned decoder"


def test_conditioning_actually_uses_the_features():
    """Two different feature vectors must give two different reconstructions, or the concat is decorative."""
    torch.manual_seed(0)
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd", decoder_cond_dim=N_FEATURES)
    model.eval()
    x = torch.randn(2, 4, 32, 1)
    torch.manual_seed(7)
    first = model(x, torch.zeros(2, N_FEATURES))["recon"]
    torch.manual_seed(7)
    second = model(x, torch.ones(2, N_FEATURES))["recon"]
    assert not torch.allclose(first, second), "the conditioning vector changed nothing"


def test_missing_conditioning_vector_fails_loud():
    """A cond_dec cell run without a features_path must crash, never train against implicit zeros."""
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd", decoder_cond_dim=N_FEATURES)
    with pytest.raises(AssertionError):
        model(torch.randn(2, 4, 32, 1))


def test_decorr_penalty_gradient_reaches_the_encoder():
    """E2's pressure exists only if it flows back through mu into the encoder weights."""
    torch.manual_seed(0)
    model = WorldModel(in_ch=1, enc_channels=[4, 8], kernel_size=5, z_dim=8, window=32,
                       gru_hidden=16, gru_layers=1, dyn_mode="fwd_bwd")
    out = model(torch.randn(4, 4, 32, 1))
    decorr_loss(out["mu_seq"], torch.randn(4, N_FEATURES)).backward()
    reached = False
    for parameter in model.encoder.parameters():
        if parameter.grad is not None and parameter.grad.abs().sum() > 0:
            reached = True
    assert reached, "the decorrelation penalty never reached the encoder"


def test_decorr_penalty_is_a_correlation_not_a_covariance():
    """D-E10.11: rescaling one latent dim must not change the penalty, or dim 51 absorbs the pressure.

    mu's variance is 84-86% concentrated in a single dimension. A covariance-based penalty would spend
    almost all of its gradient there and act as a disguised variance penalty on that one dim; a
    correlation-based penalty is invariant to per-dim scale, which is exactly what this asserts.
    """
    torch.manual_seed(0)
    mu = torch.randn(6, 4, 8)
    feats = torch.randn(6, N_FEATURES)
    plain = decorr_loss(mu, feats)
    inflated = mu.clone()
    inflated[:, :, 3] *= 50.0 # one dim made to dominate the variance, as dim 51 does in the real latent
    assert decorr_loss(inflated, feats) == pytest.approx(float(plain), rel=1e-4)


def test_decorr_penalty_sits_on_the_scale_its_weight_was_calibrated_against():
    """The term is a mean of squared correlations in [0, 1]; the manifest's weight guess of 5.0 assumes it.

    A latent that copies the features exactly gives 1.0 on each of the z diagonal pairs and ~0 elsewhere,
    so the mean over a 4x4 grid is 0.25. Independent columns give ~0. Both anchors are checked, because a
    term that silently lived on a different scale would make the pilot's "halve the achieved corr2" rule
    unreadable.
    """
    torch.manual_seed(0)
    feats = torch.randn(64, 4)
    copied = feats.unsqueeze(1) # (64, 1, 4): latent dim d IS feature d
    assert float(decorr_loss(copied, feats)) == pytest.approx(0.25, abs=0.02)
    torch.manual_seed(5)
    independent = float(decorr_loss(torch.randn(256, 4, 8), torch.randn(256, N_FEATURES)))
    assert independent < 0.05, "unrelated columns should give a near-zero mean squared correlation"


def write_synthetic_pack(tmp_path: Path, tics: list[int], n_win: int = 6, window: int = 8) -> None:
    """One segment per star in a packed layout, enough for the dataset's index + memmap contract."""
    rows = []
    blocks = []
    cursor = 0
    for seg_idx, tic in enumerate(tics):
        block = np.full((n_win, window), float(seg_idx), dtype=np.float32)
        blocks.append(block)
        rows.append({"seg_id": f"seg{seg_idx}", "tic_id": tic, "sector": 1, "seg_idx": 0,
                     "row_start": cursor, "n_win": n_win})
        cursor += n_win
    (tmp_path / "train_windows.dat").write_bytes(np.concatenate(blocks, axis=0).tobytes())
    pd.DataFrame(rows).to_parquet(tmp_path / "train_index.parquet", index=False)


def write_synthetic_features(tmp_path: Path, tics: list[int]) -> Path:
    """A feature table in the exp10 schema: tic_id, split, feats_missing, then the 25 standardized columns."""
    records = []
    for i, tic in enumerate(tics):
        record = {"tic_id": tic, "split": "train", "feats_missing": False}
        for f in range(N_FEATURES):
            record[f"feat{f}"] = float(i) # the star's index, so a joined row identifies its star
        records.append(record)
    path = tmp_path / "features.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    return path


def test_features_join_covers_every_star_in_the_pack(tmp_path: Path):
    tics = [11, 22, 33]
    write_synthetic_pack(tmp_path, tics)
    features_path = write_synthetic_features(tmp_path, tics)
    dataset = SeqWindowDataset(tmp_path, "train", seq_len=4, window=8, randomize=False,
                               features_path=features_path)
    assert dataset.n_missing_features == 0
    assert dataset.features.shape == (len(tics), N_FEATURES)
    for i in range(len(tics)):
        x, feats = dataset[i]
        assert x.shape == (4, 8, 1)
        assert feats.shape == (N_FEATURES,)
        assert torch.equal(feats, torch.full((N_FEATURES,), float(i))) # the row of THIS star, not its neighbour


def test_a_star_absent_from_the_table_is_counted_not_silently_zeroed(tmp_path: Path):
    """D-E10.10 requires the missing count to be exact: zeros are a legal value, so nothing else flags it."""
    write_synthetic_pack(tmp_path, [11, 22, 33])
    features_path = write_synthetic_features(tmp_path, [11, 22])
    dataset = SeqWindowDataset(tmp_path, "train", seq_len=4, window=8, randomize=False,
                               features_path=features_path)
    assert dataset.n_missing_features == 1
    _, feats = dataset[2]
    assert torch.equal(feats, torch.zeros(N_FEATURES)) # standardized zero vector = the train mean


def test_no_features_path_returns_bare_windows(tmp_path: Path):
    """The exp00-09 dataset contract: an item is a tensor, not a pair."""
    write_synthetic_pack(tmp_path, [11, 22])
    dataset = SeqWindowDataset(tmp_path, "train", seq_len=4, window=8, randomize=False)
    assert dataset.features is None
    assert isinstance(dataset[0], torch.Tensor)


@pytest.mark.skipif(not FEATURES.exists(), reason="exp10 feature table not built yet")
def test_the_shipped_feature_table_matches_what_the_cells_expect():
    """The parquet the runner will actually read: 25 columns, unique TICs, standardized on train, finite."""
    table = pd.read_parquet(FEATURES)
    feature_cols = []
    for column in table.columns:
        if column not in ("tic_id", "split", "feats_missing"):
            feature_cols.append(column)
    assert len(feature_cols) == N_FEATURES
    assert table["tic_id"].is_unique
    assert np.isfinite(table[feature_cols].to_numpy()).all()
    train = table[table["split"] == "train"][feature_cols].to_numpy()
    assert np.abs(train.mean(axis=0)).max() < 1e-6 # train-split standardization, by construction
    assert np.abs(train.std(axis=0) - 1.0).max() < 1e-6
    assert set(table["split"].unique()) == {"train", "val", "test"} # val exists here even though F1 skips it
