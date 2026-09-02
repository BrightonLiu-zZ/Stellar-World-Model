"""The runner's run ORDER is a manifest knob, and it decides what a -MaxHours cutoff leaves finished.

`sweep.order` sat in the exp09 manifest as a comment-shaped key that gen_sweep never read: the
generator hardcoded seed-major, so anyone reading the manifest believed they were configuring
something inert. These tests pin both orders and, more importantly, pin that a manifest which does
NOT declare the key still generates the seed-major runner exp06-exp08 were built with.
"""
from __future__ import annotations

import pytest

from swm.exp.gen_sweep import expand_cells, render_runner


def manifest(order: str | None) -> dict:
    """Two cells with deliberately different seed lists, so the two orders cannot coincide."""
    sweep: dict = {
        "cells": [
            {"exp_name": "cell_a", "geometry": "w256", "seeds": [0, 1],
             "overrides": {"train.lambda_dyn": 60}},
            {"exp_name": "cell_b", "geometry": "w256", "seeds": [0, 1],
             "overrides": {"train.lambda_dyn": 0}},
        ],
        "seeds": [0, 1],
        "budget_hours": 1,
        "per_run_minutes": {"ep100": 25},
    }
    if order is not None:
        sweep["order"] = order
    return {
        "name": "exp99_ordering",
        "plan": "docs/plans/none.md",
        "env": {"python": "C:/py.exe"},
        "paths": {"packed_sources": {"w256": "experiments/packed_w256"}},
        "base": {"data": {"window": 256, "seq_len": 16},
                 "model": {"dyn_mode": "fwd_bwd"},
                 "train": {"lambda_dyn": 60, "max_epochs": 100}},
        "recipes": {"r": {}},
        "sweep": sweep,
        "eval": {"variant": "B"},
    }


def runner_for(order: str | None) -> str:
    spec = manifest(order)
    return render_runner(spec, expand_cells(spec), __import__("pathlib").Path("m.yaml"))


def test_absent_order_key_keeps_the_seed_major_runner():
    """exp06-exp08 declare no `order`; their generated runners must not change under this feature."""
    assert runner_for(None) == runner_for("seed-major")


def test_seed_major_walks_seeds_outermost():
    text = runner_for("seed-major")
    assert "foreach ($seed in @(0, 1))" in text
    assert "if ($c[2] -contains $seed)" in text
    assert "even seed coverage" in text


def test_cell_major_walks_cells_outermost():
    """Each cell's whole seed list finishes before the next cell starts -- a cutoff leaves cells DONE."""
    text = runner_for("cell-major")
    assert "foreach ($c in $cells) {" in text
    assert "foreach ($seed in $c[2])" in text
    assert "if ($c[2] -contains $seed)" not in text  # the seed-major guard has no place here
    assert "LEADING cells complete" in text


def test_unknown_order_is_rejected_rather_than_silently_ignored():
    """The failure this feature exists to prevent: a manifest key that reads as config and is inert."""
    with pytest.raises(AssertionError, match="sweep.order"):
        runner_for("random")


@pytest.mark.parametrize("order", [None, "seed-major", "cell-major"])
def test_runner_stays_pure_ascii(order):
    """PS 5.1 reads the .ps1 as ANSI; a stray non-ASCII char is a ParserError at launch time."""
    assert all(ord(ch) < 128 for ch in runner_for(order))
