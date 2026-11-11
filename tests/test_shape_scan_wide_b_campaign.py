"""Cutoff analysis and crash/restart contract for the wide-b scan CLI."""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_shape_scan_wide_b as campaign


def test_cutoff_metrics_ratio_and_radial_yield():
    q = np.array([0., .02, .03, .06, .1])
    spectrum = 1. - q/.1
    spectra = np.asarray([[spectrum, 1.5*spectrum], [3*spectrum, 6*spectrum]])
    case = dict(name="test", density_id="skin", beta2=.1, beta3=.2)
    rows = campaign.cutoff_metrics(case, 19, q, [60., 10000.], spectra)
    ref, last = rows[:2]
    assert ref["qdq_yield"] == pytest.approx(.1**2/6)
    assert last["Q10_GeV"] == pytest.approx(.09)
    assert last["K_tail_over_first_cutoff"] == pytest.approx(1.)
    assert last["qdq_yield_over_first_cutoff"] == pytest.approx(3.)
    assert ref["qdq_yield_over_last_cutoff"] == pytest.approx(1/3)
    assert rows[-1]["qdq_yield_over_first_cutoff"] == pytest.approx(4.)


def test_committed_prefix_and_averaging(tmp_path, monkeypatch):
    from upc_deformation import shape_scan_wide_b

    calls = []

    class DummyForward:
        def __init__(self, cfg, case, *, device, step_scale, cutoffs):
            self.cutoffs = np.asarray(cutoffs)
            self.stored_q = np.linspace(0., .17, 64)
            self.b_nodes = np.array([12., *cutoffs])

        def calculate(self, r1, r2):
            calls.extend(zip(r1, r2))
            if len(calls) == 4:
                raise InterruptedError("uncommitted chunk")
            slopes = (40. + 10.*r1[:, 0, 0])[:, None]
            profile = np.exp(-slopes*self.stored_q)
            return profile[:, None, None, :] * np.asarray([[1., 2.], [4., 6.]])[None, :, :, None]

    monkeypatch.setattr(shape_scan_wide_b, "WideBRefiner", DummyForward)
    args = SimpleNamespace(config=campaign.ROOT/"configs/shape_scan_smoke.yaml", output=tmp_path,
                           cases=["none_example__b2_001__b3_000"], all_cases=False,
                           n_orientations=5, cutoffs=[60., 10000.], step_scale=1.,
                           device="cpu", batch_size=2, checkpoint_size=2, threads=1)
    with pytest.raises(InterruptedError):
        campaign.run(args)
    directory = tmp_path/"cases"/args.cases[0]
    assert len(list(directory.glob("chunk_*.npz"))) == 1
    campaign.run(args)
    assert len(calls) == 7  # two pairs retried, the committed first two are reused
    chunks = sorted(directory.glob("chunk_*.npz"))
    all_a0 = np.concatenate([np.load(path)["A0"] for path in chunks])
    with np.load(directory/"mean_A0.npz") as mean:
        np.testing.assert_allclose(mean["mean_A0"], all_a0.mean(axis=0))
        expected = campaign.extract_tail_features(mean["stored_q"], mean["mean_A0"][0, 0])
    with (tmp_path/"cutoff_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["Q10_GeV"]) == expected["Q10_GeV"]
    campaign.run(args)
    assert len(calls) == 7
    args.step_scale = .5
    with pytest.raises(ValueError, match="identity differ"):
        campaign.run(args)
    assert len(list(directory.glob("chunk_*.npz"))) == 3
