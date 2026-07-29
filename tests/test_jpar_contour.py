"""Unit tests for the main-well J∥ contour diagnostic (connectivity layer)."""

import numpy as np
import pytest

from desc.integrals.jpar_contour import (
    default_tol,
    diagnose,
    diagnose_margin,
    firm3d_vpar_pitch_weights,
    hard_range,
    loss_cone_fraction,
    minimax_margin,
    p_loss_from_margin,
    persistence_check,
    reaches_wall_flood_fill,
    select_main_well,
    weighted_fraction,
)


class TestSelectMainWell:
    """Main-well selection from multi-well arrays."""

    def test_picks_deepest_B(self):
        # wells: shallow, deep, missing
        J_all = np.array([1.0, 2.0, 0.0])
        B_bot = np.array([1.2, 0.8, 0.0])
        z1 = np.array([0.1, 1.0, 0.0])
        z2 = np.array([0.5, 1.5, 0.0])
        J_m, mask, idx, Bb, dz, zm1, zm2 = select_main_well(J_all, B_bot, z1, z2)
        assert idx == 1
        assert mask
        np.testing.assert_allclose(J_m, 2.0)
        np.testing.assert_allclose(Bb, 0.8)
        np.testing.assert_allclose(dz, 0.5)
        np.testing.assert_allclose(zm1, 1.0)
        np.testing.assert_allclose(zm2, 1.5)

    def test_no_valid_well(self):
        J_all = np.zeros(3)
        B_bot = np.zeros(3)
        z1 = np.zeros(3)
        z2 = np.zeros(3)
        J_m, mask, idx, Bb, dz, zm1, zm2 = select_main_well(J_all, B_bot, z1, z2)
        assert not mask
        assert np.isnan(J_m)
        assert np.isnan(dz)
        assert np.isnan(zm1)
        assert np.isnan(zm2)


class TestPersistenceAndFloodFill:
    """Synthetic (s, α) connectivity tests."""

    def test_hard_range_empty(self):
        assert hard_range(np.array([1.0]), np.array([False])) == (-np.inf, np.inf)

    def test_persistence_pinch(self):
        # J varies with α on surface 0, collapses away from J0 on surface 1.
        J = np.array(
            [
                [1.0, 2.0, 3.0, 2.0],
                [0.0, 0.1, 0.2, 0.1],  # cannot host J0=2
                [1.5, 2.0, 2.5, 2.0],
            ]
        )
        mask = np.ones_like(J, dtype=bool)
        out = persistence_check(
            J, mask, s_idx0=0, alpha_idx0=1, s_grid=np.array([0.1, 0.5, 0.9])
        )
        assert not out["connected"]
        assert out["pinch_idx"] == 1
        np.testing.assert_allclose(out["pinch_s"], 0.5)

    def test_persistence_ok(self):
        J = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.5, 2.0, 2.5],
                [1.8, 2.1, 2.2],
            ]
        )
        mask = np.ones_like(J, dtype=bool)
        out = persistence_check(J, mask, 0, 1)
        assert out["connected"]
        np.testing.assert_allclose(out["J0"], 2.0)

    def test_flood_fill_open_to_wall(self):
        # Constant J band from seed to wall along α=0 column.
        J = np.full((5, 4), 10.0)
        mask = np.ones_like(J, dtype=bool)
        reached, visited = reaches_wall_flood_fill(J, mask, 1, 0, tol=0.1, wall_idx=4)
        assert reached
        assert visited[4, 0]

    def test_flood_fill_blocked_gap(self):
        J = np.full((5, 4), 10.0)
        mask = np.ones_like(J, dtype=bool)
        # Break radial connection at s=2 for all α.
        mask[2, :] = False
        reached, visited = reaches_wall_flood_fill(J, mask, 0, 0, tol=0.1, wall_idx=4)
        assert not reached
        assert not visited[3, 0]

    def test_flood_fill_disjoint_branch(self):
        # Two disjoint α-bands share the same J; seed sits in the confined core patch.
        # Use mid-α indices so periodicity does not bridge the branches.
        J = np.full((4, 6), 5.0)
        mask = np.zeros_like(J, dtype=bool)
        mask[0:2, 1:3] = True  # confined core patch (does not reach wall_idx)
        mask[:, 4:6] = True  # separate open channel to the wall
        J[0:2, 1:3] = 1.0
        J[:, 4:6] = 1.0
        reached, visited = reaches_wall_flood_fill(J, mask, 0, 1, tol=0.05, wall_idx=3)
        assert not reached
        assert visited[0, 1] and visited[1, 2]
        assert not visited[0, 4]

    def test_alpha_periodicity(self):
        J = np.full((3, 4), 2.0)
        mask = np.zeros_like(J, dtype=bool)
        mask[0, 0] = True
        mask[0, 3] = True  # neighbor across α wrap
        mask[1, 3] = True
        mask[2, 3] = True
        reached, visited = reaches_wall_flood_fill(J, mask, 0, 0, tol=0.1, wall_idx=2)
        assert reached
        assert visited[0, 3]


class TestMinimaxMargin:
    """Tol-free bottleneck path."""

    def test_exact_level_set_zero_margin(self):
        J = np.full((5, 4), 10.0)
        mask = np.ones_like(J, dtype=bool)
        m = minimax_margin(J, mask, 1, 0, wall_idx=4)
        np.testing.assert_allclose(m, 0.0)

    def test_must_climb(self):
        # Seed J=0; wall only reachable through cells with J=2.
        J = np.zeros((4, 3))
        J[1:, :] = 2.0
        mask = np.ones_like(J, dtype=bool)
        m = minimax_margin(J, mask, 0, 1, wall_idx=3)
        np.testing.assert_allclose(m, 2.0)

    def test_masked_barrier_inf(self):
        J = np.full((5, 4), 1.0)
        mask = np.ones_like(J, dtype=bool)
        mask[2, :] = False
        m = minimax_margin(J, mask, 0, 0, wall_idx=4)
        assert np.isinf(m)

    def test_disjoint_positive_margin_or_inf(self):
        # Same topology as flood-fill disjoint test: seed in core patch.
        J = np.full((4, 6), 5.0)
        mask = np.zeros_like(J, dtype=bool)
        mask[0:2, 1:3] = True
        mask[:, 4:6] = True
        J[0:2, 1:3] = 1.0
        J[:, 4:6] = 1.0
        m = minimax_margin(J, mask, 0, 1, wall_idx=3)
        # No masked path from core to wall → inf
        assert np.isinf(m)

    def test_eight_neighbor_diagonal_bridge(self):
        # 4-neighbor blocked; diagonal opens path of constant J.
        J = np.full((3, 3), 0.0)
        mask = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        )
        m4 = minimax_margin(J, mask, 0, 0, wall_idx=2, eight_neighbor=False)
        m8 = minimax_margin(J, mask, 0, 0, wall_idx=2, eight_neighbor=True)
        assert np.isinf(m4)
        np.testing.assert_allclose(m8, 0.0)

    def test_diagnose_margin_and_fraction(self):
        J = np.zeros((4, 3, 2))
        mask = np.ones_like(J, dtype=bool)
        J[..., 0] = 1.0  # exact open
        J[:, :, 1] = np.array([0.0, 1.0, 2.0, 3.0])[:, None]  # climb to wall
        seeds = [(0, 0), (0, 1)]
        lambdas = np.array([0.5, 0.8])
        margins, flags = diagnose_margin(J, mask, seeds, lambdas)
        assert margins.shape == (2, 2)
        np.testing.assert_allclose(margins[:, 0], 0.0)
        np.testing.assert_allclose(margins[:, 1], 3.0)
        assert set(flags.ravel()) == {"ok"}
        p0 = p_loss_from_margin(margins, 0.0)
        np.testing.assert_allclose(weighted_fraction(p0), 0.5)
        p_soft = p_loss_from_margin(margins, 3.0, soft=True)
        # pitch0: p=1; pitch1: margin=3 → p=0
        np.testing.assert_allclose(weighted_fraction(p_soft), 0.5)

    def test_diagnose_margin_barely_trapped_filter(self):
        J = np.ones((4, 3, 2))
        mask = np.ones_like(J, dtype=bool)
        # Seed (0,0) pitch 0 is barely trapped (large Δζ); others OK.
        dz = np.full_like(J, 0.5)
        dz[0, 0, 0] = 10.0  # ≫ 2 π / nfp=3 ≈ 2.09
        seeds = [(0, 0), (0, 1)]
        lambdas = np.array([0.5, 0.8])
        margins, flags = diagnose_margin(
            J, mask, seeds, lambdas, delta_zeta=dz, nfp=3
        )
        assert flags[0, 0] == "barely trapped"
        assert np.isinf(margins[0, 0])
        assert flags[0, 1] == "ok"
        assert flags[1, 0] == "ok"
        elig = flags != "barely trapped"
        # only one of four seed×pitch pairs excluded from eligibility beyond NMW
        assert elig.sum() == 3


class TestFirm3dPitchWeights:
    """Uniform-v∥ (firm3d) pitch measure on a 1/λ grid."""

    def test_voronoi_covers_trapped_xi_interval(self):
        B = 5.0
        # ξ = 0.2, 0.5, 0.8 → p = B/(1-ξ²)
        xi = np.array([0.2, 0.5, 0.8])
        p = B / (1.0 - xi**2)
        w = firm3d_vpar_pitch_weights(p, B)
        # Edges at 0, midpoints, and ξ_max=0.8 → total mass = 0.8
        np.testing.assert_allclose(w.sum(), xi[-1])
        assert np.all(w > 0)
        # Deeper-trapped (smaller ξ) gets the [0, mid] cell
        assert w[0] > w[1]

    def test_invalid_when_Bcrit_below_birth_B(self):
        w = firm3d_vpar_pitch_weights(np.array([4.0, 6.0, 8.0]), 5.0)
        assert w[0] == 0.0
        assert w[1] > 0.0 and w[2] > 0.0


class TestDiagnoseAggregate:
    """End-to-end diagnose + scalar aggregation on toy data."""

    def test_diagnose_and_fraction(self):
        # Pitch 0: open to wall; pitch 1: pinched.
        J = np.zeros((4, 3, 2))
        mask = np.ones_like(J, dtype=bool)
        J[..., 0] = 1.0
        J[:, :, 1] = np.array([1.0, 0.5, 0.3, 0.2])[:, None]

        seeds = [(0, 0), (0, 1)]
        lambdas = np.array([0.5, 0.8])
        report, verdicts = diagnose(
            J, mask, seeds, lambdas, s_grid=np.linspace(0.1, 1.0, 4), tol=0.15
        )
        assert verdicts.shape == (2, 2)
        assert set(verdicts[:, 0]) == {"loss cone"}
        assert set(verdicts[:, 1]) == {"blocked"}
        frac = loss_cone_fraction(verdicts)
        np.testing.assert_allclose(frac, 0.5)
        assert len(report) == 4

    def test_default_tol_positive(self):
        J = np.array([[1.0, 1.1], [1.05, 1.2]])
        mask = np.ones_like(J, dtype=bool)
        tol = default_tol(J, mask, 0, 0)
        assert tol > 0
