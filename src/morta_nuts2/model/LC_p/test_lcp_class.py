# -*- coding: utf-8 -*-
"""
Tests unitaires pour lcp_class.py
Couvre :
  - LeeCarter.make_penalty_matrix
  - LeeCarter.poisson_lnL
  - LeeCarter.update_ax_coef
  - LeeCarter.update_kappa
  - LeeCarter.compute_fit_stats
  - LeeCarter.build_input_from_dataframe
  - LeeCarter.Parametric.Multiregion.compute_logmu
  - LeeCarter.Parametric.Multiregion.rescale_bx_kappa
  - LeeCarter.Parametric.Multiregion.__init__
  - LeeCarter.Classic.__init__ + fit (smoke test)

Run :
    python -m pytest test_lcp_class.py -v
    # ou
    python -m unittest test_lcp_class -v
"""

import unittest
import numpy as np
import pandas as pd
import sys, os
from unittest.mock import MagicMock

# ── Mock du module externe morta_nuts2 (non requis pour les tests unitaires) ─
# make_bspline_basis  : retourne (B, knots, n_basis)
# eval_bspline_from_coef : retourne un vecteur de zéros

def _fake_make_bspline_basis(xv, degree, n_knots, xmin, xmax):
    n_basis = n_knots + degree
    B       = np.ones((len(xv), n_basis)) / n_basis
    knots   = np.linspace(xmin, xmax, n_basis + degree + 1)
    return B, knots, n_basis

def _fake_eval_bspline_from_coef(coef, xv, knots, degree):
    return np.zeros(len(xv))

mock_bsplines_module = MagicMock()
mock_bsplines_module.make_bspline_basis      = _fake_make_bspline_basis
mock_bsplines_module.eval_bspline_from_coef  = _fake_eval_bspline_from_coef

sys.modules["morta_nuts2"]                             = MagicMock()
sys.modules["morta_nuts2.model"]                       = MagicMock()
sys.modules["morta_nuts2.model.Bsplines"]              = MagicMock()
sys.modules["morta_nuts2.model.Bsplines.Bsplines"]     = mock_bsplines_module

# ── Ajuste le chemin si nécessaire ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from morta_nuts2.model.LC_p.lcp_class import LeeCarter


# =============================================================================
# Helpers partagés
# =============================================================================

def _make_fake_data(nb_ages=10, nb_years=8, nb_regions=2, seed=42):
    """Génère des données (Extg, Dxtg, xv, tv) de test reproductibles."""
    rng = np.random.default_rng(seed)
    xv  = np.arange(nb_ages, dtype=float)
    tv  = np.arange(nb_years, dtype=float)
    Extg = rng.uniform(100, 500, (nb_ages, nb_years, nb_regions))
    # Taux de mortalité plausibles : 0.001 – 0.05
    mu   = rng.uniform(1e-3, 5e-2, (nb_ages, nb_years, nb_regions))
    Dxtg = rng.poisson(Extg * mu).astype(float)
    Dxtg = np.maximum(Dxtg, 0.0)
    return Extg, Dxtg, xv, tv


def _make_fake_dataframe(nb_ages=6, nb_years=5, nb_regions=2, seed=7):
    """Génère un DataFrame long compatible avec build_input_from_dataframe."""
    rng = np.random.default_rng(seed)
    rows = []
    for age in range(nb_ages):
        for year in range(nb_years):
            for reg in range(nb_regions):
                exp   = rng.uniform(100, 400)
                mu    = rng.uniform(1e-3, 4e-2)
                deaths = max(0.0, rng.poisson(exp * mu))
                rows.append({
                    "age": age, "year": year, "region": reg,
                    "deaths": deaths, "exposure": exp, "mortality_rate": mu
                })
    return pd.DataFrame(rows)


# =============================================================================
# 1. make_penalty_matrix
# =============================================================================

class TestMakePenaltyMatrix(unittest.TestCase):

    def test_shape(self):
        """DtD doit être (n_basis, n_basis) et diag de taille (n_basis,)."""
        n = 8
        DtD, diag = LeeCarter.make_penalty_matrix(n, diff_order=2)
        self.assertEqual(DtD.shape, (n, n))
        self.assertEqual(diag.shape, (n,))

    def test_symmetric(self):
        """D^T D est symétrique."""
        DtD, _ = LeeCarter.make_penalty_matrix(10, diff_order=2)
        np.testing.assert_array_almost_equal(DtD, DtD.T)

    def test_positive_semidefinite(self):
        """Toutes les valeurs propres doivent être ≥ 0."""
        DtD, _ = LeeCarter.make_penalty_matrix(10, diff_order=2)
        eigenvalues = np.linalg.eigvalsh(DtD)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_diag_matches_diagonal(self):
        """diag doit correspondre à la diagonale de DtD."""
        DtD, diag = LeeCarter.make_penalty_matrix(7, diff_order=1)
        np.testing.assert_array_almost_equal(diag, np.diag(DtD))

    def test_diff_order_1_vs_2(self):
        """L'ordre 2 doit produire une matrice différente de l'ordre 1."""
        DtD1, _ = LeeCarter.make_penalty_matrix(8, diff_order=1)
        DtD2, _ = LeeCarter.make_penalty_matrix(8, diff_order=2)
        self.assertFalse(np.allclose(DtD1, DtD2))


# =============================================================================
# 2. poisson_lnL
# =============================================================================

class TestPoissonLnL(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.Extg  = rng.uniform(50, 200, (5, 4, 2))
        self.logmu = rng.uniform(-6, -2, (5, 4, 2))
        self.Dxtg  = np.maximum(rng.poisson(self.Extg * np.exp(self.logmu)), 0.0)
        from scipy.special import gammaln
        self.logFact = gammaln(self.Dxtg + 1)

    def test_returns_four_elements(self):
        result = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertEqual(len(result), 4)

    def test_lnL_is_float(self):
        lnL, *_ = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertIsInstance(lnL, float)

    def test_lnL_is_finite(self):
        lnL, *_ = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertTrue(np.isfinite(lnL))

    def test_exp_logmu_shape(self):
        _, exp_logmu, *_ = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertEqual(exp_logmu.shape, self.logmu.shape)

    def test_residual_shape(self):
        _, _, _, residual = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertEqual(residual.shape, self.Dxtg.shape)

    def test_exp_logmu_positive(self):
        _, exp_logmu, *_ = LeeCarter.poisson_lnL(self.Dxtg, self.Extg, self.logmu, self.logFact)
        self.assertTrue(np.all(exp_logmu > 0))


# =============================================================================
# 3. update_ax_coef
# =============================================================================

class TestUpdateAxCoef(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.n_basis  = 6
        self.nb_ages  = 10
        self.nb_years = 5
        self.nb_reg   = 2
        self.ax_coef  = rng.normal(0, 0.1, self.n_basis)
        self.B        = rng.uniform(0, 1, (self.nb_ages, self.n_basis))
        shape         = (self.nb_ages, self.nb_years, self.nb_reg)
        self.residual     = rng.normal(0, 1, shape)
        self.weighted_exp = np.abs(rng.normal(10, 2, shape)) + 0.1
        self.DtD, self.diag_DtD = LeeCarter.make_penalty_matrix(self.n_basis)

    def test_shape_preserved(self):
        new_coef = LeeCarter.update_ax_coef(
            self.ax_coef, self.B, self.residual, self.weighted_exp,
            eta=0.1, lam=0.1, DtD=self.DtD, diag_DtD=self.diag_DtD
        )
        self.assertEqual(new_coef.shape, self.ax_coef.shape)

    def test_no_nan(self):
        new_coef = LeeCarter.update_ax_coef(
            self.ax_coef, self.B, self.residual, self.weighted_exp,
            eta=0.1, lam=0.1, DtD=self.DtD, diag_DtD=self.diag_DtD
        )
        self.assertFalse(np.any(np.isnan(new_coef)))

    def test_zero_penalty(self):
        """Avec lam=0 le code suit un chemin différent (sans pénalité)."""
        new_coef = LeeCarter.update_ax_coef(
            self.ax_coef, self.B, self.residual, self.weighted_exp,
            eta=0.1, lam=0.0, DtD=self.DtD, diag_DtD=self.diag_DtD
        )
        self.assertEqual(new_coef.shape, self.ax_coef.shape)

    def test_does_not_modify_input(self):
        """La fonction ne doit pas modifier ax_coef en place."""
        original = self.ax_coef.copy()
        LeeCarter.update_ax_coef(
            self.ax_coef, self.B, self.residual, self.weighted_exp,
            eta=0.1, lam=0.1, DtD=self.DtD, diag_DtD=self.diag_DtD
        )
        np.testing.assert_array_equal(self.ax_coef, original)


# =============================================================================
# 4. update_kappa
# =============================================================================

class TestUpdateKappa(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(2)
        self.nb_ages  = 8
        self.nb_years = 6
        self.nb_reg   = 3
        shape = (self.nb_ages, self.nb_years, self.nb_reg)
        self.kappa        = rng.normal(0, 1, self.nb_years)
        self.bx_reg       = np.abs(rng.normal(0.1, 0.05, (self.nb_ages, self.nb_reg)))
        self.residual     = rng.normal(0, 1, shape)
        self.weighted_exp = np.abs(rng.normal(10, 2, shape)) + 0.1

    def test_shape_preserved(self):
        new_kappa = LeeCarter.update_kappa(
            self.kappa, self.bx_reg, self.residual, self.weighted_exp, eta=0.1
        )
        self.assertEqual(new_kappa.shape, self.kappa.shape)

    def test_no_nan(self):
        new_kappa = LeeCarter.update_kappa(
            self.kappa, self.bx_reg, self.residual, self.weighted_exp, eta=0.1
        )
        self.assertFalse(np.any(np.isnan(new_kappa)))

    def test_does_not_modify_input(self):
        original = self.kappa.copy()
        LeeCarter.update_kappa(
            self.kappa, self.bx_reg, self.residual, self.weighted_exp, eta=0.1
        )
        np.testing.assert_array_equal(self.kappa, original)


# =============================================================================
# 5. compute_fit_stats
# =============================================================================

class TestComputeFitStats(unittest.TestCase):

    def setUp(self):
        from scipy.special import gammaln
        rng = np.random.default_rng(3)
        self.nb_ages, self.nb_years, self.nb_reg = 10, 8, 2
        self.Extg  = rng.uniform(100, 500, (self.nb_ages, self.nb_years, self.nb_reg))
        self.logmu = rng.uniform(-6, -2,   (self.nb_ages, self.nb_years, self.nb_reg))
        self.Dxtg  = np.maximum(
            rng.poisson(self.Extg * np.exp(self.logmu)).astype(float), 0.0
        )
        self.logFact = gammaln(self.Dxtg + 1)

    def test_returns_dataframe(self):
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, self.logmu, self.logFact,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        self.assertIsInstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, self.logmu, self.logFact,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        for col in ["N", "n_basis", "dofs", "lnL", "deviance", "AIC", "BIC"]:
            self.assertIn(col, df.columns)

    def test_single_row(self):
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, self.logmu, self.logFact,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        self.assertEqual(len(df), 1)

    def test_deviance_nonnegative_when_logmu_is_saturated(self):
        """La déviance est >= 0 quand logmu correspond au modèle saturé (log(D/E))."""
        safe_D = np.maximum(self.Dxtg, 1e-12)
        logmu_sat = np.log(safe_D / self.Extg)
        from scipy.special import gammaln
        logFact_sat = gammaln(self.Dxtg + 1)
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, logmu_sat, logFact_sat,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        self.assertGreaterEqual(df["deviance"].iloc[0], -1e-6)  # tolérance numérique

    def test_deviance_is_float(self):
        """La déviance doit être un nombre fini."""
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, self.logmu, self.logFact,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        self.assertTrue(np.isfinite(df["deviance"].iloc[0]))

    def test_n_matches_array_size(self):
        df = LeeCarter.compute_fit_stats(
            self.Dxtg, self.Extg, self.logmu, self.logFact,
            n_basis=6, nb_years=self.nb_years, nb_regions=self.nb_reg
        )
        self.assertEqual(df["N"].iloc[0], self.Dxtg.size)


# =============================================================================
# 6. build_input_from_dataframe
# =============================================================================

class TestBuildInputFromDataframe(unittest.TestCase):

    def setUp(self):
        self.df = _make_fake_dataframe(nb_ages=5, nb_years=4, nb_regions=3)

    def test_returns_six_elements(self):
        result = LeeCarter.build_input_from_dataframe(self.df)
        self.assertEqual(len(result), 6)

    def test_shapes(self):
        Muxtg, Dxtg, Extg, xv, tv, regions = LeeCarter.build_input_from_dataframe(self.df)
        nb_ages, nb_years, nb_reg = 5, 4, 3
        self.assertEqual(Muxtg.shape, (nb_ages, nb_years, nb_reg))
        self.assertEqual(Dxtg.shape,  (nb_ages, nb_years, nb_reg))
        self.assertEqual(Extg.shape,  (nb_ages, nb_years, nb_reg))
        self.assertEqual(len(xv), nb_ages)
        self.assertEqual(len(tv), nb_years)
        self.assertEqual(len(regions), nb_reg)

    def test_exposure_positive(self):
        _, _, Extg, *_ = LeeCarter.build_input_from_dataframe(self.df)
        self.assertTrue(np.all(Extg > 0))

    def test_deaths_nonnegative(self):
        _, Dxtg, *_ = LeeCarter.build_input_from_dataframe(self.df)
        self.assertTrue(np.all(Dxtg >= 0))

    def test_xv_sorted(self):
        _, _, _, xv, *_ = LeeCarter.build_input_from_dataframe(self.df)
        np.testing.assert_array_equal(xv, np.sort(xv))

    def test_tv_sorted(self):
        _, _, _, _, tv, _ = LeeCarter.build_input_from_dataframe(self.df)
        np.testing.assert_array_equal(tv, np.sort(tv))


# =============================================================================
# 7. Parametric.Multiregion — compute_logmu & rescale_bx_kappa
# =============================================================================

class TestMultiregionComputeLogmu(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(4)
        self.nb_ages   = 10
        self.nb_years  = 6
        self.nb_regions = 3
        self.n_basis   = 5
        self.ax_coef   = rng.normal(0, 0.1, self.n_basis)
        self.bx_coef   = rng.normal(0, 0.05, (self.nb_regions, self.n_basis))
        self.kappa     = rng.normal(0, 1, self.nb_years)
        self.xv        = np.linspace(0, 9, self.nb_ages)
        self.B         = np.random.default_rng(5).uniform(0, 1, (self.nb_ages, self.n_basis))
        # knots factices (non utilisés dans le calcul direct via B @)
        self.knots     = np.linspace(0, 9, self.n_basis + 3)

    def test_logmu_shape(self):
        logmu, ax, bx_reg = LeeCarter.Parametric.Multiregion.compute_logmu(
            self.ax_coef, self.bx_coef, self.kappa, self.xv,
            self.B, self.knots, degree=2
        )
        self.assertEqual(logmu.shape, (self.nb_ages, self.nb_years, self.nb_regions))

    def test_ax_shape(self):
        _, ax, _ = LeeCarter.Parametric.Multiregion.compute_logmu(
            self.ax_coef, self.bx_coef, self.kappa, self.xv,
            self.B, self.knots, degree=2
        )
        self.assertEqual(ax.shape, (self.nb_ages,))

    def test_bx_reg_shape(self):
        _, _, bx_reg = LeeCarter.Parametric.Multiregion.compute_logmu(
            self.ax_coef, self.bx_coef, self.kappa, self.xv,
            self.B, self.knots, degree=2
        )
        self.assertEqual(bx_reg.shape, (self.nb_ages, self.nb_regions))

    def test_logmu_finite(self):
        logmu, *_ = LeeCarter.Parametric.Multiregion.compute_logmu(
            self.ax_coef, self.bx_coef, self.kappa, self.xv,
            self.B, self.knots, degree=2
        )
        self.assertTrue(np.all(np.isfinite(logmu)))


class TestRescaleBxKappa(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(6)
        self.nb_ages   = 12
        self.nb_regions = 4
        self.n_basis   = 6
        self.bx_coef   = rng.uniform(0.01, 0.1, (self.nb_regions, self.n_basis))
        self.bx_reg    = rng.uniform(0.01, 0.1, (self.nb_ages, self.nb_regions))
        self.kappa     = rng.normal(0, 1, 8)

    def test_normalization_after_rescale(self):
        """Après rescaling, sum(mean_g β_{x,g}) doit être proche de 1."""
        bx_coef_new, bx_reg_new, _ = LeeCarter.Parametric.Multiregion.rescale_bx_kappa(
            self.bx_coef, self.bx_reg, self.kappa
        )
        bx_avg = np.mean(bx_reg_new, axis=1)
        self.assertAlmostEqual(float(np.sum(bx_avg)), 1.0, places=10)

    def test_kappa_scaled_inversely(self):
        """kappa doit être multiplié par le même facteur que bx_reg est divisé."""
        scal = float(np.sum(np.mean(self.bx_reg, axis=1)))
        _, _, kappa_new = LeeCarter.Parametric.Multiregion.rescale_bx_kappa(
            self.bx_coef, self.bx_reg, self.kappa
        )
        np.testing.assert_array_almost_equal(kappa_new, self.kappa * scal)

    def test_zero_scale_factor_safe(self):
        """Avec bx_reg = 0, la fonction doit retourner les valeurs inchangées."""
        bx_coef_zero = np.zeros_like(self.bx_coef)
        bx_reg_zero  = np.zeros_like(self.bx_reg)
        bx_coef_out, bx_reg_out, kappa_out = LeeCarter.Parametric.Multiregion.rescale_bx_kappa(
            bx_coef_zero, bx_reg_zero, self.kappa
        )
        np.testing.assert_array_equal(bx_coef_out, bx_coef_zero)
        np.testing.assert_array_equal(bx_reg_out,  bx_reg_zero)


# =============================================================================
# 8. Parametric.Multiregion.__init__
# =============================================================================

class TestMultiregionInit(unittest.TestCase):

    def test_default_params(self):
        m = LeeCarter.Parametric.Multiregion()
        self.assertEqual(m.degree, 2)
        self.assertEqual(m.n_knots, 10)
        self.assertIsNone(m.xmin)
        self.assertIsNone(m.xmax)
        self.assertEqual(m.lam, 0.0)
        self.assertEqual(m.diff_order, 2)
        self.assertEqual(m.nb_iter, 800)
        self.assertFalse(m.verbose)

    def test_custom_params(self):
        m = LeeCarter.Parametric.Multiregion(
            degree=3, n_knots=15, lam=0.5, nb_iter=200, verbose=True
        )
        self.assertEqual(m.degree, 3)
        self.assertEqual(m.n_knots, 15)
        self.assertEqual(m.lam, 0.5)
        self.assertEqual(m.nb_iter, 200)
        self.assertTrue(m.verbose)


# =============================================================================
# 9. Classic.__init__
# =============================================================================

class TestClassicInit(unittest.TestCase):

    def test_default_params(self):
        m = LeeCarter.Classic()
        self.assertEqual(m.nb_iter, 500)
        self.assertEqual(m.eta, 1)

    def test_custom_params(self):
        m = LeeCarter.Classic(nb_iter=100, eta=0.5)
        self.assertEqual(m.nb_iter, 100)
        self.assertEqual(m.eta, 0.5)


# =============================================================================
# 10. Classic.fit — smoke test (peu d'itérations)
# =============================================================================

class TestClassicFit(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(9)
        nb_ages, nb_years, nb_reg = 8, 5, 2
        self.Extg, self.Dxtg, self.xv, self.tv = _make_fake_data(nb_ages, nb_years, nb_reg)
        self.ax    = rng.normal(-4, 0.5, nb_ages)
        self.bx    = np.ones(nb_ages) / nb_ages
        self.kappa = rng.normal(0, 1, nb_years)

    def test_returns_dict(self):
        model = LeeCarter.Classic(nb_iter=5)
        result = model.fit(
            self.ax, self.bx, self.kappa,
            self.Extg, self.Dxtg, self.xv, self.tv
        )
        self.assertIsInstance(result, dict)

    def test_result_keys(self):
        model = LeeCarter.Classic(nb_iter=5)
        result = model.fit(
            self.ax, self.bx, self.kappa,
            self.Extg, self.Dxtg, self.xv, self.tv
        )
        self.assertIn("parameters",    result)
        self.assertIn("fitted_values", result)
        self.assertIn("fit_statistics", result)

    def test_fit_statistics_is_dataframe(self):
        model = LeeCarter.Classic(nb_iter=5)
        result = model.fit(
            self.ax, self.bx, self.kappa,
            self.Extg, self.Dxtg, self.xv, self.tv
        )
        self.assertIsInstance(result["fit_statistics"], pd.DataFrame)

    def test_log_mu_shape(self):
        nb_ages, nb_years = len(self.xv), len(self.tv)
        model = LeeCarter.Classic(nb_iter=5)
        result = model.fit(
            self.ax, self.bx, self.kappa,
            self.Extg, self.Dxtg, self.xv, self.tv
        )
        log_mu = result["fitted_values"]["log_mu"]
        self.assertEqual(log_mu.shape, (nb_ages, nb_years))

    def test_mu_positive(self):
        model = LeeCarter.Classic(nb_iter=5)
        result = model.fit(
            self.ax, self.bx, self.kappa,
            self.Extg, self.Dxtg, self.xv, self.tv
        )
        mu = result["fitted_values"]["mu"]
        self.assertTrue(np.all(mu > 0))


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
