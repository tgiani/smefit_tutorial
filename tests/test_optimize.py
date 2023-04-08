# -*- coding: utf-8 -*-
import copy
import pathlib

import numpy as np
from scipy import optimize as sciopt

import smefit_tutorial.optimize as opt
import tests.test_loader as ld
from smefit_tutorial.compute_theory import flatten, make_predictions

commondata_path = pathlib.Path(__file__).parents[0] / "fake_data"


# fix the wilson coefficients to some random value and compute the corresponding expected chi2

coeffs_dict = {
    "Op1": {
        "min": -2,
        "max": 1,
    },
    "Op2": {
        "min": -2,
        "max": 1,
    },
    "Op3": {
        "min": -2,
        "max": 1,
    },
    "Op4": {
        "constrain": True,
        "value": 0.0,
        "min": -2,
        "max": 1,
    },
}

random_point = np.random.rand(3)
prior = random_point * (
    np.array([1.0, 1.0, 1.0]) - np.array([-2.0, -2.0, -2.0])
) + np.array([-2.0, -2.0, -2.0])

# generate random values for the wilson coefficients
wilson_coeff = np.random.rand(3)

### first consider the case of dataset 1 and dataset 2, which do not have common systemtics

# Theory predictions for dataset1
th_pred_1 = (
    np.array(ld.theory_test_1["best_sm"])
    + np.array(ld.theory_test_1["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(ld.theory_test_1["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(ld.theory_test_1["LO"]["Op3"]) * wilson_coeff[2]
    + np.array(ld.theory_test_1["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(ld.theory_test_1["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(ld.theory_test_1["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
    + np.array(ld.theory_test_1["LO"]["Op2*Op3"]) * wilson_coeff[1] * wilson_coeff[2]
)

# Theory predictions for dataset1 when using mult prescription
th_pred_1_mult = np.array(ld.theory_test_1["best_sm"]) * (
    1.0
    + np.array(ld.theory_test_1["LO"]["Op1"])
    * wilson_coeff[0]
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op2"])
    * wilson_coeff[1]
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op3"])
    * wilson_coeff[2]
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op1*Op1"])
    * wilson_coeff[0] ** 2
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op2*Op2"])
    * wilson_coeff[1] ** 2
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op2*Op1"])
    * wilson_coeff[1]
    * wilson_coeff[0]
    / ld.theory_test_1["LO"]["SM"]
    + np.array(ld.theory_test_1["LO"]["Op2*Op3"])
    * wilson_coeff[1]
    * wilson_coeff[2]
    / ld.theory_test_1["LO"]["SM"]
)

# exp data for dataset1
exp_data_1 = np.array(ld.exp_test_1["data"])
exp_stat_1 = np.array([0.2, 0.3])
exp_sys_1 = np.array([[0.01, 0.01], [0.02, 0.02]])
exp_cov_1 = np.diag(exp_stat_1**2) + exp_sys_1 @ exp_sys_1.T
tot_cov_1 = exp_cov_1 + np.array(ld.theory_test_1["theory_cov"])

# chi2 for dataset 1
chi2_1 = (
    (exp_data_1 - th_pred_1) @ np.linalg.inv(tot_cov_1) @ (exp_data_1 - th_pred_1).T
)

# chi2 for dataset 1 when using mult prescription
chi2_1_mult = (
    (exp_data_1 - th_pred_1_mult)
    @ np.linalg.inv(tot_cov_1)
    @ (exp_data_1 - th_pred_1_mult).T
)

# theory predictions for dataset2
th_pred_2 = (
    np.array(ld.theory_test_2["best_sm"])
    + np.array(ld.theory_test_2["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(ld.theory_test_2["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(ld.theory_test_2["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(ld.theory_test_2["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(ld.theory_test_2["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
)

# theory predictions for dataset2 when using mult prescription
th_pred_2_mult = np.array(ld.theory_test_2["best_sm"]) * (
    1.0
    + np.array(ld.theory_test_2["LO"]["Op1"])
    * wilson_coeff[0]
    / ld.theory_test_2["LO"]["SM"]
    + np.array(ld.theory_test_2["LO"]["Op2"])
    * wilson_coeff[1]
    / ld.theory_test_2["LO"]["SM"]
    + np.array(ld.theory_test_2["LO"]["Op1*Op1"])
    * wilson_coeff[0] ** 2
    / ld.theory_test_2["LO"]["SM"]
    + np.array(ld.theory_test_2["LO"]["Op2*Op2"])
    * wilson_coeff[1] ** 2
    / ld.theory_test_2["LO"]["SM"]
    + np.array(ld.theory_test_2["LO"]["Op2*Op1"])
    * wilson_coeff[1]
    * wilson_coeff[0]
    / ld.theory_test_2["LO"]["SM"]
)


exp_data_2 = np.array(ld.exp_test_2["data"])
exp_stat_2 = np.array([0.2, 0.3, 0.2, 0.3])
exp_sys_2 = np.array([[0.03, 0.03], [0.04, 0.04], [0.05, 0.05], [0.06, 0.06]])
exp_cov2 = np.diag(exp_stat_2**2) + exp_sys_2 @ exp_sys_2.T
tot_cov_2 = exp_cov2 + np.array(ld.theory_test_2["theory_cov"])

# chi2 for dataset2
chi2_2 = (
    (exp_data_2 - th_pred_2) @ np.linalg.inv(tot_cov_2) @ (exp_data_2 - th_pred_2).T
)

# chi2 for dataset2 when using mult prescription
chi2_2_mult = (
    (exp_data_2 - th_pred_2_mult)
    @ np.linalg.inv(tot_cov_2)
    @ (exp_data_2 - th_pred_2_mult).T
)

# total expected chi2. Since there are no correlations between the two datasets I can sum the
# two independent contributions
chi2_tot = chi2_1 + chi2_2

# total expected chi2 when using mult prescription
chi2_tot_mult = chi2_1_mult + chi2_2_mult


datasets_no_corr = ["data_test1", "data_test2"]
config_no_corr = {}
config_no_corr["data_path"] = commondata_path
config_no_corr["coefficients"] = coeffs_dict
config_no_corr["result_path"] = commondata_path
config_no_corr["result_ID"] = "test"
config_no_corr["datasets"] = datasets_no_corr
config_no_corr["order"] = "LO"
config_no_corr["use_quad"] = True
config_no_corr["use_theory_covmat"] = True
config_no_corr["use_t0"] = False
config_no_corr["theory_path"] = commondata_path
config_no_corr["rot_to_fit_basis"] = None
config_no_corr["replica"] = 0


### now consider the case of dataset 3 and dataset 4 having a common systematic named SPECIAL
# The theory predictions for the dataset 3 are the same as for dataset 1

th_pred_3 = th_pred_1

# while dataset 4 we have

theory_test_4 = {
    "best_sm": [1.1, 2.3, 3.0],
    "theory_cov": [
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.3],
    ],
    "LO": {
        "Op1": [1.1, 1.2, 1.3],
        "Op4": [2.5, 1.6, 1.7],
        "Op2": [1.3, 4.4, 4.5],
        "Op1*Op1": [1.7, 1.8, 1.9],
        "Op2*Op2": [1.9, 1.11, 2.1],
        "Op2*Op1": [3.12, 2.13, 3.1],
        "Op2*Op4": [9.14, 5.15, 5.16],
    },
    "NLO": {
        "Op1": [2.1, 2.2, 2.3],
        "Op4": [3.5, 3.6, 2.7],
        "Op2": [4.3, 4.4, 5.5],
        "Op1*Op1": [2.7, 1.8, 2.9],
        "Op2*Op2": [9.9, 5.11, 6.16],
        "Op2*Op1": [1.12, 2.13, 6.18],
        "Op2*Op4": [7.14, 4.15, 4.17],
    },
}

th_pred_4 = (
    np.array(theory_test_4["best_sm"])
    + np.array(theory_test_4["LO"]["Op1"]) * wilson_coeff[0]
    + np.array(theory_test_4["LO"]["Op2"]) * wilson_coeff[1]
    + np.array(theory_test_4["LO"]["Op1*Op1"]) * wilson_coeff[0] ** 2
    + np.array(theory_test_4["LO"]["Op2*Op2"]) * wilson_coeff[1] ** 2
    + np.array(theory_test_4["LO"]["Op2*Op1"]) * wilson_coeff[1] * wilson_coeff[0]
)


# This time the total experimental covmat is not block diagonal, and it is given by

tot_cov_corr = [
    [1.53, 0.53, 0.02, 0.02, 0.02],
    [0.53, 1.53, 0.02, 0.02, 0.02],
    [0.02, 0.02, 1.38, 0.34, 0.34],
    [0.02, 0.02, 0.34, 1.38, 0.34],
    [0.02, 0.02, 0.34, 0.34, 1.38],
]


# The experimental central values are

exp_data_3 = np.array([1.0, 2.0])
exp_data_4 = np.array([1.0, 2.0, 3.0])


diff = np.concatenate([[exp_data_3 - th_pred_3], [exp_data_4 - th_pred_4]], axis=1)
chi2_corr = diff @ np.linalg.inv(tot_cov_corr) @ diff.T

# if we consider the case in which we use t0, assuming that the sys SPECIAL is MULT
# the t0 covarinace matrix is given by

tot_cov_corr_t0 = [
    [1.53, 0.53, 0.022, 0.023, 0.02],
    [0.53, 1.53, 0.022, 0.023, 0.02],
    [0.022, 0.022, 1.3884, 0.3506, 0.344],
    [0.023, 0.023, 0.3506, 1.3929, 0.346],
    [0.02, 0.02, 0.344, 0.346, 1.38],
]

# and the chi2 is

chi2_corr_t0 = diff @ np.linalg.inv(tot_cov_corr_t0) @ diff.T


datasets_corr = ["data_test3", "data_test4"]
config_corr = {}
config_corr["data_path"] = commondata_path
config_corr["coefficients"] = coeffs_dict
config_corr["result_path"] = commondata_path
config_corr["result_ID"] = "test"
config_corr["datasets"] = datasets_corr
config_corr["order"] = "LO"
config_corr["use_quad"] = True
config_corr["use_theory_covmat"] = True
config_corr["theory_path"] = commondata_path
config_corr["rot_to_fit_basis"] = None
config_corr["use_multiplicative_prescription"] = False


class TestOptimize_NS:

    config_no_corr["use_multiplicative_prescription"] = True
    test_opt_mult = opt.ns.NSOptimizer.from_dict(config_no_corr)

    config_no_corr["use_multiplicative_prescription"] = False
    test_opt = opt.ns.NSOptimizer.from_dict(config_no_corr)

    config_corr["use_t0"] = False
    test_opt_corr = opt.ns.NSOptimizer.from_dict(config_corr)

    config_corr["use_t0"] = True
    test_opt_corr_t0 = opt.ns.NSOptimizer.from_dict(config_corr)

    def test_init(self):
        assert self.test_opt.results_path == commondata_path / "test"
        np.testing.assert_equal(
            self.test_opt.loaded_datasets.ExpNames, datasets_no_corr
        )
        np.testing.assert_equal(
            self.test_opt.coefficients.name, ["Op1", "Op2", "Op3", "Op4"]
        )

    def test_free_parameters(self):
        np.testing.assert_equal(
            list(self.test_opt.free_parameters.index), ["Op1", "Op2", "Op3"]
        )

    def test_chi2_func_ns(self):
        # set free parameters to random values generated above
        params = wilson_coeff

        # test experimental chi2 in case of no cross correlations between dataset
        np.testing.assert_allclose(
            self.test_opt.chi2_func_ns(params), chi2_tot, rtol=1e-10
        )
        # test experimental chi2 in case of cross correlations between dataset
        np.testing.assert_allclose(
            self.test_opt_corr.chi2_func_ns(params), chi2_corr, rtol=1e-10
        )
        # test t0 chi2 in case of cross correlations between dataset
        np.testing.assert_allclose(
            self.test_opt_corr_t0.chi2_func_ns(params), chi2_corr_t0, rtol=1e-10
        )

        # test experimental chi2 when using multiplicative prescription for theory predictions
        np.testing.assert_allclose(
            self.test_opt_mult.chi2_func_ns(params), chi2_tot_mult, rtol=1e-10
        )

    def test_gaussian_loglikelihood(self):
        params = wilson_coeff
        np.testing.assert_allclose(
            self.test_opt.gaussian_loglikelihood(params), -0.5 * chi2_tot, rtol=1e-10
        )
        np.testing.assert_allclose(
            self.test_opt_corr.gaussian_loglikelihood(params),
            -0.5 * chi2_corr,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            self.test_opt_corr_t0.gaussian_loglikelihood(params),
            -0.5 * chi2_corr_t0,
            rtol=1e-10,
        )

    def test_flat_prior(self):
        np.testing.assert_equal(self.test_opt.flat_prior(random_point), prior)


class TestOptimize_MC:

    test_opt = opt.mc.MCOptimizer.from_dict(config_no_corr)

    def test_init(self):
        assert self.test_opt.results_path == commondata_path / "test"
        np.testing.assert_equal(
            self.test_opt.loaded_datasets.ExpNames, datasets_no_corr
        )
        np.testing.assert_equal(
            self.test_opt.coefficients.name, ["Op1", "Op2", "Op3", "Op4"]
        )

    def test_free_parameters(self):
        np.testing.assert_equal(
            list(self.test_opt.free_parameters.index), ["Op1", "Op2", "Op3"]
        )

    def test_jacobian(self):
        # test chi2_mc computing the jacobian

        def jacobian(params):

            self.test_opt.coefficients.set_free_parameters(params)
            self.test_opt.coefficients.set_constraints()

            data = self.test_opt.loaded_datasets
            coeff_val = self.test_opt.coefficients.value

            diff = 2 * (
                data.Replica
                - make_predictions(data, coeff_val, self.test_opt.use_quad, False)
            )

            # temp coefficiens
            temp_coeff = copy.deepcopy(self.test_opt.coefficients)
            free_coeffs = temp_coeff.free_parameters.index

            # propagate contrain to linear and quad corrections
            new_linear_corrections = np.zeros(
                (data.Replica.shape[0], free_coeffs.shape[0])
            )
            new_quad_corrections = np.zeros(
                (data.Replica.shape[0], free_coeffs.shape[0])
            )

            for idx in range(free_coeffs.shape[0]):
                params = np.zeros_like(free_coeffs)
                params[idx] = 1.0
                temp_coeff.set_free_parameters(params)
                temp_coeff.set_constraints()

                # update corrections
                new_linear_corrections[:, idx] = (
                    data.LinearCorrections @ temp_coeff.value
                )
                if self.test_opt.use_quad:
                    # derivative of the outher product
                    quad_coeff_mat = np.outer(temp_coeff.value, coeff_val)
                    quad_coeff_mat = np.maximum(quad_coeff_mat, quad_coeff_mat.T)
                    np.fill_diagonal(quad_coeff_mat, 2 * np.diag(quad_coeff_mat))
                    new_quad_corrections[:, idx] = np.einsum(
                        "ij,j->i", data.QuadraticCorrections, flatten(quad_coeff_mat)
                    )

            jac = -new_linear_corrections - new_quad_corrections
            return np.einsum("i,ij,jk->k", diff, data.InvCovMat, jac)

        test = sciopt.check_grad(
            self.test_opt.chi2_func_mc, jacobian, np.random.rand(3)
        )
        np.testing.assert_allclose(test, 0, atol=5e-5)
