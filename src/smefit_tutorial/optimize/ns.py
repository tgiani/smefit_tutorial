# -*- coding: utf-8 -*-

"""
Fitting the Wilson coefficients with NS
"""
import os
import time

from mpi4py import MPI
from pymultinest.solve import solve
from rich.style import Style
from rich.table import Table

from .. import log
from ..coefficients import CoefficientManager
from ..loader import load_datasets
from . import Optimizer

_logger = log.logging.getLogger(__name__)


class NSOptimizer(Optimizer):
    """
    Optimizer specification for |NS|

    Parameters
    ----------
        loaded_datasets : `smefit_tutorial.loader.DataTuple`,
            dataset tuple
        coefficients : `smefit_tutorial.coefficients.CoefficientManager`
            instance of `CoefficientManager` with all the relevant coefficients to fit
        use_quad : bool
            If True use also |HO| corrections
        live_points : int
            number of |NS| live points
        efficiency : float
            sampling efficiency
        const_efficiency: bool
            if True use the constant efficiency mode
        tolerance: float
            evidence tolerance
    """

    print_rate = 5000

    def __init__(
        self,
        loaded_datasets,
        coefficients,
        result_path,
        use_quad,
        result_ID,
        single_parameter_fits,
        pairwise_fits,
        use_multiplicative_prescription,
        live_points=500,
        efficiency=0.01,
        const_efficiency=False,
        tolerance=0.5,
    ):
        super().__init__(
            f"{result_path}/{result_ID}",
            loaded_datasets,
            coefficients,
            use_quad,
            single_parameter_fits,
            use_multiplicative_prescription,
        )
        self.live_points = live_points
        self.efficiency = efficiency
        self.const_efficiency = const_efficiency
        self.tolerance = tolerance
        self.npar = self.free_parameters.shape[0]
        self.result_ID = result_ID
        self.pairwise_fits = pairwise_fits

    @classmethod
    def from_dict(cls, config):
        """
        Create object from theory dictionary.

        Parameters
        ----------
            config : dict
                configuration dictionary

        Returns
        -------
            cls : Optimizer
                created object
        """

        loaded_datasets = load_datasets(
            config["data_path"],
            config["datasets"],
            config["coefficients"],
            config["order"],
            config["use_quad"],
            config["use_theory_covmat"],
            config["use_t0"],
            config.get("use_multiplicative_prescription", False),
            config.get("theory_path", None),
            config.get("rot_to_fit_basis", None),
            config.get("uv_coupligs", False),
        )

        coefficients = CoefficientManager.from_dict(config["coefficients"])

        single_parameter_fits = config.get("single_parameter_fits", False)
        pairwise_fits = config.get("pairwise_fits", False)
        nlive = config.get("nlive", 500)

        if "nlive" not in config:
            _logger.warning(
                f"Number of live points (nlive) not set in the input card. Using default: {nlive}"
            )

        efr = config.get("efr", 0.01)
        if "efr" not in config:
            _logger.warning(
                f"Sampling efficiency (efr) not set in the input card. Using default: {efr}"
            )

        ceff = config.get("ceff", False)
        if "ceff" not in config:
            _logger.warning(
                f"Constant efficiency mode (ceff) not set in the input card. Using default: {ceff}"
            )

        toll = config.get("toll", 0.5)
        if "toll" not in config:
            _logger.warning(
                f"Evidence tolerance (toll) not set in the input card. Using default: {toll}"
            )

        use_multiplicative_prescription = config.get(
            "use_multiplicative_prescription", False
        )
        return cls(
            loaded_datasets,
            coefficients,
            config["result_path"],
            config["use_quad"],
            config["result_ID"],
            single_parameter_fits,
            pairwise_fits,
            use_multiplicative_prescription,
            live_points=nlive,
            efficiency=efr,
            const_efficiency=ceff,
            tolerance=toll,
        )

    def chi2_func_ns(self, params):
        """
        Wrap the chi2 in a function for the optimizer. Pass noise and
        data info as args. Log the chi2 value and values of the coefficients.

        Parameters
        ----------
            params : np.ndarray
                noise and data info
        Returns
        -------
            current_chi2 : np.ndarray
                chi2 function

        """
        self.coefficients.set_free_parameters(params)
        self.coefficients.set_constraints()

        return self.chi2_func()

    def gaussian_loglikelihood(self, hypercube):
        """
        Multi gaussian log likelihood function

        Parameters
        ----------
            hypercube :  np.ndarray
                hypercube prior

        Returns
        -------
            -0.5 * chi2 : np.ndarray
                multi gaussian log likelihood
        """

        return -0.5 * self.chi2_func_ns(hypercube)

    def flat_prior(self, hypercube):
        """
        Update the prior function

        Parameters
        ----------
            hypercube : np.ndarray
                hypercube prior

        Returns
        -------
            flat_prior : np.ndarray
                updated hypercube prior
        """
        min_val = self.free_parameters.minimum
        max_val = self.free_parameters.maximum
        return hypercube * (max_val - min_val) + min_val

    def clean(self):
        """Remove raw |NS| output if you want to keep raw output, don't call this method"""

        for f in os.listdir(self.results_path):
            if f.startswith(f"{self.live_points}_"):
                os.remove(self.results_path / f)

    def run_sampling(self):
        """Run the minimization with Nested Sampling"""

        # Prefix for results
        prefix = self.results_path / f"{self.live_points}_"

        # Additional check
        # Multinest will crash if the length of the results
        # path+post_equal_weights.txt is longer than 100.
        # you can solve this making you path or fit id shorter.
        # Otherwise you can hack /pymultinest/solve.py
        if len(f"{prefix}post_equal_weights.txt") >= 100:
            raise UserWarning(
                f"Py multinest support a buffer or maximum 100 characters: \
                    {prefix}post_equal_weights.txt is too long, \
                         please chose a shorter path or Fit ID"
            )

        t1 = time.time()

        result = solve(
            LogLikelihood=self.gaussian_loglikelihood,
            Prior=self.flat_prior,
            n_dims=self.npar,
            n_params=self.npar,
            outputfiles_basename=str(prefix),
            n_live_points=self.live_points,
            sampling_efficiency=self.efficiency,
            verbose=False,
            importance_nested_sampling=True,
            const_efficiency_mode=self.const_efficiency,
            evidence_tolerance=self.tolerance,
        )

        t2 = time.time()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            log.console.log(f"Time : {((t2 - t1) / 60.0):.3f} minutes")
            log.console.log(f"Number of samples: {result['samples'].shape[0]}")

            table = Table(
                style=Style(color="white"), title_style="bold cyan", title=None
            )
            table.add_column("Parameter", style="bold red", no_wrap=True)
            table.add_column("Best value")
            table.add_column("Error")
            for par, col in zip(self.free_parameters.index, result["samples"].T):
                table.add_row(f"{par}", f"{col.mean():.3f}", f"{col.std():.3f}")
            log.console.print(table)

            self.save(result)
            self.clean()

    def save(self, result):
        """
        Save |NS| replicas to json inside a dictionary:
        {coff: [replicas values]}

        Parameters
        ----------
            result : dict
                result dictionary

        """
        values = {}
        for c in self.coefficients.name:
            values[c] = []

        for sample in result["samples"]:

            self.coefficients.set_free_parameters(sample)
            self.coefficients.set_constraints()

            for c in self.coefficients.name:
                values[c].append(self.coefficients[c].value)

        if self.pairwise_fits:
            posterior_file = (
                self.results_path
                / f"posterior_{self.coefficients.name[0]}_{self.coefficients.name[1]}.json"
            )
        else:
            posterior_file = self.results_path / "posterior.json"

        self.dump_posterior(posterior_file, values)
