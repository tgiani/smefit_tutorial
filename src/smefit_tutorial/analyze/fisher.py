# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.progress import track

from .latex_tools import latex_packages
from .pca import impose_constrain


class FisherCalculator:

    """Computes and writes the Fisher information table, and plots heat map.

    Linear Fisher information depends only on the theoretical corrections,
    while quadratic information requires fit results.
    Parameter constraints are also taken into account. Only fitted
    degrees of freedom are shown in the tables.

    Parameters
    ----------
    coefficients: smefit_tutorial.coefficients.CoefficienManager
        coefficient manager
    datasets: smefit_tutorial.loader.DataTuple
        DataTuple object with all the data information

    """

    def __init__(self, coefficients, datasets, compute_quad):

        self.coefficients = coefficients
        self.free_parameters = self.coefficients.free_parameters.index
        self.datasets = datasets

        # update eft corrections with the constraints
        if compute_quad:
            (
                self.new_LinearCorrections,
                self.new_QuadraticCorrections,
            ) = impose_constrain(self.datasets, self.coefficients, update_quad=True)
        else:
            self.new_LinearCorrections = impose_constrain(
                self.datasets, self.coefficients
            )

        self.lin_fisher = None
        self.quad_fisher = None
        self.summary_table = None
        self.summary_HOtable = None

    def compute_linear(self):
        """Compute linear Fisher information."""
        fisher_tab = []
        cnt = 0
        for ndat in self.datasets.NdataExp:
            fisher_row = np.zeros(self.free_parameters.size)
            idxs = slice(cnt, cnt + ndat + 1)
            sigma = self.new_LinearCorrections[:, idxs]
            fisher_row = np.diag(sigma @ self.datasets.InvCovMat[idxs, idxs] @ sigma.T)
            fisher_tab.append(fisher_row)
            cnt += ndat
        self.lin_fisher = pd.DataFrame(
            fisher_tab, index=self.datasets.ExpNames, columns=self.free_parameters
        )

    def compute_quadratic(self, posterior_df, smeft_predictions):
        """Compute quadratic Fisher information."""
        quad_fisher = []

        # compute some average values over the replicas
        # delta exp - th (n_dat)
        delta_th = self.datasets.Commondata - np.mean(smeft_predictions, axis=0)
        # c, c**2 mean (n_free_op)
        posterior_df = posterior_df[self.free_parameters]
        c_mean = np.mean(posterior_df.values, axis=0)
        c2_mean = np.mean(posterior_df.values**2, axis=0)

        # squared quad corr
        diag_corr = np.diagonal(self.new_QuadraticCorrections, axis1=0, axis2=1)
        off_diag_corr = self.new_QuadraticCorrections
        diag_index = np.diag_indices(self.free_parameters.size)
        off_diag_corr[diag_index[0], diag_index[1], :] = 0

        # additional tensors
        tmp = np.einsum("ri,ijk->rjk", posterior_df, off_diag_corr, optimize="optimal")
        A_all = np.mean(tmp, axis=0)  # (n_free_op, n_dat)
        B_all = (
            np.einsum("rj,rjk->jk", posterior_df, tmp, optimize="optimal")
            / posterior_df.shape[0]
        )  # (n_free_op, n_dat)
        D_all = (
            np.einsum("rjk,rjl->jkl", tmp, tmp, optimize="optimal")
            / posterior_df.shape[0]
        )  # (n_free_op, n_dat, n_dat)

        cnt = 0
        for ndat in track(
            self.datasets.NdataExp,
            description="[green]Computing quadratic Fisher information per dataset...",
        ):
            # slice the big matrices
            idxs = slice(cnt, cnt + ndat + 1)
            quad_corr = diag_corr[idxs, :].T
            lin_corr = self.new_LinearCorrections[:, idxs]
            inv_corr = self.datasets.InvCovMat[idxs, idxs]
            delta = delta_th[idxs]
            A = A_all[:, idxs]
            B = B_all[:, idxs]
            D = D_all[:, idxs, idxs]

            # (n_free_op)
            fisher_row = (
                -quad_corr @ inv_corr @ delta.T
                - delta @ inv_corr @ quad_corr.T
                + lin_corr @ inv_corr @ A.T
                + A @ inv_corr @ lin_corr.T
                + 2
                * c_mean
                @ (
                    lin_corr @ inv_corr @ quad_corr.T
                    + quad_corr @ inv_corr @ lin_corr.T
                )
                + 2 * (B @ inv_corr @ quad_corr.T + quad_corr @ inv_corr @ B.T)
                + 4 * c2_mean @ quad_corr @ inv_corr @ quad_corr.T
                + np.einsum("ikl,kl -> i", D, inv_corr, optimize="optimal")
            )
            quad_fisher.append(np.diag(fisher_row))
            cnt += ndat

        self.quad_fisher = pd.DataFrame(
            quad_fisher + self.lin_fisher.values,
            index=self.datasets.ExpNames,
            columns=self.free_parameters,
        )

    @staticmethod
    def normalize(table, norm, log):
        """
        Normalize a Pandas DataFrame

        Parameters
        ----------
        table: pandas.DataFrame
            table to normalize
        norm: "data", "coeff"
            if "data" it normalize by columns, if "coeff" by rows
        log: bool
            presents the log of the Fisher if True

        Returns
        -------
        pandas.DataFrame
                normalized table

        """
        if table is None or table.empty:
            return None
        if norm == "data":
            axis_sum, axis_div = 1, 0
        elif norm == "coeff":
            axis_sum, axis_div = 0, 1

        table = table.div(table.sum(axis=axis_sum), axis=axis_div) * 100
        if log:
            table = np.log(table[table > 0.0])
        return table.replace(np.nan, 0.0)

    def groupby_data(self, table, data_groups, norm, log):
        """Merge fisher per data group."""
        summary_table = pd.merge(
            data_groups.reset_index(), table, left_on="level_1", right_index=True
        )
        summary_table = summary_table.groupby("level_0").sum(numeric_only=True)
        summary_table.index.name = "data_group"
        return self.normalize(summary_table, norm, log)

    def write_grouped(self, coeff_config, data_groups, summary_only):
        """Write Fisher information tables in latex, both for grouped data and for summary.

        Parameters
        ----------
        coeff_config: dict
            coefficient dictionary per group with latex names
        data_groups: dict
            dictionary with datasets  per group and relative links
        summary_only: bool
            if True only the summary Fisher table fro grouped data is written

        Returns
        -------
        list(str)
            list of the latex commands

        """
        L = latex_packages()
        L.extend(
            [
                r"\begin{document}",
                r"\begin{landscape}",
            ]
        )

        # fisher tables per data_group
        if not summary_only:
            for data_group, data_dict in data_groups.groupby(level=0):
                temp_table = self.lin_fisher.loc[data_dict.index.get_level_values(1)]
                temp_HOtable = None

                if self.quad_fisher is not None:
                    temp_HOtable = self.quad_fisher.loc[
                        data_dict.index.get_level_values(1)
                    ]
                L.extend(
                    self._write(
                        temp_table,
                        temp_HOtable,
                        coeff_config,
                        data_dict.droplevel(0),
                        data_group,
                    )
                )
        L.extend(
            self._write(
                self.summary_table,
                self.summary_HOtable,
                coeff_config,
            )
        )
        L.append(r"\end{landscape}")
        return L

    def _write(
        self, lin_fisher, quad_fisher, coeff_config, data_dict=None, data_group=None
    ):
        """Write Fisher information table in latex.

        Parameters
        ----------
        lin_fisher: pandas.DataFrame
            linear Fisher information table
        quad_fisher: pandas.DataFrame, None
            quadratic Fisher information table, None if linear only
        coeff_config: dict
            coefficient dictionary per group with latex names
        data_dict: dict, optional
            dictionary with datasets and relative links
        data_group: str, optional
            data group name

        Returns
        -------
        list(str)
            list of the latex commands

        """

        def color(value, thr_val=10):
            if value > thr_val:
                return ("blue", value)
            return ("black", value)

        L = [
            r"\begin{table}[H]",
            r"\scriptsize",
            r"\centering",
            r"\begin{tabular}{|c|c|" + "c|" * lin_fisher.shape[0] + "}",
            r"\hline",
            f"\\multicolumn{{2}}{{|c|}}{{}} \
                & \\multicolumn{{{lin_fisher.shape[0]}}}{{c|}}{{Processes}} \\\\ \\hline",
        ]
        temp = " Class & Coefficient "
        if data_dict is None:
            for dataset in lin_fisher.index:
                temp += f"& {{\\rm {dataset} }}"
        else:
            for dataset, link in data_dict.items():
                temp += f"& \\href{{{link}}}{{${{\rm {dataset}}}$}}".replace("_", r"\_")
        temp += r"\\ \hline"
        L.append(temp)

        # loop on coeffs
        for coeff_group, coeff_dict in coeff_config.groupby(level=0):
            coeff_dict = coeff_dict.droplevel(0)
            L.append(f"\\multirow{{{coeff_dict.shape[0]}}}{{*}}{{{coeff_group}}}")
            for coeff, latex_name in coeff_dict.items():
                idx = np.where(coeff == self.free_parameters)[0][0]
                temp = f" & {latex_name}"

                # loop on columns
                for idj, fisher_col in enumerate(lin_fisher.values):
                    temp += r" & \textcolor{%s}{%.2f}" % color(fisher_col[idx])
                    if quad_fisher is not None:
                        temp += r"(\textcolor{%s}{%0.2f})" % color(
                            quad_fisher.iloc[idj, idx]
                        )
                temp += (
                    r"\\ \hline"
                    if coeff == [*coeff_dict.keys()][-1]
                    else f"\\\\ \\cline{{2-{(2 + lin_fisher.shape[0])}}}"
                )
                L.append(temp)

        caption = (
            "Fisher information"
            if data_group is None
            else f"Fisher information in {data_group} datasets"
        )
        L.extend(
            [
                r"\end{tabular}",
                f"\\caption{{{caption}}}",
                r"\end{table}",
            ]
        )
        return L

    def plot(
        self,
        latex_names,
        fig_name,
        title=None,
        summary_only=True,
        figsize=(11, 15),
    ):
        """Plot the heat map of Fisher table.

        Parameters
        ----------
        latex_names : list
            list of coefficients latex names
        fig_name: str
            figure path
        summary_only:
            if True plot the fisher grouped per datsets,
            else the fine grained dataset per dataset
        figsize : tuple
            figure size
        title: str, None
            plot title
        """
        if summary_only:
            fisher_df = self.summary_table
            quad_fisher_df = self.summary_HOtable
        else:
            fisher_df = self.lin_fisher
            quad_fisher_df = self.quad_fisher

        fig = plt.figure(figsize=figsize)
        if quad_fisher_df is not None:
            ax = fig.add_subplot(121)
        else:
            ax = plt.gca()

        # clolor map
        cmap_full = cm.get_cmap("Blues")
        cmap = colors.LinearSegmentedColormap.from_list(
            f"trunc({{{cmap_full.name}}},{{0}},{{0.8}})",
            cmap_full(np.linspace(0, 0.8, 100)),
        )
        norm = colors.BoundaryNorm(np.arange(110, step=10), cmap.N)

        # thicks
        yticks = np.arange(fisher_df.shape[1])
        xticks = np.arange(fisher_df.shape[0])
        x_labels = [f"\\rm{{{name}}}".replace("_", "\\_") for name in fisher_df.index]

        def set_ticks(ax):
            ax.set_yticks(yticks, labels=latex_names, fontsize=15)
            ax.set_xticks(
                xticks,
                labels=x_labels,
                rotation=90,
                fontsize=15,
            )
            ax.tick_params(which="major", top=False, bottom=False, left=False)
            # minor grid
            ax.set_xticks(xticks - 0.5, minor=True)
            ax.set_yticks(yticks - 0.5, minor=True)
            ax.tick_params(which="minor", bottom=False)
            ax.grid(visible=True, which="minor", alpha=0.2)

        def plot_values(ax, df):
            for i, row in enumerate(df.values.T):
                for j, elem in enumerate(row):
                    if elem > 10:
                        ax.text(
                            j,
                            i,
                            f"{elem:.1f}",
                            va="center",
                            ha="center",
                            fontsize=8,
                        )

        cax = ax.matshow(fisher_df.values.T, cmap=cmap, norm=norm)
        plot_values(ax, fisher_df)
        set_ticks(ax)
        ax.set_title(r"\rm Linear", fontsize=20, y=-0.08)
        cax1 = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.1)
        colour_bar = fig.colorbar(cax, cax=cax1)

        if quad_fisher_df is not None:
            ax = fig.add_subplot(122)
            cax = ax.matshow(quad_fisher_df.values.T, cmap=cmap, norm=norm)
            plot_values(ax, quad_fisher_df)
            set_ticks(ax)
            ax.set_title(r"\rm Quadratic", fontsize=20, y=-0.08)
            cax1 = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.1)
            colour_bar = fig.colorbar(cax, cax=cax1)

        colour_bar.set_label(
            r"${\rm Normalized\ Value}$",
            fontsize=25,
            labelpad=30,
            rotation=270,
        )
        if title is not None:
            plt.suptitle(f"\\rm Fisher\\ information:\\ {title}", fontsize=25)
        plt.tight_layout()
        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")
