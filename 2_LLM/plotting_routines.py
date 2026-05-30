from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from globals import plot_dir



label_dict = {
    'a': r'$a$ [-]',
    'aline' : r"$a'$ [-]",
    'alpha' : r"$\alpha$ [deg]",
    'phi' : r"$\phi$ [deg]",
    'Cx' : r'$C_n$ [-]',
    'Cy' : r'$C_t$ [-]',
    'T' : r"$A$ (azimuthal) [N]",
    'N' : r"$T$ [N]",
    'Q' : r"$Q$ [Nm]",
    'J' : r"$J$ [-]",
    'Ct' : r"$C_T$ [-]",
    'QC' : r"$C_Q$ [-]",
    'TC' : r"$C_T$ [-]",
    'Cl' : r"$C_l$ [-]",
    'Cd' : r"$C_d$ [-]",
    'V_i' : r"$V_i$ [m/s]",
    'F_azim' : r"$F_\mathrm{azim}$ [N/m]",
    'F_axial' : r"$F_\mathrm{axial}$ [N/m]",
    'gamma' : r"$\Gamma$ [m$^2$/s]",
    'CP'          : r"$C_P$ [-]",
    'n_elem'      : r"$N_\mathrm{elem}$ [-]",
    'conv_factor' : r"$u_c\,/\,V_\infty$ [-]",
    'n_azim'      : r"$N_\mathrm{azim}$ [-]",
    'periods'     : r"Wake rotations [-]",
}


def forces(plot_vs: Literal['J', 'n_elem', 'conv_factor', 'n_azim', 'periods'],
           plot_ag: Literal['T', 'Q', 'QC', 'TC', 'CP'],
           data_df:pd.DataFrame,
           ax:Axes=None,
           verif:bool=False,
           **kwargs) -> Axes:

    x = data_df[plot_vs]

    if verif:
        y = data_df['Thrust' if plot_ag=='T' else 'Torque'] * 1000
    else:
        y = data_df[plot_ag]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))

    ax.plot(x, y, **kwargs)

    ax.set_xlabel(label_dict[plot_vs])
    ax.set_ylabel(label_dict[plot_ag])
    ax.grid(True,alpha=0.3)

    return ax



def distribution(plot_vs: Literal['a', 'aline', 'alpha', 'phi', 'Cx', 'Cy', 'Ct', 'T', 'N',
                                   'Cl', 'Cd', 'V_i', 'gamma'],
                 data_df:pd.DataFrame,
                 ax:Axes=None,
                 **kwargs) -> Axes:

    x = data_df['r_R']
    y = data_df[plot_vs]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))    
    
    ax.plot(x, y, **kwargs)

    ax.set_xlabel(r"$r/R$ [-]")
    ax.set_ylabel(label_dict[plot_vs])
    ax.grid(True, alpha=0.3)

    return ax



# ── Sensitivity study helpers ──────────────────────────────────────────────

def _sensitivity_distribution(dist_dfs: dict,
                               plot_vs: str = 'Cx',
                               ax: Axes = None,
                               legend_title: str = None) -> Axes:
    """Overlay radial distributions for multiple parameter values.

    dist_dfs: {label: DataFrame} — each DataFrame is the output of export_dist().
              Dict keys are used directly as legend labels, so format them
              before passing (e.g. {f'$u_c/V_\\infty$ = {v:.1f}': df, ...}).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for label, df in dist_dfs.items():
        distribution(plot_vs, df, ax=ax, label=str(label))
    ax.legend(title=legend_title, fontsize=8)
    return ax


def plot_conv_speed_sensitivity(global_df: pd.DataFrame,
                                dist_dfs: dict,
                                save: bool = False) -> None:
    """Convection-speed sensitivity: CT and CP vs conv_factor + Cx radial dist.

    global_df columns: ['conv_factor', 'TC', 'CP']
    dist_dfs: {conv_factor_value: DataFrame from export_dist()}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Sensitivity to Convection Speed')

    df_s = global_df.sort_values('conv_factor')

    forces('conv_factor', 'TC', df_s, ax=axes[0], marker='o')
    forces('conv_factor', 'CP', df_s, ax=axes[1], marker='o')

    formatted = {f'$u_c/V_\\infty={k:.2g}$': v for k, v in dist_dfs.items()}
    _sensitivity_distribution(formatted, plot_vs='Cx', ax=axes[2],
                              legend_title=label_dict['conv_factor'])

    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / 'sensitivity_conv_speed.pdf', bbox_inches='tight')
    plt.show()


def plot_disc_sensitivity(global_df: pd.DataFrame,
                          save: bool = False) -> None:
    """Blade discretization sensitivity: CT and CP vs n_elem for uniform and cosine.

    global_df columns: ['n_elem', 'scheme', 'TC', 'CP']
    scheme values: 'uniform', 'cosine'
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Sensitivity to Blade Discretization')

    for scheme, grp in global_df.groupby('scheme'):
        grp_s = grp.sort_values('n_elem')
        forces('n_elem', 'TC', grp_s, ax=axes[0], marker='o', label=scheme)
        forces('n_elem', 'CP', grp_s, ax=axes[1], marker='o', label=scheme)

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / 'sensitivity_disc.pdf', bbox_inches='tight')
    plt.show()


def plot_azim_sensitivity(global_df: pd.DataFrame,
                          save: bool = False) -> None:
    """Azimuthal discretization sensitivity: CT and CP vs n_azim.

    global_df columns: ['n_azim', 'TC', 'CP']
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Sensitivity to Azimuthal Discretization')

    df_s = global_df.sort_values('n_azim')

    forces('n_azim', 'TC', df_s, ax=axes[0], marker='o')
    forces('n_azim', 'CP', df_s, ax=axes[1], marker='o')

    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / 'sensitivity_azim.pdf', bbox_inches='tight')
    plt.show()


def plot_wake_length_sensitivity(global_df: pd.DataFrame,
                                 dist_dfs: dict,
                                 save: bool = False) -> None:
    """Wake-length sensitivity: CT and CP vs periods + Cx radial dist.

    global_df columns: ['periods', 'TC', 'CP']
    dist_dfs: {periods_value: DataFrame from export_dist()}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Sensitivity to Wake Length')

    df_s = global_df.sort_values('periods')

    forces('periods', 'TC', df_s, ax=axes[0], marker='o')
    forces('periods', 'CP', df_s, ax=axes[1], marker='o')

    formatted = {f'{k} rot.': v for k, v in dist_dfs.items()}
    _sensitivity_distribution(formatted, plot_vs='Cx', ax=axes[2],
                              legend_title=label_dict['periods'])

    plt.tight_layout()
    if save:
        fig.savefig(plot_dir / 'sensitivity_wake_length.pdf', bbox_inches='tight')
    plt.show()
