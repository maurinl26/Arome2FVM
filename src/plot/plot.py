# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import logging

logging.getLogger(__name__)


def plot_zcr(zcr: np.ndarray, xc: np.ndarray, levels: int):
    """Plot all levels of zcr (grid boundaries in z).
    Levels are plotted on a median plan.

    Args:
        grid (Grid): _description_
        zcr (np.ndarray): _description_
    """

    logging.info(f"Max height : {np.max(zcr)}")

    _, ny, _ = zcr.shape

    fig, ax = plt.subplots(nrows=1)
    for lev in range(levels):
        ax.plot(xc, zcr[:, ny // 2, lev], "black", linewidth=0.5)

    ax.set_title("Z coordinates on median plan")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")

    return fig, ax


def plot_orography(x: np.ndarray, y: np.ndarray, orog: np.ndarray):
    """Plot contour 3D map from x, y,z coordinates

    Args:
        x (np.ndarray): longitude array
        y (np.ndarray): latitude array
        orog (np.ndarray): altitude mesh
    """

    # mpl figure
    fig, ax = plt.subplots(nrows=1)

    # contour 3D
    c = ax.contour(x, y, orog, levels=100, cmap="viridis", linewidths=0.5)
    ax.clabel(c, inline=True, fontsize=6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_label("orography")

    return fig
