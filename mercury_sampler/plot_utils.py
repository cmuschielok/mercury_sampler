"""Convenience functions for plotting the pore network samplings.

This module is the interface of the `mercury_sampler` package to the `atp` package.

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from matplotlib.colors import Colormap
from scipy.spatial import cKDTree
from typing import Optional, Tuple

from ase import Atoms


def plot_data(xy: Tuple[float, float] = (0.0, 0.0),
              z: Optional[float] = 0.0,
              z_order: Optional[float] = None,
              values: Optional[np.ndarray] = None,
              radius: Optional[float] = 0.05,
              ax: Optional[Axes] = None,
              edgecolor: Optional[str] = None,
              facecolor: Optional[str] = None,
              cmap: Optional[Colormap] = None,
              **kwargs):
    """Plots a data point onto the canvas.

    The color of the point is chosen according to a scalar value from
    a colormap. Can replace the default `plot_data` method of the
    AtomsPlotter class.

    Args:
        xy: Tuple[float, float], default=(0.0, 0.0)

        z: float, default=0.0

        z_order: float or None, default=None

        values: (N, ) array_like or None, default=None
        Any values attached to the data point. This function takes the first
        entry and uses it to choose a color from `cmap`. `values` has to be
        normalized.

        radius: float, default=0.05
        The radius of the circle representing the data point.

        ax: Axes or None, default=None

        edgecolor: str or None, default=None
        The color of the circle edge.

        facecolor: str or None, default=None
        The color of the circle area.

        cmap: Colormap or None, default=None

    """
    if ax is None:
        ax = plt.gca()
    if cmap is not None and facecolor is None:
        facecolor = cmap(values[0])
    circle = mpl.patches.Circle(xy=xy, radius=radius, edgecolor=edgecolor, facecolor=facecolor,
                                zorder=z_order, **kwargs)
    ax.add_patch(circle)


def get_neighbor_field(data: np.ndarray, radius: Optional[float]=1.0,
                       normalize=False) -> np.ndarray:
    """
    Calculate the number of neighbors within a given distance for a set of data
    points.

    Args:
        data: (N, 3) array_like
        A set of data points.

        radius: float, default=1.0
        The radius within which the neighbors are considered to lie.

        normalize: bool, default=False
        Normalize the output.

    Returns:
        n_neighs_arr: (N, ) ndarray

    """
    _tree = cKDTree(data)
    neighbors = _tree.query_ball_point(x=data, r=radius, p=2.0)
    n_neighbors_arr = [len(n) for n in neighbors]
    n_neighbors_arr = np.array(n_neighbors_arr)
    if normalize:
        _span = n_neighbors_arr.max() - n_neighbors_arr.min()
        n_neighbors_arr = (n_neighbors_arr - n_neighbors_arr.min()) / _span
    return n_neighbors_arr


def _correct_for_pbc(atoms: Atoms, data: np.ndarray, radius: Optional[float]=1.0) -> np.ndarray:
    """
    Add points within `radius` of the boundary of the periodic images.
    """
    _data_frac = atoms.cell.scaled_positions(data)
    _radius_a = radius / np.linalg.norm(atoms.cell[0])
    _radius_b = radius / np.linalg.norm(atoms.cell[1])
    _radius_c = radius / np.linalg.norm(atoms.cell[2])
    _data_ext_1 = _data_frac[np.where(_data_frac[:, 0] >= 1.0 - _radius_a)[0], :]
    _data_ext_1 -= np.array([1.0, 0.0, 0.0])
    _data_ext_6 = _data_frac[np.where(_data_frac[:, 0] < _radius_a)[0], :]
    _data_ext_6 += np.array([1.0, 0.0, 0.0])
    _data_ext_2 = _data_frac[np.where(_data_frac[:, 1] >= 1.0 - _radius_b)[0], :]
    _data_ext_2 -= np.array([0.0, 1.0, 0.0])
    _data_ext_5 = _data_frac[np.where(_data_frac[:, 1] < _radius_b)[0], :]
    _data_ext_5 += np.array([0.0, 1.0, 0.0])
    _data_ext_3 = _data_frac[np.where(_data_frac[:, 2] >= 1.0 - _radius_c)[0], :]
    _data_ext_3 -= np.array([0.0, 0.0, 1.0])
    _data_ext_4 = _data_frac[np.where(_data_frac[:, 2] < _radius_c)[0], :]
    _data_ext_4 += np.array([0.0, 0.0, 1.0])
    _data_frac_ext = np.vstack((_data_frac, _data_ext_1, _data_ext_2, _data_ext_3,
                                _data_ext_4, _data_ext_5, _data_ext_6))
    data_ext = atoms.cell.cartesian_positions(_data_frac_ext)
    return data_ext


def _cut_out_original_data(atoms: Atoms, data: np.ndarray) -> np.ndarray:
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 0] >= 0.0)[0], :]
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 0] < 1.0)[0], :]
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 1] >= 0.0)[0], :]
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 1] < 1.0)[0], :]
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 2] >= 0.0)[0], :]
    data = data[np.where(atoms.cell.scaled_positions(data[:, :3])[:, 2] < 1.0)[0], :]
    return data


def get_pore_sampling_with_neighbors(atoms: Atoms, data: np.ndarray,
                                     radius=1.0) -> np.ndarray:
    _data = _correct_for_pbc(atoms, data, radius=radius)
    _n_neigh_arr = get_neighbor_field(_data, radius=radius, normalize=True)
    _data = np.vstack((_data.T, _n_neigh_arr)).T
    data_neigh = _cut_out_original_data(atoms, _data)
    return data_neigh