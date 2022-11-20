import numpy as np
import scipy.ndimage as scn

__all__ = ["plot_data_projections"]

_default_labels = {
    "phi2": r"$\phi_2$",
    "pm1": r"$\mu_{\phi_1}$",
    "pm2": r"$\mu_{\phi_2}$",
}


def plot_data_projections(data, grids, smooth=2.0, axes=None, label=True, **kwargs):
    if grids is None:
        grids = {}

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True, sharey="row", constrained_layout=True
        )

    for i, name in enumerate(["phi2", "pm1", "pm2"]):
        H_data, xe, ye = np.histogram2d(
            data["phi1"], data[name], bins=(grids["phi1"], grids[name])
        )
        if smooth is not None:
            H_data = scn.gaussian_filter(H_data, smooth)
        axes[i].pcolormesh(xe, ye, H_data.T, **kwargs)
        axes[i].set_ylim(grids[name].min(), grids[name].max())

        if label:
            axes[i].set_ylabel(_default_labels[name])

    axes[0].set_xlim(grids["phi1"].min(), grids["phi1"].max())

    return axes.flat[0].figure, axes
