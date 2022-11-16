import jax
from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import scipy.interpolate as sci
import scipy.ndimage as scn
import scipy.optimize as sco
from stream_membership.helpers import two_normal_mixture, two_truncated_normal_mixture

from .gd1_model import GD1BackgroundModel, GD1StreamModel, w_to_z


class Initializer:
    pass


class BackgroundInitializer(Initializer):
    def __init__(self, data, phi1_bins=None, plot=False):
        # `data` should be a dictionary:
        self.data = data

        if phi1_bins is None:
            # 5 degree bins in phi1
            phi1_bins = np.arange(
                GD1BackgroundModel.coord_bounds["phi1"][0],
                GD1BackgroundModel.coord_bounds["phi1"][1] + 1e-3,
                5.0,
            )
        self.phi1_bins = phi1_bins
        self._phi1x = 0.5 * (self.phi1_bins[:-1] + self.phi1_bins[1:])

        self.plot = plot

    def init_ln_n0(self, **_):
        H, _ = np.histogram(self.data["phi1"], bins=self.phi1_bins)
        dx = np.diff(self.phi1_bins)[0]
        log_n = np.log(scn.gaussian_filter(H, 2.0)) - np.log(dx)

        tmp = sci.InterpolatedUnivariateSpline(self._phi1x, log_n, ext=3)
        knot_vals = tmp(GD1BackgroundModel.knots["ln_n0"])

        if self.plot:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
            ax.plot(self._phi1x, log_n, drawstyle="steps-mid")
            ax.scatter(GD1BackgroundModel.knots["ln_n0"], knot_vals, color="tab:purple")
            ax.set_xlabel(r"$\phi_1$")
            ax.set_ylabel(r"$\ln n(\phi_1)$")

        return knot_vals

    def init_pm1(self, x0_pm1=None, **_):
        # Collapse in phi1 and fit total distribution:

        # Initial guess - from playing in notebook
        x0_pm1 = jnp.array([0.65, 0.75, np.log(3), -4, np.log(4)])

        @jax.jit
        def objective(p, data, data_err):
            w, mean1, ln_std1, mean2, ln_std2 = p

            model = two_truncated_normal_mixture(
                *p,
                low=GD1BackgroundModel.coord_bounds["pm1"][0],
                high=GD1BackgroundModel.coord_bounds["pm1"][1],
                yerr=data_err
            )
            ln_prob = model.log_prob(data)
            ln_prob = (
                ln_prob
                + dist.Normal(2, 5).log_prob(mean1)
                + dist.Normal(2, 5).log_prob(mean2)
            )
            return -ln_prob.sum() / len(data)

        obj_grad = jax.jit(jax.grad(objective))
        res_pm1 = sco.minimize(
            objective,
            x0=x0_pm1,
            jac=lambda x, *args: np.array(obj_grad(x, *args)),
            method="l-bfgs-b",
            bounds=[(1e-2, 1 - 1e-2), (-10, 20), (-5, 5), (-10, 20), (-5, 5)],
            args=(self.data["pm1"], self.data["pm1_err"]),
            options=dict(maxls=1000),
        )
        assert res_pm1.success

        if self.plot:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

            grid = np.linspace(*GD1BackgroundModel.coord_bounds["pm1"], 128)
            ax.hist(self.data["pm1"], bins=grid, density=True)

            init_grid_vals = two_truncated_normal_mixture(
                *x0_pm1,
                low=GD1BackgroundModel.coord_bounds["pm1"][0],
                high=GD1BackgroundModel.coord_bounds["pm1"][1],
                yerr=0.0
            ).log_prob(grid)
            ax.plot(grid, np.exp(init_grid_vals), color="tab:green", label="init")

            _grid_vals = two_truncated_normal_mixture(
                *res_pm1.x,
                low=GD1BackgroundModel.coord_bounds["pm1"][0],
                high=GD1BackgroundModel.coord_bounds["pm1"][1],
                yerr=0.0
            ).log_prob(grid)
            ax.plot(grid, np.exp(_grid_vals), color="tab:red", label="opt")
            ax.set_xlabel(r"$\mu_{\phi_1}$")
            ax.legend(loc="best")

        knot_vals = {}
        for i, (name, size) in enumerate(GD1BackgroundModel.shapes["pm1"].items()):
            knot_vals[name] = np.full(size, res_pm1.x[i])

        return knot_vals

    def init_pm2(self, x0_pm2=None, **_):
        if x0_pm2 is None:
            x0_pm2 = [0.0, 0.75, np.log(2.5), -4, np.log(4)]

        @jax.jit
        def objective(p, data, data_err):
            w, mean1, ln_std1, mean2, ln_std2 = p
            model = two_normal_mixture(*p, yerr=data_err)
            ln_prob = model.log_prob(data)
            ln_prob = (
                ln_prob
                + dist.Normal(0, 5).log_prob(mean1)
                + dist.Normal(0, 5).log_prob(mean2)
            )
            return -ln_prob.sum() / len(data)

        obj_grad = jax.jit(jax.grad(objective))
        res_pm2 = sco.minimize(
            objective,
            x0=x0_pm2,
            jac=lambda x, *args: np.array(obj_grad(x, *args)),
            method="l-bfgs-b",
            bounds=[(0 + 1e-2, 1 - 1e-2), (-10, 10), (-5, 5), (-10, 10), (-5, 5)],
            args=(self.data["pm2"], self.data["pm2_err"]),
            options=dict(maxls=1000),
        )
        assert res_pm2.success

        if self.plot:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

            # TODO: hard-set
            grid = np.linspace(-15, 15, 128)
            ax.hist(self.data["pm2"], bins=grid, density=True)

            init_grid_vals = two_normal_mixture(*x0_pm2, yerr=0.0).log_prob(grid)
            ax.plot(grid, np.exp(init_grid_vals), color="tab:green", label="init")

            _grid_vals = two_normal_mixture(*res_pm2.x, yerr=0.0).log_prob(grid)
            ax.plot(grid, np.exp(_grid_vals), color="tab:red", label="opt")
            ax.set_xlabel(r"$\mu_{\phi_2}$")
            ax.legend(loc="best")

        knot_vals = {}
        for i, (name, size) in enumerate(GD1BackgroundModel.shapes["pm2"].items()):
            knot_vals[name] = np.full(size, res_pm2.x[i])

        return knot_vals

    def init(self, **kwargs):
        init_p = {}

        init_p["ln_n0"] = self.init_ln_n0()

        if "pm1" in GD1BackgroundModel.coord_names:
            pp = self.init_pm1(**kwargs)
            if "w" in pp:
                pp["z"] = w_to_z(pp.pop("w"))
            init_p["pm1"] = pp

        if "pm2" in GD1BackgroundModel.coord_names:
            pp = self.init_pm2(**kwargs)
            if "w" in pp:
                pp["z"] = w_to_z(pp.pop("w"))
            init_p["pm2"] = pp

        return init_p


class StreamInitializer(Initializer):
    def __init__(self, data, phi1_bins=None, plot=False):
        # `data` should be a dictionary:
        self.data = data

        if phi1_bins is None:
            # 5 degree bins in phi1
            phi1_bins = np.arange(
                GD1StreamModel.coord_bounds["phi1"][0],
                GD1StreamModel.coord_bounds["phi1"][1] + 1e-3,
                5.0,
            )
        self.phi1_bins = phi1_bins
        self._phi1x = 0.5 * (self.phi1_bins[:-1] + self.phi1_bins[1:])

        self.plot = plot

    def init_ln_n0(self, **_):
        H, _ = np.histogram(self.data["phi1"], bins=self.phi1_bins)
        dx = np.diff(self.phi1_bins)[0]
        log_n = np.log(H) - np.log(dx)

        tmp = sci.InterpolatedUnivariateSpline(self._phi1x, log_n, k=1, ext=3)
        knot_vals = tmp(GD1StreamModel.knots["ln_n0"])

        if self.plot:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
            ax.plot(self._phi1x, log_n)
            ax.scatter(GD1StreamModel.knots["ln_n0"], knot_vals, color="tab:purple")

            interp_tmp = sci.InterpolatedUnivariateSpline(
                GD1StreamModel.knots["ln_n0"],
                knot_vals,
                k=GD1StreamModel.spline_ks["ln_n0"],
            )
            ax.plot(
                self.phi1_bins,
                interp_tmp(self.phi1_bins),
                marker="",
                color="tab:purple",
            )

            ax.set_xlabel(r"$\phi_1$")
            ax.set_ylabel(r"$\ln n(\phi_1)$")

        return knot_vals

    def init_phi2(self, phi2_bins=None, **_):
        if phi2_bins is None:
            phi2_bins = np.arange(
                GD1StreamModel.coord_bounds["phi2"][0],
                GD1StreamModel.coord_bounds["phi2"][1] + 1e-3,
                0.5,
            )

        H2d, xe, ye = np.histogram2d(
            self.data["phi1"], self.data["phi2"], bins=(self.phi1_bins, phi2_bins)
        )

        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])

        H2d = scn.gaussian_filter1d(H2d, sigma=1)
        peak_idx = H2d.argmax(axis=1)
        H2d /= H2d[np.arange(H2d.shape[0]), peak_idx][:, None]

        knot_vals = {}
        knot_vals["mean"] = sci.InterpolatedUnivariateSpline(xc, yc[peak_idx], k=1)(
            GD1StreamModel.knots["phi2"]
        )
        knot_vals["ln_std"] = np.full_like(knot_vals["mean"], -1)

        if self.plot:
            fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))

            ax.pcolormesh(xe, ye, H2d.T, shading="auto", vmin=0, vmax=1, cmap="Greys")
            ax.plot(xc, yc[peak_idx], color="tab:green", label="init")

            interp_tmp = sci.InterpolatedUnivariateSpline(
                GD1StreamModel.knots["phi2"],
                knot_vals["mean"],
                k=GD1StreamModel.spline_ks["phi2"]["mean"],
            )
            ax.plot(
                self.phi1_bins,
                interp_tmp(self.phi1_bins),
                marker="",
                color="tab:purple",
            )

            ax.set_xlabel(r"$\phi_1$")
            ax.set_xlabel(r"$\phi_2$")

        return knot_vals

    def init_pm1(self, phi1_bins=None, **_):
        if phi1_bins is None:
            phi1_bins = self.phi1_bins

        H, xe, ye = np.histogram2d(
            self.data["phi1"],
            self.data["pm1"],
            bins=(phi1_bins, np.arange(*GD1StreamModel.coord_bounds["pm1"], 0.1)),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])

        H = scn.gaussian_filter1d(H, sigma=1)
        peak_idx = H.argmax(axis=1)
        ii = np.arange(H.shape[0])
        good_peaks = peak_idx > 0
        H /= H[ii, peak_idx][:, None]

        knot_vals = {}
        knot_vals["mean"] = sci.InterpolatedUnivariateSpline(
            xc[good_peaks], yc[peak_idx[good_peaks]], k=1
        )(GD1StreamModel.knots["pm1"])
        knot_vals["ln_std"] = np.full_like(GD1StreamModel.knots["pm1"], -3)

        if self.plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
            cs = ax.pcolormesh(xe, ye, H.T, vmin=0, vmax=1, cmap="Greys")
            ax.plot(xc, yc[peak_idx], color="tab:green")

            ax.scatter(
                GD1StreamModel.knots["pm1"], knot_vals["mean"], color="tab:purple"
            )
            interp_tmp = sci.InterpolatedUnivariateSpline(
                GD1StreamModel.knots["pm1"], knot_vals["mean"], k=3
            )
            ax.plot(
                self.phi1_bins,
                interp_tmp(self.phi1_bins),
                marker="",
                color="tab:purple",
            )

            ax.set_xlabel(r"$\mu_{\phi_1}$")

        return knot_vals

    def init_pm2(self, phi1_bins=None, **_):
        if phi1_bins is None:
            phi1_bins = self.phi1_bins

        H, xe, ye = np.histogram2d(
            self.data["phi1"],
            self.data["pm2"],
            bins=(phi1_bins, np.arange(-4, 4 + 1e-3, 0.1)),
        )
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])

        H = scn.gaussian_filter1d(H, sigma=1)
        peak_idx = H.argmax(axis=1)
        ii = np.arange(H.shape[0])
        good_peaks = peak_idx > 0
        H /= H[ii, peak_idx][:, None]

        knot_vals = {}
        knot_vals["mean"] = sci.InterpolatedUnivariateSpline(
            xc[good_peaks], yc[peak_idx[good_peaks]], k=1
        )(GD1StreamModel.knots["pm2"])
        knot_vals["ln_std"] = np.full_like(GD1StreamModel.knots["pm2"], -3)

        if self.plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
            cs = ax.pcolormesh(xe, ye, H.T, vmin=0, vmax=1, cmap="Greys")
            ax.plot(xc, yc[peak_idx], color="tab:green")

            ax.scatter(
                GD1StreamModel.knots["pm2"], knot_vals["mean"], color="tab:purple"
            )
            interp_tmp = sci.InterpolatedUnivariateSpline(
                GD1StreamModel.knots["pm2"], knot_vals["mean"], k=3
            )
            ax.plot(
                self.phi1_bins,
                interp_tmp(self.phi1_bins),
                marker="",
                color="tab:purple",
            )

            ax.set_xlabel(r"$\mu_{\phi_2}$")

        return knot_vals

    def init(self, **kwargs):
        init_p = {}

        init_p["ln_n0"] = self.init_ln_n0()

        if "phi2" in GD1StreamModel.coord_names:
            init_p["phi2"] = self.init_phi2(**kwargs)

        if "pm1" in GD1StreamModel.coord_names:
            pp = self.init_pm1(**kwargs)
            if "w" in pp:
                pp["z"] = w_to_z(pp.pop("w"))
            init_p["pm1"] = pp

        if "pm2" in GD1StreamModel.coord_names:
            pp = self.init_pm2(**kwargs)
            if "w" in pp:
                pp["z"] = w_to_z(pp.pop("w"))
            init_p["pm2"] = pp

        return init_p
