import h5py

import colorcet as cc
from matplotlib import pyplot as plt 
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

LABEL_SIZE = 12
TEXT_SIZE = 8
TICK_SIZE = 8

FNAME_SETUP = "data/setup.h5"
FNAME_PRIOR = "data/prior/prior.h5"
FNAME_MCMC = "data/pcn/pcn.h5"

FOLDER_EKI = "data/eki"

FNAMES_EKI = [
    f"{FOLDER_EKI}/eki_100.h5",
    f"{FOLDER_EKI}/eki_fisher_100.h5",
    f"{FOLDER_EKI}/eki_inflation_100.h5",
]

PLOTS_FOLDER = "plots/thesis"

PLOT_SETUP = True

PLOT_TRACES = False

PLOT_PRIOR_SAMPLES = False
PLOT_POST_SAMPLES = False

PLOT_LOC_COMPARISON = False

PLOT_CBAR = False

# MCMC plotting
WARM_UP = 500
N_CHAINS = 5

with h5py.File("data/pcn/pcn.h5", "r") as f:

    trace_1 = f["trace_1"][:, :]
    trace_2 = f["trace_2"][:, :]
    trace_3 = f["trace_3"][:, :]
    mu_post = f["mean"][:, :]
    
    samples_1 = f["samples_1"][:, :]
    samples_2 = f["samples_2"][:, :]

def read_setup():

    with h5py.File(FNAME_SETUP, "r") as f:
        
        xs = f["xs"][:]
        well_centres = f["well_centres"][:]

        well_ps = f["well_ps"][:, :].T
        well_ts = f["well_ts"][:]

        well_ps_obs = f["well_ps_obs"][:, :].T
        well_ts_obs = f["well_ts_obs"][:]

        u_t = f["u_t"][:, :]

    return xs, well_centres, well_ps, well_ts, well_ps_obs, well_ts_obs, u_t

def read_prior_results():

    with h5py.File(FNAME_PRIOR, "r") as f:
        samples = f["us"][:, :]

    return samples

def read_mcmc():

    with h5py.File(FNAME_MCMC, "r") as f:
        mu = f["mean"][:, :]
        std = f["stds"][:, :]
    
    return mu, std

def read_ensemble_results(fname):
    
    with h5py.File(fname, "r") as f: 
        mu = f["μ_post_2"][:, :]
        std = f["σ_post_2"][:, :]
    
    return mu, std

if PLOT_SETUP:

    def well_name(i):
        well_nums = [7, 4, 1, 8, 5, 2, 9, 6, 3]
        return r"\texttt{Well " + f"{well_nums[i]}" + r"}"

    well_to_plot = 3
    pmin, pmax = -33, -28.5

    xs, well_centres, well_ps, well_ts, well_ps_obs, well_ts_obs, u_t = read_setup()

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), layout="constrained")
    
    axes[0].set_facecolor("lightskyblue")

    axes[0].set_xlim((0, 1000))
    axes[0].set_ylim((0, 1000))

    axes[0].set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("$y$ [m]", fontsize=LABEL_SIZE)

    axes[0].set_xticks([0, 500, 1000])
    axes[0].set_yticks([0, 500, 1000])

    axes[0].tick_params(length=0)
    axes[1].tick_params(length=0)
    
    for (i, c) in enumerate(well_centres):
        axes[0].scatter(c[0], c[1], s=10, facecolors="none", edgecolors="k")
        axes[0].text(c[0], c[1]+40, well_name(i), ha="center", fontsize=TEXT_SIZE)

    m = axes[1].pcolormesh(xs, xs, u_t.T, cmap="turbo", vmin=pmin, vmax=pmax)
    cbar = fig.colorbar(m, ax=axes[1], label="ln(Permeability) [ln(m$^2$)]")
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    axes[1].set_xlabel("$x$ [m]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("$y$ [m]", fontsize=LABEL_SIZE)

    axes[1].set_xticks([0, 500, 1000])
    axes[1].set_yticks([0, 500, 1000])

    axes[2].plot(well_ts, well_ps[well_to_plot] / 1e6, color="k", linestyle="--", linewidth=1.2)
    axes[2].scatter(well_ts_obs, well_ps_obs[well_to_plot] / 1e6, color="k", s=5)

    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    axes[2].spines["bottom"].set_bounds(0, 120)
    axes[2].set_xlim((-12, 132))

    axes[2].spines["left"].set_bounds(16, 21)
    axes[2].set_ylim((15.5, 21.5))

    axes[2].set_xticks([0, 60, 120])

    axes[2].set_xlabel("Time [Days]", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Pressure [MPa]", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        ax.set_box_aspect(1)

    # plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/setup.pdf")

if PLOT_TRACES:

    trace_colors = plt.cm.coolwarm(np.linspace(0, 1, N_CHAINS))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    xs_trace_1 = 10 * np.arange(10 * WARM_UP, 200_000)
    xs_trace_2 = 100 * np.arange(WARM_UP, 20_000)

    for i in range(N_CHAINS):
        axes[0].plot(xs_trace_1, trace_1[i, 10 * WARM_UP:], color=trace_colors[i], linewidth=0.5)
        axes[1].plot(xs_trace_2, trace_2[i, WARM_UP:], color=trace_colors[i], linewidth=0.5)
        axes[2].plot(xs_trace_2, trace_3[i, WARM_UP:], color=trace_colors[i], linewidth=0.5)

    for ax in axes.flat:

        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["bottom"].set_bounds(0, 2e6)
        ax.set_xlim((-1e5, 2.1e6))
        
        ax.set_box_aspect(1)

    axes[0].spines["left"].set_bounds(-7000, -6000)
    axes[0].set_ylim((-7050, -5950))

    axes[1].spines["left"].set_bounds(-4, 4)
    axes[1].set_ylim((-4-8/20, 4+8/20))
    
    axes[2].spines["left"].set_bounds(-4, 1)
    axes[2].set_ylim((-4-5/20, 1+5/20))

    axes[0].set_ylabel("log-posterior", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("$\\xi_{2049}$", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("$\omega_{l}$", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Iteration", fontsize=LABEL_SIZE)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/traces.pdf")

if PLOT_PRIOR_SAMPLES:

    kmin, kmax = -34, -28

    samples = read_prior_results()

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))

    for i, s in enumerate(samples[16:24]):
        s = np.reshape(s, (80, 80))
        axes.flat[i].pcolormesh(s.T, vmin=kmin, vmax=kmax, cmap="turbo")
        axes.flat[i].set_box_aspect(1)
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/prior_samples.pdf")

if PLOT_POST_SAMPLES:

    kmin, kmax = -34, -28

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))

    samples = np.vstack((samples_1, samples_2[1:]))

    for i, s in enumerate(samples[:8]):
        s = np.reshape(s, (80, 80))
        axes.flat[i].pcolormesh(s.T, vmin=kmin, vmax=kmax, cmap="turbo")
        axes.flat[i].set_box_aspect(1)
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/post_samples.pdf")

if PLOT_LOC_COMPARISON:

    pmin, pmax = -33, -28.5
    cmin, cmax = 0.1, 0.8
    
    mu_mcmc, std_mcmc = read_mcmc()

    mu_eki, std_eki = read_ensemble_results(FNAMES_EKI[0])
    mu_eki_ft, std_eki_ft = read_ensemble_results(FNAMES_EKI[1])
    mu_eki_inf, std_eki_inf = read_ensemble_results(FNAMES_EKI[2])

    fig, axes = plt.subplots(4, 4, figsize=(10, 11))

    axes[0][0].pcolormesh(mu_mcmc.T, vmin=pmin, vmax=pmax, cmap="turbo")
    axes[0][1].pcolormesh(mu_eki.T, vmin=pmin, vmax=pmax, cmap="turbo")
    axes[0][2].pcolormesh(mu_eki_ft.T, vmin=pmin, vmax=pmax, cmap="turbo")
    axes[0][3].pcolormesh(mu_eki_inf.T, vmin=pmin, vmax=pmax, cmap="turbo")

    axes[2][0].pcolormesh(std_mcmc.T, vmin=cmin, vmax=cmax, cmap="turbo")
    axes[2][1].pcolormesh(std_eki.T, vmin=cmin, vmax=cmax, cmap="turbo")
    axes[2][2].pcolormesh(std_eki_ft.T, vmin=cmin, vmax=cmax, cmap="turbo")
    axes[2][3].pcolormesh(std_eki_inf.T, vmin=cmin, vmax=cmax, cmap="turbo")

    axes[0][0].set_title("pCN-MCMC", fontsize=LABEL_SIZE)
    axes[0][1].set_title("EKI", fontsize=LABEL_SIZE)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()

if PLOT_CBAR:

    vmin, vmax = -34, -28
    cmap = "turbo"
    label = "$u$ [ln(m$^2$)]"
    plt.figure(figsize=(4, 3))

    plt.pcolormesh(np.ones((2, 2)), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(label=label)

    plt.savefig(f"{PLOTS_FOLDER}/cbar.pdf")

#plt.pcolormesh(mu_post.T)
#plt.show()