import h5py

import colorcet as cc
from matplotlib import pyplot as plt 
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

LABEL_SIZE = 12
TICK_SIZE = 8

PLOT_TRACES = False
PLOT_POST_SAMPLES = True

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
    plt.savefig("traces.pdf")

if PLOT_POST_SAMPLES:

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))

    samples = np.vstack((samples_1, samples_2))

    kmin = -34
    kmax = -27

    for i, s in enumerate(samples[:8]):
        s = np.reshape(s, (80, 80))
        axes.flat[i].pcolormesh(s.T, vmin=np.min(samples), vmax=np.max(samples), cmap="turbo")
        axes.flat[i].set_box_aspect(1)
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])

    plt.tight_layout()
    plt.savefig("post_samples.pdf")

#plt.pcolormesh(mu_post.T)
#plt.show()