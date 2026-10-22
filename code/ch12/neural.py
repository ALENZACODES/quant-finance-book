"""Section 12.4: a feed-forward network trained by backpropagation to reproduce the Black-Scholes call
price as a function of moneyness and total volatility, with its delta read off by backpropagating to
the inputs, against a polynomial fitted by least squares. Also writes fig17_nn_gamma.png, the
network's second difference in moneyness against the Black-Scholes gamma.

Run on its own:  python code/ch12/neural.py   (writes code/ch12/out/neural.json)
"""
from __future__ import annotations
import numpy as np
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

MLO, MHI, VLO, VHI = 0.75, 1.25, 0.05, 0.5
NTR, NVA, HID, EPOCHS, LR, LREND, BATCH, DEG = 4000, 1000, 32, 4000, 5e-3, 5e-5, 128, 6


def scale(m, v):
    return np.column_stack([(2 * m - MLO - MHI) / (MHI - MLO), (2 * v - VLO - VHI) / (VHI - VLO)])


def poly(Z, deg=DEG):
    return np.column_stack([Z[:, 0] ** i * Z[:, 1] ** j for i in range(deg + 1) for j in range(deg + 1 - i)])


def main():
    rng = rng_for(4)
    m = rng.uniform(MLO, MHI, NTR + NVA); v = rng.uniform(VLO, VHI, NTR + NVA)
    Z = scale(m, v); c = L.call_from_total_vol(m, v)
    Ztr, ctr, Zva, cva = Z[:NTR], c[:NTR], Z[NTR:], c[NTR:]
    params = L.mlp_init([2, HID, HID, 1], rng)
    nparam = sum(W.size + b.size for W, b in params)
    params, hist = L.adam_train(params, Ztr, ctr, EPOCHS, rng, lr=LR, batch=BATCH, Xv=Zva, yv=cva, every=25, lr_end=LREND)
    put("nnNtr", NTR, "{:,.0f}"); put("nnHid", HID, "{:.0f}"); put("nnEpochs", EPOCHS, "{:,.0f}"); put("nnParams", nparam, "{:,.0f}")
    put("nnLr", LR, "{:.3f}"); put("nnLrEnd", LREND, "{:.5f}"); put("nnBatch", BATCH, "{:.0f}"); put("nnSteps", EPOCHS * int(np.ceil(NTR / BATCH)), "{:,.0f}")
    put("nnMlo", MLO); put("nnMhi", MHI); put("nnVlo", VLO); put("nnVhi", VHI, "{:.1f}")
    # test grid
    mg, vg = np.meshgrid(np.linspace(MLO, MHI, 101), np.linspace(VLO, VHI, 46))
    Zg = scale(mg.ravel(), vg.ravel()); cg = L.call_from_total_vol(mg.ravel(), vg.ravel())
    out, acts = L.mlp_forward(params, Zg)
    e = out[:, 0] - cg
    put("nnRmse", 1e4 * np.sqrt(np.mean(e ** 2)), "{:.1f}"); put("nnMax", 1e4 * np.abs(e).max(), "{:.1f}")
    # delta: d c / d m by backpropagating a unit output to the inputs, then the chain rule for the scaling
    _, gin = L.mlp_backward(params, acts, np.ones((len(Zg), 1)))
    dnn = gin[:, 0] * 2 / (MHI - MLO)
    dbs = L.bs_delta(mg.ravel(), 1.0, vg.ravel() ** 2, 1.0)        # sigma = 1, T = v^2 gives total vol v
    put("nnDeltaRmse", np.sqrt(np.mean((dnn - dbs) ** 2)), "{:.4f}"); put("nnDeltaMax", np.abs(dnn - dbs).max(), "{:.3f}")
    i = int(np.argmax(np.abs(dnn - dbs))); put("nnDeltaMaxM", mg.ravel()[i], "{:.2f}"); put("nnDeltaMaxV", vg.ravel()[i], "{:.2f}")
    RAW["nnDeltaMaxSign"] = float(np.sign(dnn[i] - dbs[i]))
    # convexity in moneyness on the grid
    C = out[:, 0].reshape(mg.shape)
    sec = C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]
    pct("nnConvex", (sec < 0).mean(), 1)
    # the polynomial benchmark
    Ptr = poly(Ztr); coef = np.linalg.lstsq(Ptr, ctr, rcond=None)[0]
    ep = poly(Zg) @ coef - cg
    put("nnPolyTerms", Ptr.shape[1], "{:.0f}"); put("nnPolyDeg", DEG, "{:.0f}")
    put("nnPolyRmse", 1e4 * np.sqrt(np.mean(ep ** 2)), "{:.1f}"); put("nnPolyMax", 1e4 * np.abs(ep).max(), "{:.1f}")
    put("nnTrainLoss", hist[-1, 1] * 2 * 1e8, "{:.2f}"); put("nnValLoss", hist[-1, 2] * 2 * 1e8, "{:.2f}")
    RAW["nnHistFirst"] = float(hist[0, 1])
    np.savez(__import__("pathlib").Path(__file__).parent / "out" / "nn_params.npz",
             *[a for W, b in params for a in (W, b)])
    # figure 7: the error surface
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    lim = 1e4 * max(np.abs(e).max(), np.abs(ep).max())
    for a, err, nm in [(ax[0], e, "network"), (ax[1], ep, f"polynomial of degree {DEG}")]:
        im = a.pcolormesh(mg, vg, 1e4 * err.reshape(mg.shape), cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        a.set_xlabel("moneyness $S/K$"); a.set_title(nm, fontsize=9); a.grid(False)
    ax[0].set_ylabel(r"total volatility $\sigma\sqrt{T}$")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("error (basis points of $K$)", fontsize=8)
    save(fig, "fig07_nn_error")
    # figure 8: the training history and the delta
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax[0].semilogy(hist[:, 0], 2 * hist[:, 1], color=BLUE, label="training")
    ax[0].semilogy(hist[:, 0], 2 * hist[:, 2], color=RED, ls="--", label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean squared error"); ax[0].legend(fontsize=7.5)
    for vv, col, ls in [(0.1, BLUE, "-"), (0.4, RED, "--")]:
        mm = np.linspace(MLO, MHI, 200); Zs = scale(mm, np.full_like(mm, vv))
        o, ac = L.mlp_forward(params, Zs); _, gi = L.mlp_backward(params, ac, np.ones((len(mm), 1)))
        ax[1].plot(mm, gi[:, 0] * 2 / (MHI - MLO), color=col, ls=ls, lw=1.2, label=rf"network, $\sigma\sqrt{{T}}={vv}$")
        ax[1].plot(mm, L.bs_delta(mm, 1.0, vv ** 2, 1.0), color=GREY, lw=0.7, ls=":")
    ax[1].plot([], [], color=GREY, lw=0.7, ls=":", label="Black-Scholes")
    ax[1].set_xlabel("moneyness $S/K$"); ax[1].set_ylabel("delta"); ax[1].legend(fontsize=7.2, loc="upper left")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig08_nn_training")
    gamma_figure(params)
    return finish("neural")


def gamma_figure(params):
    """Second difference of the network's price in moneyness, against the Black-Scholes gamma, along
    two lines of total volatility; points where the network's second difference is negative are marked."""
    mm = np.linspace(MLO, MHI, 201); h = mm[1] - mm[0]
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    for a, vv, nm in ((ax[0], 0.1, "Low"), (ax[1], 0.4, "High")):
        o, _ = L.mlp_forward(params, scale(mm, np.full_like(mm, vv))); c = o[:, 0]
        g = (c[2:] - 2 * c[1:-1] + c[:-2]) / h ** 2; x = mm[1:-1]
        gb = L.bs_gamma(x, 1.0, vv ** 2, 1.0)
        a.plot(x, gb, color=GREY, ls=":", lw=1.0, label="Black-Scholes")
        a.plot(x, g, color=BLUE, lw=1.0, label="network")
        neg = g < 0
        a.plot(x[neg], g[neg], "o", color=RED, ms=2.5, label="negative")
        pct(f"gmNeg{nm}", neg.mean(), 0); put(f"gmPeak{nm}", gb.max(), "{:.2f}")
        put(f"gmEdge{nm}", L.bs_gamma(np.array([MHI]), 1.0, vv ** 2, 1.0)[0], "{:.2f}")  # at the edge, m = MHI
        put(f"gmErr{nm}", np.sqrt(np.mean((g - gb) ** 2)), "{:.2f}")
        a.axhline(0, color="#333333", lw=0.5)
        a.set_xlabel("moneyness $S/K$"); a.set_title(rf"$\sigma\sqrt{{T}}={vv}$", fontsize=8.5)
    ax[0].set_ylabel(r"$K\times$ gamma ($\partial^2c/\partial m^2$)"); ax[0].legend(fontsize=7, loc="lower left")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig17_nn_gamma")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
