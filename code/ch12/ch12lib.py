"""Library for Chapter 12, Machine Learning in Finance. Plain NumPy and SciPy: regularised regression,
time-series cross-validation, regression trees and their ensembles, a small neural network with
backpropagation written out, out-of-sample statistics for return forecasts, learned hedges, and the
statistics of backtest overfitting."""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

EULER = 0.5772156649015329


# ---------------------------------------------------------------- 12.1 regularised regression
def standardise(X, mean=None, sd=None):
    mean = X.mean(axis=0) if mean is None else mean
    sd = X.std(axis=0) if sd is None else sd
    return (X - mean) / sd, mean, sd


def ridge(X, y, lam):
    """argmin ||y - Xb||^2 + lam ||b||^2 for centred X and y, through the SVD (works for p > n)."""
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((d / (d ** 2 + lam)) * (U.T @ y))


def ridge_path(X, y, lams):
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    uy = U.T @ y
    return np.array([Vt.T @ ((d / (d ** 2 + l)) * uy) for l in lams])


def ridge_df(X, lam):
    d = np.linalg.svd(X, compute_uv=False)
    return float(np.sum(d ** 2 / (d ** 2 + lam)))


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso(X, y, lam, b0=None, tol=1e-10, itmax=100000):
    """argmin (1/2n)||y - Xb||^2 + lam ||b||_1 by cyclic coordinate descent, for centred X and y.
    Works on the Gram matrix: g = X'(y - Xb)/n is updated after each coordinate move."""
    n, p = X.shape
    G = (X.T @ X) / n
    b = np.zeros(p) if b0 is None else np.asarray(b0, float).copy()
    g = ((X.T @ y) / n - G @ b).tolist()
    Gl, bl, d = G.tolist(), b.tolist(), np.diag(G).tolist()
    for _ in range(itmax):
        dmax = 0.0
        for j in range(p):
            if d[j] == 0:
                continue
            old = bl[j]; z = g[j] + d[j] * old
            new = (z - lam if z > lam else z + lam if z < -lam else 0.0) / d[j]   # soft threshold
            if new != old:
                dl = new - old; Gj = Gl[j]
                for k in range(p):
                    g[k] -= Gj[k] * dl
                bl[j] = new; dmax = max(dmax, abs(dl))
        if dmax < tol:
            break
    return np.array(bl)


def lasso_path(X, y, lams):
    out, b = [], None
    for l in sorted(lams, reverse=True):
        b = lasso(X, y, l, b); out.append(b.copy())
    order = np.argsort(np.argsort(-np.asarray(lams)))
    return np.array(out)[order]


def lasso_lam_max(X, y):
    return float(np.max(np.abs(X.T @ y)) / len(y))


def kkt_gap(X, y, b, lam):
    """Largest violation of the lasso optimality conditions."""
    g = X.T @ (y - X @ b) / len(y)
    on = b != 0
    v1 = np.abs(g[on] - lam * np.sign(b[on])).max() if on.any() else 0.0
    v2 = np.maximum(np.abs(g[~on]) - lam, 0).max() if (~on).any() else 0.0
    return float(max(v1, v2))


def toeplitz_corr(p, rho):
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


# ---------------------------------------------------------------- 12.2 cross-validation in time
def kfold_blocks(n, k):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def purged_train(n, test, h, embargo=0):
    """Training indices for a contiguous test block when label t uses data t+1..t+h: drop every
    observation whose label window overlaps the test labels' windows, and an embargo after the block."""
    lo, hi = test[0], test[-1]
    idx = np.arange(n)
    keep = (idx + h < lo + 1) | (idx > hi + h - 1 + embargo)
    keep &= ~np.isin(idx, test)
    return idx[keep]


def cv_r2(X, y, folds, train_of):
    """Pooled out-of-fold R^2 of OLS against the training mean, over the given folds."""
    sse = sst = 0.0
    for te in folds:
        tr = train_of(te)
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        f = np.column_stack([np.ones(len(te)), X[te]]) @ b
        sse += np.sum((y[te] - f) ** 2); sst += np.sum((y[te] - y[tr].mean()) ** 2)
    return 1 - sse / sst


# ---------------------------------------------------------------- 12.3 trees and ensembles
class Tree:
    """A CART regression tree grown greedily on squared error. Nodes are stored in flat arrays."""

    def __init__(self, max_depth=6, min_leaf=5, mtry=None, rng=None):
        self.max_depth, self.min_leaf, self.mtry, self.rng = max_depth, min_leaf, mtry, rng

    def fit(self, X, y):
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []
        self._grow(X, y, np.arange(len(y)), 0)
        self.feat = np.array(self.feat); self.thr = np.array(self.thr)
        self.left = np.array(self.left); self.right = np.array(self.right); self.val = np.array(self.val)
        return self

    def _new(self, v):
        self.feat.append(-1); self.thr.append(0.0); self.left.append(-1); self.right.append(-1); self.val.append(v)
        return len(self.val) - 1

    def _grow(self, X, y, idx, depth):
        yi = y[idx]; node = self._new(float(yi.mean()))
        n = len(idx)
        if depth >= self.max_depth or n < 2 * self.min_leaf:
            return node
        p = X.shape[1]
        feats = np.arange(p) if self.mtry is None else self.rng.choice(p, self.mtry, replace=False)
        best = (0.0, -1, 0.0)
        tot, m = yi.sum(), self.min_leaf
        for j in feats:
            xj = X[idx, j]; o = np.argsort(xj, kind="stable"); xs = xj[o]; cs = np.cumsum(yi[o])
            k = np.arange(m, n - m + 1)              # left child holds the first k sorted points
            k = k[xs[k - 1] < xs[k]]                 # split only between distinct values
            if len(k) == 0:
                continue
            sl = cs[k - 1]
            gain = sl ** 2 / k + (tot - sl) ** 2 / (n - k) - tot ** 2 / n
            i = int(np.argmax(gain))
            if gain[i] > best[0] + 1e-12:
                best = (float(gain[i]), int(j), 0.5 * (xs[k[i] - 1] + xs[k[i]]))
        if best[1] < 0:
            return node
        j, t = best[1], best[2]
        go = X[idx, j] <= t
        self.feat[node], self.thr[node] = j, t
        l = self._grow(X, y, idx[go], depth + 1); self.left[node] = l
        r = self._grow(X, y, idx[~go], depth + 1); self.right[node] = r
        return node

    def predict(self, X):
        node = np.zeros(len(X), dtype=int)
        for _ in range(self.max_depth + 1):
            f = self.feat[node]; inner = f >= 0
            if not inner.any():
                break
            x = X[np.arange(len(X)), np.where(inner, f, 0)]
            go = x <= self.thr[node]
            node = np.where(inner, np.where(go, self.left[node], self.right[node]), node)
        return self.val[node]

    @property
    def leaves(self):
        return int((self.feat < 0).sum())


def forest(X, y, B, rng, mtry=None, max_depth=30, min_leaf=5, keep=False):
    """Bagging (mtry=None) or a random forest: trees on bootstrap samples, averaged."""
    trees = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        trees.append(Tree(max_depth, min_leaf, mtry, rng).fit(X[i], y[i]))
    return trees


def forest_predict(trees, X, cumulative=False):
    P = np.array([t.predict(X) for t in trees])
    return np.cumsum(P, axis=0) / np.arange(1, len(trees) + 1)[:, None] if cumulative else P.mean(axis=0)


def boost(X, y, M, nu=0.1, depth=3, min_leaf=5, Xtest=None):
    """Gradient boosting for squared loss: each tree fits the current residuals, the negative gradient.
    Returns (f0, trees) and, if Xtest is given, the test predictions after each round."""
    f0 = float(y.mean()); F = np.full(len(y), f0); trees = []
    path = None if Xtest is None else np.empty((M, len(Xtest)))
    G = None if Xtest is None else np.full(len(Xtest), f0)
    for m in range(M):
        t = Tree(depth, min_leaf).fit(X, y - F)
        F += nu * t.predict(X); trees.append(t)
        if Xtest is not None:
            G = G + nu * t.predict(Xtest); path[m] = G
    return f0, trees, path


def boost_predict(f0, trees, X, nu):
    return f0 + nu * sum(t.predict(X) for t in trees)


# ---------------------------------------------------------------- 12.4 a feed-forward network
def mlp_init(sizes, rng):
    """Weights scaled by 1/sqrt(fan in) so the tanh units start in their linear range."""
    return [(rng.standard_normal((a, b)) / math.sqrt(a), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def mlp_forward(params, X):
    """Hidden layers use tanh, the output is linear. Returns the output and the list of activations."""
    acts = [X]
    for l, (W, b) in enumerate(params):
        z = acts[-1] @ W + b
        acts.append(z if l == len(params) - 1 else np.tanh(z))
    return acts[-1], acts


def mlp_backward(params, acts, dout):
    """Backpropagation: dout is dLoss/d(output), shape (n, 1). Returns the gradients, layer by layer,
    and dLoss/d(input)."""
    grads = [None] * len(params)
    delta = dout                                   # dLoss/dz for the output layer (linear)
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        grads[l] = (acts[l].T @ delta, delta.sum(axis=0))
        g_in = delta @ W.T
        if l > 0:
            delta = g_in * (1 - acts[l] ** 2)      # through tanh: d tanh(z)/dz = 1 - tanh(z)^2
    return grads, g_in


def mse_loss(params, X, y):
    out, acts = mlp_forward(params, X)
    e = out[:, 0] - y
    return 0.5 * float(np.mean(e ** 2)), acts, (e / len(y))[:, None]


def adam_train(params, X, y, epochs, rng, lr=1e-3, batch=256, Xv=None, yv=None, every=50, lr_end=None):
    """Mini-batch Adam (Kingma and Ba), with the step size falling geometrically from lr to lr_end over
    the epochs if lr_end is given. Returns the trained parameters and the loss history."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    t, hist = 0, []
    n = len(y)
    lr0 = lr
    for ep in range(epochs):
        if lr_end is not None:
            lr = lr0 * (lr_end / lr0) ** (ep / max(epochs - 1, 1))
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            _, acts, dout = mse_loss(params, X[i], y[i])
            grads, _ = mlp_backward(params, acts, dout)
            t += 1
            new = []
            for l, ((W, b), (gW, gb)) in enumerate(zip(params, grads)):
                mW = b1 * m[l][0] + (1 - b1) * gW; mb = b1 * m[l][1] + (1 - b1) * gb
                vW = b2 * v[l][0] + (1 - b2) * gW ** 2; vb = b2 * v[l][1] + (1 - b2) * gb ** 2
                m[l] = (mW, mb); v[l] = (vW, vb)
                c1, c2 = 1 - b1 ** t, 1 - b2 ** t
                new.append((W - lr * (mW / c1) / (np.sqrt(vW / c2) + eps), b - lr * (mb / c1) / (np.sqrt(vb / c2) + eps)))
            params = new
        if ep % every == every - 1 or ep == epochs - 1:
            tr = mse_loss(params, X, y)[0]
            va = mse_loss(params, Xv, yv)[0] if Xv is not None else float("nan")
            hist.append((ep + 1, tr, va))
    return params, np.array(hist)


def bs_call(S, K, T, sig, r=0.0):
    S, T = np.asarray(S, float), np.asarray(T, float)
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / v
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - v)


def bs_delta(S, K, T, sig, r=0.0):
    S, T = np.asarray(S, float), np.asarray(T, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return np.where(T > 0, norm.cdf(d1), (S > K).astype(float))


def call_from_total_vol(m, v):
    """Call price per unit of strike at zero rate, as a function of moneyness m = S/K and v = sigma sqrt(T)."""
    d1 = np.log(m) / v + 0.5 * v
    return m * norm.cdf(d1) - norm.cdf(d1 - v)


# ---------------------------------------------------------------- 12.5 out-of-sample statistics
def r2_os(r, f, bench):
    r, f, bench = map(np.asarray, (r, f, bench))
    return 1 - np.sum((r - f) ** 2) / np.sum((r - bench) ** 2)


def clark_west(r, f, bench):
    """Clark and West's adjusted MSPE statistic: t-statistic of the mean of
    (r - bench)^2 - [(r - f)^2 - (bench - f)^2], with its one-sided p-value."""
    r, f, bench = map(np.asarray, (r, f, bench))
    d = (r - bench) ** 2 - ((r - f) ** 2 - (bench - f) ** 2)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return float(t), float(1 - norm.cdf(t))


def ols_fit(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def ct_gain(S, R2):
    """Campbell and Thompson: proportional rise in a mean-variance timer's expected excess return,
    relative to the best constant weight, from a predictor with population R^2."""
    return R2 * (1 + S ** 2) / (S ** 2 * (1 - R2))


def mv_weights(f, var, gamma=3.0, lo=0.0, hi=1.5):
    return np.clip(f / (gamma * var), lo, hi)


def ce_annual(rx, w, rf, gamma=3.0):
    """Certainty equivalent return, annualised (times 12), of a portfolio with weight w in equities."""
    rp = w * rx + rf
    return 12 * (rp.mean() - 0.5 * gamma * rp.var(ddof=1))


# ---------------------------------------------------------------- 12.6 learned hedges
def gbm_paths(n, steps, T, S0, mu, sig, rng):
    dt = T / steps
    z = rng.standard_normal((n, steps))
    lr = (mu - 0.5 * sig ** 2) * dt + sig * math.sqrt(dt) * z
    return S0 * np.exp(np.column_stack([np.zeros(n), np.cumsum(lr, axis=1)]))


def hedge_features(S, tau, K, centres, width):
    """Features for the hedge ratio: a constant and logistic steps in z = ln(S/K)/sqrt(tau), alone and
    multiplied by sqrt(tau). They are the hidden units of a one-layer network whose first layer is
    fixed, and one set of output weights serves every date."""
    z = (np.log(S / K) / np.sqrt(tau))[..., None]
    b = 1 / (1 + np.exp(-(z - centres) / width))
    return np.concatenate([np.ones(S.shape + (1,)), b, np.sqrt(tau)[..., None] * b], axis=-1)


def hedge_design(paths, K, T, centres, width):
    """Phi[n, k, j] is feature j at date k; G[n, j] = sum_k Phi[n, k, j] (S_{k+1} - S_k) is its gain."""
    steps = paths.shape[1] - 1
    tau = T - np.arange(steps) * T / steps
    S = paths[:, :-1]; dS = np.diff(paths, axis=1)
    Phi = hedge_features(S, np.broadcast_to(tau, S.shape), K, centres, width)
    return Phi, np.einsum("nkj,nk->nj", Phi, dS)


def regression_hedge(paths, H, K, T, centres, width):
    """Quadratic hedge with no costs: regress the claim on a constant and the feature gains."""
    Phi, G = hedge_design(paths, K, T, centres, width)
    A = np.column_stack([np.ones(len(H)), G])
    c = np.linalg.lstsq(A, H, rcond=None)[0]
    return float(c[0]), c[1:]


def hedge_deltas(theta, paths, K, T, centres, width):
    Phi, _ = hedge_design(paths, K, T, centres, width)
    return Phi @ theta


def pnl_from_deltas(D, paths, H, p, kappa):
    """Hedger's P&L: premium p, gains of holding D_k over each step, the claim paid, and proportional
    costs on every trade, including setting up the hedge and closing it at expiry."""
    dS = np.diff(paths, axis=1)
    full = np.column_stack([np.zeros(len(D)), D, np.zeros(len(D))])
    trades = np.diff(full, axis=1)
    cost = kappa * np.sum(np.abs(trades) * paths, axis=1)
    return p + np.sum(D * dS, axis=1) - H - cost, cost


def turnover(D):
    return float(np.mean(np.sum(np.abs(np.diff(np.column_stack([np.zeros(len(D)), D, np.zeros(len(D))]), axis=1)), axis=1)))


def mv_price(loss, gamma):
    """Mean-variance price of a loss X: E[X] + (gamma/2) Var X."""
    return float(loss.mean() + 0.5 * gamma * loss.var())


def cost_objective(th, Phi, G, H, paths, kappa, gamma, eps=1e-3):
    """E[X] + (gamma/2) Var X and its gradient, where X = H - gains + costs is the hedger's loss and the
    cost of a trade x at price S is kappa S (sqrt(x^2 + eps^2) - eps), a smoothed |x|. The gradient is
    E[(1 + gamma (X - E X)) dX/dtheta]."""
    n = len(H)
    D = Phi @ th
    tr = np.diff(np.column_stack([np.zeros(n), D, np.zeros(n)]), axis=1)
    root = np.sqrt(tr ** 2 + eps ** 2)
    X = H - G @ th + kappa * np.sum((root - eps) * paths, axis=1)
    m = X.mean()
    w = (1 + gamma * (X - m)) / n
    dc = tr / root * paths                              # d cost / d trade
    dD = dc[:, :-1] - dc[:, 1:]                         # trade k = D_k - D_{k-1}, with D_{-1} = D_steps = 0
    g = -G.T @ w + kappa * np.einsum("n,nk,nkj->j", w, dD, Phi)
    return m + 0.5 * gamma * np.mean((X - m) ** 2), g


def cost_hedge(theta0, paths, H, K, T, centres, width, kappa, gamma, eps=1e-3, iters=500):
    """Minimise the mean-variance price of the hedger's loss over the weights by L-BFGS."""
    Phi, G = hedge_design(paths, K, T, centres, width)
    res = optimize.minimize(cost_objective, theta0, args=(Phi, G, H, paths, kappa, gamma, eps), jac=True,
                            method="L-BFGS-B", options=dict(maxiter=iters, gtol=1e-9))
    return res.x, res


# ---------------------------------------------------------------- 12.7 overfitting
def expected_max_normal(N):
    """E max of N independent standard normals, by quadrature of x N phi(x) Phi(x)^(N-1)."""
    f = lambda x: x * N * norm.pdf(x) * norm.cdf(x) ** (N - 1)
    return integrate.quad(f, -12, 12, limit=400, epsabs=1e-12)[0]


def expected_max_approx(N):
    """The approximation of Bailey and Lopez de Prado (2014)."""
    return (1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * math.e))


def sharpe_se(sr, T, skew=0.0, kurt=3.0):
    """Standard error of a per-period Sharpe ratio from T observations with the given skewness and
    kurtosis (not excess), as in Chapter 8."""
    return math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1))


def psr(sr, T, skew=0.0, kurt=3.0, bench=0.0):
    """Probabilistic Sharpe ratio: probability that the true Sharpe ratio exceeds bench."""
    return float(norm.cdf((sr - bench) / sharpe_se(sr, T, skew, kurt)))


def dsr(sr, T, N, var_sr, skew=0.0, kurt=3.0):
    """Deflated Sharpe ratio: PSR against the expected maximum of N zero-skill Sharpe ratios whose
    cross-sectional variance is var_sr."""
    bench = math.sqrt(var_sr) * expected_max_approx(N)
    return psr(sr, T, skew, kurt, bench), bench


def holm(p, alpha):
    """Holm's step-down procedure: boolean array of rejections."""
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    rej = np.zeros(m, bool)
    for i, j in enumerate(o):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def bh(p, alpha):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = np.nonzero(p[o] <= alpha * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[:ok[-1] + 1]] = True
    return rej


def _sr(x):
    sd = x.std(0, ddof=1)
    return np.divide(x.mean(0), sd, out=np.zeros(x.shape[1]), where=sd > 0)


def cscv_pbo(R, S=16):
    """Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey,
    Borwein, Lopez de Prado and Zhu): R is T x N returns of N strategies. Split the rows into S
    blocks; for each half of the blocks pick the strategy with the best in-sample Sharpe ratio and
    record the relative rank of that strategy out of sample. Returns PBO, the logits and the pairs of
    in-sample and out-of-sample Sharpe ratios of the chosen strategy."""
    from itertools import combinations
    T, N = R.shape
    blocks = np.array_split(np.arange(T), S)
    logits, pairs = [], []
    for c in combinations(range(S), S // 2):
        ins = np.concatenate([blocks[i] for i in c])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        si, so = _sr(R[ins]), _sr(R[oos])
        b = int(np.argmax(si))
        w = (np.sum(so < so[b]) + 0.5 * (np.sum(so == so[b]) - 1) + 1) / (N + 1)   # relative rank in (0, 1)
        logits.append(math.log(w / (1 - w))); pairs.append((si[b], so[b]))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits, np.array(pairs)


def band_policy(target, width):
    """Hold the previous position while it lies within target +/- width; otherwise trade to the
    nearer edge. Starts from no position."""
    D = np.empty_like(target); prev = np.zeros(len(target))
    for k in range(target.shape[1]):
        prev = np.clip(prev, target[:, k] - width[:, k], target[:, k] + width[:, k]); D[:, k] = prev
    return D


def bs_gamma(S, K, T, sig):
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    return norm.pdf(d1) / (S * sig * np.sqrt(T))


def ww_width(S, K, T, sig, kappa, gamma):
    """Whalley and Wilmott's asymptotic half-width of the no-trade band, (3 kappa S Gamma^2 / (2 gamma))^(1/3),
    for proportional cost kappa and absolute risk aversion gamma."""
    return (1.5 * kappa * S * bs_gamma(S, K, T, sig) ** 2 / gamma) ** (1 / 3)
