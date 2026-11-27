"""set-aware 指标实现（协议：gaze/prereg/S1.prereg.md，读数前已冻结）。

预测接口：hyps [K,2] + 权重 w[K]（降序），可选 pmf（64x64 展平成 4096，和为 1）。
所有坐标为图像归一化坐标。网格格 (r,c) 的展平下标 = r*64 + c，中心 ((c+.5)/64, (r+.5)/64)。
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment, linprog
from sklearn.cluster import DBSCAN

G = 64
NCELL = G * G
XS = (np.arange(G) + 0.5) / G
_CX, _CY = np.meshgrid(XS, XS)
CELLS = np.stack([_CX.ravel(), _CY.ravel()], 1)  # [4096,2]
LAMBDA = 1e-3
KS = (1, 3, 5, 10)
MODE_KS = (1, 3, 5)
RS = (0.05, 0.10)


def cell_index(pts):
    pts = np.atleast_2d(pts)
    ix = np.clip((pts[:, 0] * G).astype(int), 0, G - 1)
    iy = np.clip((pts[:, 1] * G).astype(int), 0, G - 1)
    return iy * G + ix


# ---------------- 热图 -> pmf / 假设 ----------------
def heat_to_pmf(h, gamma=1.0, lam=LAMBDA):
    hp = np.maximum(np.asarray(h, np.float64), 1e-12) ** gamma
    P = hp.ravel() / hp.sum()
    return (1 - lam) * P + lam / NCELL


def nms_peaks(h, K, radius=0.10):
    """贪心取峰：第 k 个峰 = 与已选峰中心距离均 >= radius 的格子中 h 最大者。返回展平格下标。"""
    flat = np.asarray(h, np.float64).ravel()
    avail = np.ones(NCELL, bool)
    out = []
    for _ in range(K):
        if not avail.any():
            break
        idx = int(np.argmax(np.where(avail, flat, -np.inf)))
        out.append(idx)
        avail &= np.hypot(CELLS[:, 0] - CELLS[idx, 0], CELLS[:, 1] - CELLS[idx, 1]) >= radius
    return np.array(out, int)


def disc_mass(P, idx, r=0.10):
    d = np.hypot(CELLS[:, 0] - CELLS[idx, 0], CELLS[:, 1] - CELLS[idx, 1])
    return float(P[d <= r].sum())


def sample_hyps(P, rng, n=10):
    cells = rng.choice(NCELL, size=n, p=P)
    jit = rng.random((n, 2))
    r, c = cells // G, cells % G
    return np.stack([(c + jit[:, 0]) / G, (r + jit[:, 1]) / G], 1)


# ---------------- 假设类指标 ----------------
def hyp_metrics(H, A, ks=KS, rs=RS):
    H, A = np.asarray(H, float), np.asarray(A, float)
    D = np.hypot(H[:, None, 0] - A[None, :, 0], H[:, None, 1] - A[None, :, 1])  # [K,n]
    dm = np.hypot(H[:, 0] - A[:, 0].mean(), H[:, 1] - A[:, 1].mean())
    out = {}
    for K in ks:
        k = min(K, len(H))
        near = D[:k].min(0)
        out[f'minL2@{K}'] = float(D[:k].min())
        out[f'avgL2@{K}'] = float(dm[:k].min())
        out[f'cham@{K}'] = float(near.mean())
        for r in rs:
            out[f'cov{r}@{K}'] = float((near <= r).mean())
    return out


def gt_modes(A, eps=0.075, m=3):
    lab = DBSCAN(eps=eps, min_samples=m).fit_predict(A)
    ms = [(int((lab == k).sum()), k, A[lab == k].mean(0)) for k in sorted(set(lab) - {-1})]
    ms.sort(key=lambda t: (-t[0], t[1]))
    if not ms:
        return np.median(A, 0)[None]
    return np.stack([c for _, _, c in ms])


def mode_pr(H, w, modes, K, r=0.10):
    k = min(K, len(H))
    Hk, wk = np.asarray(H[:k], float), np.asarray(w[:k], float)
    D = np.hypot(Hk[:, None, 0] - modes[None, :, 0], Hk[:, None, 1] - modes[None, :, 1])
    allowed = D <= r
    ri, ci = linear_sum_assignment(np.where(allowed, -1000.0 + D, 0.0))
    ok = allowed[ri, ci]
    return float(wk[ri[ok]].sum() / wk.sum()), float(ok.sum() / len(modes))


def mode_metrics(H, w, modes, ks=MODE_KS):
    out = {}
    for K in ks:
        p, r = mode_pr(H, w, modes, K)
        out[f'modeP@{K}'], out[f'modeR@{K}'] = p, r
    return out


# ---------------- 分布类指标 ----------------
def nll(P, A):
    return float(-np.log(P[cell_index(A)]).mean())


_ROWS = {}


def _sparse_blocks(n):
    if n not in _ROWS:
        rows = sp.kron(sp.identity(NCELL), np.ones((1, n)), format='csr')
        cols = sp.kron(np.ones((1, NCELL)), sp.identity(n), format='csr')
        _ROWS[n] = sp.vstack([rows, cols[:-1]], format='csr')
    return _ROWS[n]


def w1_grid(P, A):
    """P（4096 格 pmf）与 A 上均匀经验分布之间的精确 W1（HiGHS）。"""
    A = np.asarray(A, float)
    n = len(A)
    P = np.asarray(P, float) / np.sum(P)
    D = np.hypot(CELLS[:, None, 0] - A[None, :, 0], CELLS[:, None, 1] - A[None, :, 1])
    beq = np.concatenate([P, np.full(n - 1, 1.0 / n)])
    r = linprog(D.ravel(), A_eq=_sparse_blocks(n), b_eq=beq, bounds=(0, None), method='highs')
    assert r.status == 0, r.message
    return float(r.fun)


def w1_discrete(A1, A2):
    """两个均匀经验分布之间的精确 W1。"""
    A1, A2 = np.asarray(A1, float), np.asarray(A2, float)
    n1, n2 = len(A1), len(A2)
    D = np.hypot(A1[:, None, 0] - A2[None, :, 0], A1[:, None, 1] - A2[None, :, 1])
    rows = sp.kron(sp.identity(n1), np.ones((1, n2)), format='csr')
    cols = sp.kron(np.ones((1, n1)), sp.identity(n2), format='csr')
    Aeq = sp.vstack([rows, cols[:-1]], format='csr')
    beq = np.concatenate([np.full(n1, 1.0 / n1), np.full(n2 - 1, 1.0 / n2)])
    r = linprog(D.ravel(), A_eq=Aeq, b_eq=beq, bounds=(0, None), method='highs')
    assert r.status == 0, r.message
    return float(r.fun)


def disc_pairs(P, A, peak_idx, r=0.10, top=3):
    pred, obs = [], []
    for idx in peak_idx[:top]:
        pred.append(disc_mass(P, idx, r))
        obs.append(float((np.hypot(A[:, 0] - CELLS[idx, 0], A[:, 1] - CELLS[idx, 1]) <= r).mean()))
    return np.array(pred), np.array(obs)


def ece(pred, obs, bins=10):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    b = np.minimum((pred * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            tot += s.sum() / len(pred) * abs(obs[s].mean() - pred[s].mean())
    return float(tot), float((pred - obs).mean())


# ---------------- 参照 ----------------
def loo_kde_nll(A, sigma, lam=LAMBDA):
    A = np.asarray(A, float)
    n = len(A)
    D2 = ((CELLS[None, :, :] - A[:, None, :]) ** 2).sum(-1)
    Gm = np.exp(-D2 / (2 * sigma ** 2))
    Gm /= Gm.sum(1, keepdims=True)
    S = Gm.sum(0)
    cells = cell_index(A)
    nl = []
    for i in range(n):
        P = (1 - lam) * (S - Gm[i]) / (n - 1) + lam / NCELL
        nl.append(-np.log(P[cells[i]]))
    return float(np.mean(nl))


def split_half_w1(A, rng, reps=10):
    n = len(A)
    vals = []
    for _ in range(reps):
        perm = rng.permutation(n)
        vals.append(w1_discrete(A[perm[: n // 2]], A[perm[n // 2:]]))
    return float(np.mean(vals))
