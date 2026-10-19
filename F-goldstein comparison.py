"""
Multi-pitch vs Goldstein phase unwrapping on a synthetic Gaussian hill.
Needs: numpy, scipy, matplotlib (no other files).
"""
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import cKDTree

# ---------------- settings ----------------
H, W = 600, 800            # image size (px)
PITCHES = [64, 32, 16]     # fringe periods (px), coarse -> fine
PEAK_SHIFT = 28            # fringe shift at the hill top (px); keep < PITCHES[0] / 2
SIGMA = 100                # hill width (px)
NOISE = 20               # Gaussian noise std added to every image (gray levels)
TRIALS = 3                 # noisy repetitions averaged for the comparison
SMOOTH = 2                 # phase smoothing sigma (px), applied to both methods
SEED = 0
TP = 2 * np.pi

# ---------------- synthetic object and fringe images ----------------
y, x = np.mgrid[0:H, 0:W].astype(float)
shift = PEAK_SHIFT * np.exp(-((x - W / 2) ** 2 + (y - H / 2) ** 2) / (2 * SIGMA ** 2))  # object: fringe displacement
truth = TP * shift / PITCHES[-1]                                                         # true phase at finest pitch


def fringes(p, s, noise, rng):
    """4 phase-shifted images (0, 90, 180, 270 deg) of pitch p; s = fringe displacement of the surface."""
    return [np.clip(np.round(128 + 80 * np.cos(TP * (x + s) / p + k * np.pi / 2)
                             + rng.normal(0, noise, x.shape)), 0, 255) for k in range(4)]


def wrapped_diff(p, noise, rng):
    """Wrapped phase difference (deformed - reference) from the 4-step formula, then smoothed via sin/cos."""
    phase = lambda I: np.arctan2(I[3] - I[1], I[0] - I[2])
    d = np.angle(np.exp(1j * (phase(fringes(p, shift, noise, rng)) - phase(fringes(p, 0, noise, rng)))))
    return np.angle(gaussian_filter(np.cos(d), SMOOTH) + 1j * gaussian_filter(np.sin(d), SMOOTH))


# ---------------- method 1: multi-pitch (hierarchical) unwrapping ----------------
def multi_pitch(D, P):
    s = D[0] * P[0] / TP                                        # coarsest pitch: assume no wrapping (|s| < P0/2)
    for d, p in zip(D[1:], P[1:]):
        m = median_filter(np.round(s / p - d / TP), size=5)     # fringe order consistent with the coarser estimate
        s = (d + TP * m) * p / TP                               # refined fringe shift
    return s * TP / P[-1]                                       # unwrapped phase at the finest pitch


# ---------------- method 2: Goldstein branch-cut unwrapping (finest pitch only) ----------------
def wrap(a):
    return a - TP * np.round(a / TP)


def goldstein(p):
    h, w = p.shape
    # 1. residues: wrapped differences around every 2x2 loop should sum to zero; +-1 marks a residue
    q = np.round((wrap(p[:-1, 1:] - p[:-1, :-1]) + wrap(p[1:, 1:] - p[:-1, 1:])
                  + wrap(p[1:, :-1] - p[1:, 1:]) + wrap(p[:-1, :-1] - p[1:, :-1])) / TP).astype(int)
    pos, chg = np.argwhere(q != 0), q[q != 0]

    # 2. branch cuts: grow a box around each residue, connect residues found, stop when total charge is 0
    cut = np.zeros((h, w), bool)

    def draw(a, b):
        a, b = np.asarray(a), np.asarray(b)
        n = int(np.abs(b - a).max()) + 1
        i = np.round(np.linspace(a[0], b[0], n)).astype(int)
        j = np.round(np.linspace(a[1], b[1], n)).astype(int)
        for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):          # mark the 4 corner pixels of each loop
            cut[np.minimum(i + di, h - 1), np.minimum(j + dj, w - 1)] = True

    if len(pos):
        tree, used = cKDTree(pos), np.zeros(len(pos), bool)
        for r in range(len(pos)):
            if used[r]:
                continue
            used[r], members, total, box = True, [r], chg[r], 0
            while total != 0:
                box += 1
                k = 0
                while k < len(members) and total != 0:
                    c = pos[members[k]]
                    gaps = [c[0], h - 2 - c[0], c[1], w - 2 - c[1]]          # distance to top/bottom/left/right edge
                    if min(gaps) < box:                                       # box hit the border: ground the tree
                        e = int(np.argmin(gaps))
                        draw(c, [(0, c[1]), (h - 2, c[1]), (c[0], 0), (c[0], w - 2)][e])
                        total = 0
                        break
                    for n in tree.query_ball_point(c, box, p=np.inf):         # unused residues inside the box
                        if not used[n]:
                            used[n] = True
                            members.append(n)
                            total += chg[n]
                            draw(c, pos[n])
                            if total == 0:
                                break
                    k += 1

    # 3. flood-fill integration that never crosses a cut; cut pixels are filled in last
    u, done = np.zeros((h, w)), np.zeros((h, w), bool)
    free = np.argwhere(~cut)
    s0 = tuple(free[np.argmin(((free - [h / 2, w / 2]) ** 2).sum(1))])
    u[s0], done[s0] = p[s0], True
    queue = deque([s0])
    for k, allowed in enumerate((~cut, np.ones((h, w), bool))):
        if k == 1:
            queue = deque(map(tuple, np.argwhere(done)))
        while queue:
            i, j = queue.popleft()
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < h and 0 <= b < w and allowed[a, b] and not done[a, b]:
                    u[a, b], done[a, b] = u[i, j] + wrap(p[a, b] - p[i, j]), True
                    queue.append((a, b))
    return u


# ---------------- run both methods ----------------
def run(noise, seed):
    rng = np.random.default_rng(seed)
    D = [wrapped_diff(p, noise, rng) for p in PITCHES]
    t0 = time.perf_counter(); mp = multi_pitch(D, PITCHES); t_mp = time.perf_counter() - t0
    t0 = time.perf_counter(); g = goldstein(D[-1]); t_g = time.perf_counter() - t0
    g = g - TP * np.round(np.median(g - truth) / TP)             # Goldstein's absolute level is arbitrary (n * 2pi)
    return mp, g, t_mp, t_g


def err(ph):
    """RMS error (rad) between the unwrapped phase and the true phase."""
    e = ph - truth
    return np.sqrt(np.mean(e ** 2))


print("running (Goldstein is plain Python, so it takes a few seconds)...")
clean = run(0, SEED)
noisy = [run(NOISE, SEED + i) for i in range(TRIALS)]
names = ("multi-pitch", "Goldstein")
S = [np.array([[err(t[i]), t[2 + i]] for t in noisy]) for i in (0, 1)]    # rows: [RMS, time]

print(f"\n{'':30s}{names[0]:>16s}{names[1]:>16s}")
print(f"{'NOISE-FREE  RMS error (rad)':30s}{err(clean[0]):16.4f}{err(clean[1]):16.4f}")
print(f"{'            time (s)':30s}{clean[2]:16.3f}{clean[3]:16.3f}")
for lab, j, f in (("RMS error (rad)", 0, ".4f"), ("time (s)", 1, ".3f")):
    cells = [format(S[i][:, j].mean(), f) + " ± " + format(S[i][:, j].std(), f) for i in (0, 1)]
    print(f"{'NOISE ' + str(NOISE) + '  ' + lab:30s}{cells[0]:>16s}{cells[1]:>16s}")

r_mp, r_g = S[0][:, 0].mean(), S[1][:, 0].mean()
best = "about equal" if abs(r_mp - r_g) < 0.05 * max(r_mp, r_g) else names[int(r_g < r_mp)]
print(f"\nMore accurate at noise {NOISE}: {best}  (RMS {r_mp:.4f} vs {r_g:.4f} rad)")
t_mp, t_g = S[0][:, 1].mean(), S[1][:, 1].mean()
print(f"Faster: {names[int(t_g < t_mp)]} ({max(t_mp, t_g) / min(t_mp, t_g):.0f}x)")

# ---------------- plots ----------------
fig = plt.figure(figsize=(17, 9))
for k, (lab, res) in enumerate((("noise-free", clean), (f"noise {NOISE}", noisy[0]))):
    for c in (0, 1):
        ax = fig.add_subplot(2, 3, 3 * k + c + 1, projection="3d")
        ax.plot_surface(x[::4, ::4], y[::4, ::4], res[c][::4, ::4], cmap="jet", linewidth=0,
                        vmin=truth.min(), vmax=truth.max())
        ax.set_zlim(truth.min() - 1, truth.max() + 1)
        ax.view_init(30, -60)
        ax.set_title(f"{names[c]}, {lab}\nRMS {err(res[c]):.3f} rad, {res[2 + c]:.2f} s", fontsize=10)
        ax.set_zlabel("phase (rad)")
for pos_, j, ylab in ((3, 0, "RMS error vs true phase (rad)"), (6, 1, "computation time (s)")):
    ax = fig.add_subplot(2, 3, pos_)
    ax.bar(names, [S[i][:, j].mean() for i in (0, 1)], yerr=[S[i][:, j].std() for i in (0, 1)],
           color=["tab:green", "tab:orange"], capsize=5)
    ax.set_ylabel(ylab)
    ax.set_title(f"Noise {NOISE}, mean of {TRIALS} trials")
plt.tight_layout()
plt.savefig("synthetic_compare.png", dpi=150)
plt.show()