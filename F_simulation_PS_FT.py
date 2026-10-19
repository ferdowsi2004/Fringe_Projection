import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from skimage.restoration import unwrap_phase

np.random.seed(0)

# ---------- Simulation ----------
def generate_fringe(W, H, period, n, N, height=None, A=127.5, B=127.5):
    f0 = 1 / period
    x = np.tile(np.arange(W), (H, 1)).astype(np.float64)
    phase = 2 * np.pi * x / period
    if height is not None:
        phase = phase + 2 * np.pi * f0 * height
    img = A + B * np.cos(phase + 2 * np.pi * n / N)
    return np.clip(np.round(img), 0, 255)

def simulate_object_height(W, H):
    xx, yy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    return 30 * np.exp(-(xx**2 + yy**2) / 0.15)

def add_awgn(img, sigma):
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 255)

# ---------- Method 1: Four-step Phase-Shifting (PS) ----------
def psp_phase_4step(imgs):
    I1, I2, I3, I4 = imgs
    return np.arctan2(I4 - I2, I1 - I3)

# ---------- Method 2: Fourier Transform (FT) ----------
def ftp_phase(img, f0, radius=22):
    H, W = img.shape
    shift = int(round(f0 * W))
    window = np.outer(np.hanning(H), np.hanning(W))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * window))
    y, x = np.ogrid[:H, :W]
    mask = (x - (W // 2 + shift))**2 + (y - H // 2)**2 < radius**2
    F = np.where(mask, F, 0)
    F = np.fft.ifftshift(np.roll(F, -shift, axis=1))
    return np.angle(np.fft.ifft2(F))

# ---------- Helpers ----------
def wrap(p):
    return np.arctan2(np.sin(p), np.cos(p))

def reconstruct_height(delta_phase, f0):
    h = unwrap_phase(delta_phase) / (2 * np.pi * f0)
    border = np.concatenate([h[:10].ravel(), h[-10:].ravel(),
                             h[:, :10].ravel(), h[:, -10:].ravel()])
    return h - np.median(border)

def rms_error(h, gt, roi):
    return np.sqrt(np.mean((h[roi] - gt[roi])**2))

def run_trial(sigma):
    """One noisy realization -> (PS height map, FT height map)."""
    imgs = [add_awgn(generate_fringe(W, H, period, n, N, gt), sigma) for n in range(N)]
    h_psp = reconstruct_height(wrap(psp_phase_4step(imgs) - ref_psp), f0)
    h_ftp = reconstruct_height(wrap(ftp_phase(imgs[0], f0) - ref_ftp), f0)
    return h_psp, h_ftp

# ---------- Parameters ----------
W, H = 640, 480
N = 4
period = 12
f0 = 1 / period
sigmas = [0,20,40,60]
trials = 30                      # number of noise realizations averaged per sigma
roi = (slice(60, -60), slice(120, -120))

gt = simulate_object_height(W, H)

# ---------- Noise-free references ----------
ref_psp = psp_phase_4step([generate_fringe(W, H, period, n, N) for n in range(N)])
ref_ftp = ftp_phase(generate_fringe(W, H, period, 0, N), f0)

# ---------- (1) Single realization per sigma -> 3D surfaces ----------
results = {}
for sigma in sigmas:
    h_psp, h_ftp = run_trial(sigma)
    results[sigma] = (h_psp, h_ftp, rms_error(h_psp, gt, roi), rms_error(h_ftp, gt, roi))

stride = 4
X, Y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
zmin, zmax = -5, 40
fig = plt.figure(figsize=(20, 9))
for col, sigma in enumerate(sigmas):
    h_psp, h_ftp, r_psp, r_ftp = results[sigma]
    for row, (h, name, r) in enumerate([(h_psp, "Phase-Shifting (4-step)", r_psp),
                                         (h_ftp, "Fourier Transform", r_ftp)]):
        ax = fig.add_subplot(2, len(sigmas), row * len(sigmas) + col + 1, projection='3d')
        ax.plot_surface(X, Y, h[::stride, ::stride], cmap='viridis',
                        vmin=zmin, vmax=zmax, linewidth=0, antialiased=False)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=35, azim=-60)
        ax.set_title(f"{name}\n$\\sigma$ = {sigma}   |   RMS = {r:.3f}", fontsize=11)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)"); ax.set_zlabel("height")
fig.suptitle("Reconstructed 3D surface: Phase-Shifting vs Fourier Transform at different noise levels",
             fontsize=15)
plt.tight_layout()
plt.savefig("reconstructed_surfaces.png", dpi=180, bbox_inches='tight')
plt.show()

# ---------- (2) Many trials per sigma -> averaged RMS comparison ----------
mean_psp, std_psp, mean_ftp, std_ftp = [], [], [], []
for sigma in sigmas:
    e_psp, e_ftp = [], []
    for _ in range(trials):
        h_psp, h_ftp = run_trial(sigma)
        e_psp.append(rms_error(h_psp, gt, roi))
        e_ftp.append(rms_error(h_ftp, gt, roi))
    mean_psp.append(np.mean(e_psp)); std_psp.append(np.std(e_psp))
    mean_ftp.append(np.mean(e_ftp)); std_ftp.append(np.std(e_ftp))
    print(f"sigma={sigma:>2}  PS={mean_psp[-1]:.4f} (+/-{std_psp[-1]:.4f})   "
          f"FT={mean_ftp[-1]:.4f} (+/-{std_ftp[-1]:.4f})")

plt.figure(figsize=(8, 5))
plt.errorbar(sigmas, mean_psp, yerr=std_psp, fmt='o-', capsize=4, linewidth=2,
             markersize=8, label='Phase-Shifting (4-step)')
plt.errorbar(sigmas, mean_ftp, yerr=std_ftp, fmt='s--', capsize=4, linewidth=2,
             markersize=8, label='Fourier Transform')
for s, a, b in zip(sigmas, mean_psp, mean_ftp):
    up_psp = a >= b
    plt.annotate(f"{a:.3f}", (s, a), textcoords="offset points",
                 xytext=(14, 8) if up_psp else (14, -16), fontsize=9, color='C0')
    plt.annotate(f"{b:.3f}", (s, b), textcoords="offset points",
                 xytext=(14, -16) if up_psp else (14, 8), fontsize=9, color='C1')
plt.xticks(sigmas)
plt.xlim(0, 63)
plt.xlabel(r"Image noise $\sigma$ (gray levels)")
plt.ylabel("RMS height error (central ROI)")
plt.title(f"RMS error vs noise (mean $\\pm$ std of {trials} trials)")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rms_comparison.png", dpi=200)
plt.show()
