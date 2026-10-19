import numpy as np
import cv2
import plotly.graph_objects as go
from skimage.restoration import unwrap_phase


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = (
    r"C:\Users\Gnanamoorthy\Desktop\MTech Sem 1"
    r"\Optical Instrumentation\3D Profiling\DLP"
)

X_START, X_END = 1500, 3500       # crop
Y_START, Y_END = 500, 2000

MOD_THRESH   = 8       # threshold filter: min fringe modulation
SAT_LEVEL    = 250     # threshold filter: pixels >= this are rejected
SPIKE_THRESH = 0.25    # rad; deviation from local median counted as spike
MEDIAN_SIZE  = 9       # median window (odd)
SIGN         = -1      # -1 = (ref - obj), +1 = (obj - ref)
DISPLAY_STEP = 4       # plot subsampling
ZMIN, ZMAX   = -2, 10  # z-axis range of the plot


# ============================================================
# 4-STEP PHASE + THRESHOLD MASK
# ============================================================

def four_step_phase(kind, pitch):

    images = []
    for i in range(1, 5):
        path = rf"{BASE_DIR}\{kind}_pitch{pitch}_p{i}.jpeg"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read:\n{path}")
        images.append(img[Y_START:Y_END, X_START:X_END].astype(np.float64))

    I1, I2, I3, I4 = images
    num, den = I4 - I2, I1 - I3                  # 2B sin(phi), 2B cos(phi)

    phase = np.arctan2(num, den)
    mask = 0.5 * np.hypot(num, den) > MOD_THRESH  # fringe strength

    for im in images:                             # saturation
        mask &= im < SAT_LEVEL

    return phase, mask


# ============================================================
# SPIKE REMOVAL  (local: does not change the global slope)
# ============================================================

def remove_spikes(z, valid):

    # fill invalid pixels by normalised Gaussian averaging of valid neighbours
    v = valid.astype(np.float64)
    num = cv2.GaussianBlur(np.where(valid, z, 0.0), (0, 0), 15)
    den = cv2.GaussianBlur(v, (0, 0), 15)
    fill = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.median(z[valid]))
    filled = np.where(valid, z, fill)

    # median filter via OpenCV (8-bit copy)
    lo, hi = np.percentile(filled, [0.5, 99.5])
    scale = 255.0 / max(hi - lo, 1e-9)
    q = np.clip((filled - lo) * scale, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(q, MEDIAN_SIZE).astype(np.float64) / scale + lo

    # replace spikes with the local median
    bad = valid & (np.abs(z - med) > SPIKE_THRESH)
    out = np.where(valid, z, np.nan)
    out[bad] = med[bad]

    print(f"Spike pixels replaced: {bad.sum()}")
    return out


# ============================================================
# PROCESS ONE PITCH
# ============================================================

def process_pitch(pitch):

    print(f"\n=== PITCH {pitch} ===")

    ref_phase, ref_mask = four_step_phase("ref", pitch)
    obj_phase, obj_mask = four_step_phase("object", pitch)

    mask = ref_mask & obj_mask
    print(f"Valid pixels: {100.0 * mask.mean():.1f} %")

    rel = np.angle(np.exp(1j * (obj_phase - ref_phase)))      # wrapped
    unwrapped = unwrap_phase(np.ma.array(rel, mask=~mask))    # masked unwrap
    unwrapped = np.ma.filled(unwrapped, np.nan)

    unwrapped = remove_spikes(unwrapped, mask & np.isfinite(unwrapped))

    return SIGN * (unwrapped - np.nanmedian(unwrapped))       # offset only


# ============================================================
# PLOT
# ============================================================

def plot_surface(phase, pitch):

    Z = phase[::DISPLAY_STEP, ::DISPLAY_STEP]
    x = np.arange(X_START, X_END, DISPLAY_STEP)[:Z.shape[1]]
    y = np.arange(Y_START, Y_END, DISPLAY_STEP)[:Z.shape[0]]

    fig = go.Figure(go.Surface(
        x=x, y=y, z=Z,
        colorscale="Turbo", cmin=ZMIN, cmax=ZMAX,
        colorbar=dict(title="Phase (rad)"),
    ))

    fig.update_layout(
        title=f"4-Step Relative Phase - Pitch {pitch}",
        width=1400, height=900,
        scene=dict(
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            zaxis=dict(title="Unwrapped Phase (rad)", range=[ZMIN, ZMAX]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=0.75, z=0.4),
        ),
    )
    fig.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for pitch in (1, 2, 3):
        plot_surface(process_pitch(pitch), pitch)