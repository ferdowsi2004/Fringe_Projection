import numpy as np
import os
import cv2
import plotly.graph_objects as go
from skimage.restoration import unwrap_phase


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = (
    r"C:\Users\Gnanamoorthy\Desktop\MTech Sem 1"
    r"\Optical Instrumentation\3D Profiling\DLP"
)


# ============================================================
# FILES
# ============================================================

REF_FILES = {
    1: [
        BASE_DIR + r"\ref_pitch1_p1.jpeg",
        BASE_DIR + r"\ref_pitch1_p2.jpeg",
        BASE_DIR + r"\ref_pitch1_p3.jpeg",
        BASE_DIR + r"\ref_pitch1_p4.jpeg",
    ],

    2: [
        BASE_DIR + r"\ref_pitch2_p1.jpeg",
        BASE_DIR + r"\ref_pitch2_p2.jpeg",
        BASE_DIR + r"\ref_pitch2_p3.jpeg",
        BASE_DIR + r"\ref_pitch2_p4.jpeg",
    ],

    3: [
        BASE_DIR + r"\ref_pitch3_p1.jpeg",
        BASE_DIR + r"\ref_pitch3_p2.jpeg",
        BASE_DIR + r"\ref_pitch3_p3.jpeg",
        BASE_DIR + r"\ref_pitch3_p4.jpeg",
    ],
}


OBJ_FILES = {
    1: [
        BASE_DIR + r"\object_pitch1_p1.jpeg",
        BASE_DIR + r"\object_pitch1_p2.jpeg",
        BASE_DIR + r"\object_pitch1_p3.jpeg",
        BASE_DIR + r"\object_pitch1_p4.jpeg",
    ],

    2: [
        BASE_DIR + r"\object_pitch2_p1.jpeg",
        BASE_DIR + r"\object_pitch2_p2.jpeg",
        BASE_DIR + r"\object_pitch2_p3.jpeg",
        BASE_DIR + r"\object_pitch2_p4.jpeg",
    ],

    3: [
        BASE_DIR + r"\object_pitch3_p1.jpeg",
        BASE_DIR + r"\object_pitch3_p2.jpeg",
        BASE_DIR + r"\object_pitch3_p3.jpeg",
        BASE_DIR + r"\object_pitch3_p4.jpeg",
    ],
}


# ============================================================
# CROP
# ============================================================

X_START = 1500
X_END   = 3500

Y_START = 500
Y_END   = 2000


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_STEP = 4


# ============================================================
# LOAD 4 IMAGES
# ============================================================

def load_images(file_list):

    images = []

    for filename in file_list:

        img = cv2.imread(
            filename,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise FileNotFoundError(
                f"Cannot read:\n{filename}"
            )

        img = img[
            Y_START:Y_END,
            X_START:X_END
        ]

        images.append(
            img.astype(np.float64)
        )

    return images


# ============================================================
# 4-STEP PHASE
# ============================================================

def four_step_phase(images):

    I1 = images[0]
    I2 = images[1]
    I3 = images[2]
    I4 = images[3]

    phase = np.arctan2(
        I4 - I2,
        I1 - I3
    )

    return phase


# ============================================================
# PROCESS ONE PITCH
# ============================================================

def process_pitch(pitch):

    print("\n")
    print("=" * 60)
    print(f"PITCH {pitch}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load reference
    # --------------------------------------------------------

    print("Loading reference images...")

    ref_images = load_images(
        REF_FILES[pitch]
    )

    # --------------------------------------------------------
    # Load object
    # --------------------------------------------------------

    print("Loading object images...")

    obj_images = load_images(
        OBJ_FILES[pitch]
    )

    # --------------------------------------------------------
    # Calculate reference phase
    # --------------------------------------------------------

    print("Calculating reference phase...")

    ref_phase = four_step_phase(
        ref_images
    )

    # --------------------------------------------------------
    # Calculate object phase
    # --------------------------------------------------------

    print("Calculating object phase...")

    obj_phase = four_step_phase(
        obj_images
    )

    # --------------------------------------------------------
    # Relative phase
    # --------------------------------------------------------

    print("Calculating relative phase...")

    relative_phase = (
        ref_phase - obj_phase
    )

    # --------------------------------------------------------
    # Wrap to -pi ... +pi
    # --------------------------------------------------------

    relative_phase = np.angle(
        np.exp(
            1j * relative_phase
        )
    )

    # --------------------------------------------------------
    # Unwrap
    # --------------------------------------------------------

    print("Unwrapping...")

    unwrapped_phase = unwrap_phase(
        relative_phase
    )

    # --------------------------------------------------------
    # Print values
    # --------------------------------------------------------

    print(
        f"Minimum phase = "
        f"{np.min(unwrapped_phase):.4f} rad"
    )

    print(
        f"Maximum phase = "
        f"{np.max(unwrapped_phase):.4f} rad"
    )

    print(
        f"Phase range = "
        f"{np.ptp(unwrapped_phase):.4f} rad"
    )

    return unwrapped_phase


# ============================================================
# PLOT
# ============================================================

def plot_surface(
    phase,
    pitch
):

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    Z = phase[
        ::DISPLAY_STEP,
        ::DISPLAY_STEP
    ]

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    x = np.arange(
        X_START,
        X_END,
        DISPLAY_STEP
    )

    y = np.arange(
        Y_START,
        Y_END,
        DISPLAY_STEP
    )

    x = x[:Z.shape[1]]
    y = y[:Z.shape[0]]

    X, Y = np.meshgrid(
        x,
        y
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = go.Figure(

        go.Surface(

            x=X,
            y=Y,
            z=Z,

            colorscale="Turbo",

            colorbar=dict(
                title="Phase (rad)"
            )
        )
    )

    fig.update_layout(

        title=(
            f"4-Step Relative Phase "
            f"- Pitch {pitch}"
        ),

        width=1400,
        height=900,

        scene=dict(

            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            zaxis_title="Unwrapped Phase (rad)",

            aspectmode="auto"
        )
    )

    file_name = f"optical_3d_pitch_{pitch}.html"
    fig.write_html(file_name)
    print(f" Saved interactive 3D plot to: {file_name}")
    print(f"📁 Your files are located inside this folder:\n👉 {os.getcwd()}")


    fig.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print(" SIMPLE 4-STEP PHASE SHIFTING")
    print("=" * 60)

    for pitch in [1, 2, 3]:

        phase = process_pitch(
            pitch
        )

        plot_surface(
            phase,
            pitch
        )

    print("\nDONE.")

    # Saves the file and instantly opens it in your default web browser to test it
    fig.write_html("optical_3d_surface.html", auto_open=True)
