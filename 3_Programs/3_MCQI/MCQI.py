# MCQI_Image_Upscaling.py
# Educational MCQI (Multi-Channel Quantum Image) Upscaling
# Qiskit 1.4.5 Compatible
#
# Works with ANY size TIFF image (2x2, 64x64, 128x128, 256x256, ...)
# User sets the upscale factor (SCALE) below.
#
# Loads input image from the same folder as this script.
# Saves circuit + ALL output images (as .tif) in the same folder.
# AND displays all output images visually in one figure.

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate

try:
    import tifffile
    HAVE_TIFFFILE = True
except ImportError:
    HAVE_TIFFFILE = False
    from PIL import Image


# ==========================================================
# CONFIG  (CHANGE THESE AS NEEDED)
# ==========================================================

# Name of the input image file (must be in the same folder as this script)
# Works for any size: 2x2, 64x64, 128x128, 256x256, etc.
INPUT_IMAGE_NAME = "Original_1_Lenna_64_grey.tif"

# >>> USER-DEFINED UPSCALE FACTOR <<<
# How much the image should be "zoomed"/upscaled.
# Must be a power of 2 (1, 2, 4, 8, ...) because of the e_row/e_col
# expansion qubit logic (log2(scale) must be a whole number).
SCALE = 2

# Max H/W used when BUILDING the quantum circuit (for visualization only).
# Larger images are downsampled ONLY for the circuit diagram so it
# doesn't become unreadable / unbuildable for big images.
MAX_CIRCUIT_SIZE = 8


# ==========================================================
# LOAD IMAGE (TIFF) AS RGB ARRAY  -- works for ANY size
# ==========================================================

def load_image_as_rgb(path):
    """
    Loads a TIFF image of ANY size (grayscale or RGB) and returns it
    as an (H, W, 3) uint8 RGB array. Works for 64x64, 128x128, etc.
    """

    if HAVE_TIFFFILE:
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))

    img = np.array(img)

    # Ensure uint8
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        img -= img.min()
        if img.max() > 0:
            img = img / img.max() * 255.0
        img = img.astype(np.uint8)

    # Grayscale -> RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # Drop alpha channel if present
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]

    return img.astype(np.uint8)


# ==========================================================
# MCQI UPSCALING  -- works for ANY input size and ANY scale
# ==========================================================

def mcqi_upscale(rgb_img, scale=2, save_circuit=True, max_circuit_size=8):
    """
    Works for ANY input image size (e.g. 2x2, 64x64, 128x128, 256x256, ...)
    and ANY power-of-2 scale factor (1, 2, 4, 8, ...).

    max_circuit_size : maximum H and W (in pixels) used when BUILDING
                        the quantum circuit diagram. If the input image
                        is larger than this, it is downsampled ONLY for
                        circuit construction/visualization.
                        The classical upscaling below ALWAYS uses the
                        full-resolution image, regardless of size.
    """

    rgb_img = np.array(rgb_img, dtype=np.uint8)

    H, W, _ = rgb_img.shape

    if scale < 1 or (scale & (scale - 1)) != 0:
        raise ValueError(
            f"SCALE must be a power of 2 (1, 2, 4, 8, ...). Got: {scale}"
        )

    target_H = H * scale
    target_W = W * scale

    print("=" * 60)
    print("INPUT IMAGE SIZE")
    print("=" * 60)
    print(f"H x W = {H} x {W}")
    print(f"Upscale factor = {scale}")
    print(f"Target size after upscaling = {target_H} x {target_W}")

    # ------------------------------------------------------
    # Downsample for circuit construction if image is too large
    # (works for ANY size: 64x64, 128x128, 256x256, ...)
    # ------------------------------------------------------

    circuit_img = rgb_img

    if H > max_circuit_size or W > max_circuit_size:

        step_h = max(1, math.ceil(H / max_circuit_size))
        step_w = max(1, math.ceil(W / max_circuit_size))

        circuit_img = rgb_img[::step_h, ::step_w, :]

        print("=" * 60)
        print("NOTE: Image too large for full circuit construction.")
        print(f"Original size : {H} x {W}")
        print(f"Circuit built using downsampled size : "
              f"{circuit_img.shape[0]} x {circuit_img.shape[1]}")
        print("=" * 60)

    cH, cW, _ = circuit_img.shape

    row_bits = max(1, math.ceil(math.log2(cH)))
    col_bits = max(1, math.ceil(math.log2(cW)))

    er_bits = int(math.log2(scale))
    ec_bits = int(math.log2(scale))

    # ------------------------------------------------------
    # REGISTERS
    # ------------------------------------------------------

    R = QuantumRegister(1, "R")
    G = QuantumRegister(1, "G")
    B = QuantumRegister(1, "B")

    row = QuantumRegister(row_bits, "row")
    col = QuantumRegister(col_bits, "col")

    regs = [R, G, B]

    e_col = None
    if ec_bits:
        e_col = QuantumRegister(ec_bits, "e_col")
        regs.append(e_col)

    regs.append(col)

    e_row = None
    if er_bits:
        e_row = QuantumRegister(er_bits, "e_row")
        regs.append(e_row)

    regs.append(row)

    qc = QuantumCircuit(*regs)

    print("=" * 60)
    print("STEP-1 : MCQI REGISTERS")
    print("=" * 60)

    print("Color Qubits : R, G, B")
    print("Position Qubits : row, col")
    print(f"row_bits = {row_bits}, col_bits = {col_bits}")
    print(f"e_row_bits = {er_bits}, e_col_bits = {ec_bits}")

    # ------------------------------------------------------
    # STEP-2
    # HADAMARD ON POSITION QUBITS
    # ------------------------------------------------------

    for q in row:
        qc.h(q)

    for q in col:
        qc.h(q)

    qc.barrier()

    print("=" * 60)
    print("STEP-2 : POSITION SUPERPOSITION")
    print("=" * 60)

    controls = list(row) + list(col)

    # ------------------------------------------------------
    # RGB ENCODING (uses circuit_img, NOT the full image)
    # ------------------------------------------------------

    for r in range(cH):

        for c in range(cW):

            red = int(circuit_img[r, c, 0])
            green = int(circuit_img[r, c, 1])
            blue = int(circuit_img[r, c, 2])

            theta_r = (red / 255.0) * (np.pi / 2)
            theta_g = (green / 255.0) * (np.pi / 2)
            theta_b = (blue / 255.0) * (np.pi / 2)

            r_bin = format(r, f"0{row_bits}b")
            c_bin = format(c, f"0{col_bits}b")

            # Activate controls

            for i, bit in enumerate(reversed(r_bin)):
                if bit == "0":
                    qc.x(row[i])

            for i, bit in enumerate(reversed(c_bin)):
                if bit == "0":
                    qc.x(col[i])

            # Controlled RY for R channel

            cry_r = RYGate(2 * theta_r).control(len(controls))
            qc.append(
                cry_r,
                controls + [R[0]]
            )

            # Controlled RY for G channel

            cry_g = RYGate(2 * theta_g).control(len(controls))
            qc.append(
                cry_g,
                controls + [G[0]]
            )

            # Controlled RY for B channel

            cry_b = RYGate(2 * theta_b).control(len(controls))
            qc.append(
                cry_b,
                controls + [B[0]]
            )

            # Restore controls

            for i, bit in enumerate(reversed(r_bin)):
                if bit == "0":
                    qc.x(row[i])

            for i, bit in enumerate(reversed(c_bin)):
                if bit == "0":
                    qc.x(col[i])

    qc.barrier()

    print("=" * 60)
    print("STEP-3 : MCQI ENCODING COMPLETE")
    print("=" * 60)

    # ------------------------------------------------------
    # STEP-4
    # EXPANSION QUBITS
    # ------------------------------------------------------

    if e_row:
        for q in e_row:
            qc.h(q)

    if e_col:
        for q in e_col:
            qc.h(q)

    qc.barrier()

    print("=" * 60)
    print("STEP-4 : e_row / e_col EXPANSION")
    print("=" * 60)

    # ------------------------------------------------------
    # SAVE CIRCUIT
    # ------------------------------------------------------

    if save_circuit:

        try:

            circuit_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "MCQI_Circuit.png"
            )

            fig = qc.draw(
                output="mpl",
                fold=120
            )

            fig.savefig(
                circuit_path,
                bbox_inches="tight"
            )

            plt.close(fig)

            print("\nCircuit saved:")
            print(circuit_path)

        except Exception as e:

            print("\nCircuit save failed")
            print(e)

    # ------------------------------------------------------
    # STEP-5
    # UPSCALING (uses the FULL-RESOLUTION image, any size)
    # "Zoomed image after upscaling" — every pixel repeated
    # `scale` times along both axes.
    # ------------------------------------------------------

    out = np.repeat(
        np.repeat(rgb_img, scale, axis=0),
        scale,
        axis=1
    )

    print("=" * 60)
    print("STEP-5 : UPSCALING COMPLETE")
    print("=" * 60)
    print(f"Output size = {out.shape[0]} x {out.shape[1]}")

    return qc, out


# ==========================================================
# VERTICAL STRETCHING  -- works for any size, any scale
# ==========================================================

def vertical_stretch(img, scale=2):

    return np.repeat(
        img,
        scale,
        axis=0
    )


# ==========================================================
# HORIZONTAL STRETCHING  -- works for any size, any scale
# ==========================================================

def horizontal_stretch(img, scale=2):

    return np.repeat(
        img,
        scale,
        axis=1
    )


# ==========================================================
# IMAGE REPLICATION  -- works for any size
# ==========================================================

def image_replication(img):

    top = np.concatenate(
        (img, img),
        axis=1
    )

    bottom = np.concatenate(
        (img, img),
        axis=1
    )

    return np.concatenate(
        (top, bottom),
        axis=0
    )


# ==========================================================
# SAVE IMAGE AS TIFF  -- works for any size
# ==========================================================

def save_image(path, img):
    """
    Saves the given image array (any size) as a TIFF file (.tif).
    Uses tifffile if available, otherwise falls back to PIL.
    """

    img = img.astype(np.uint8)

    if HAVE_TIFFFILE:
        tifffile.imwrite(path, img)
    else:
        Image.fromarray(img).save(path, format="TIFF")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # ------------------------------------------------------
    # LOAD INPUT IMAGE FROM SAME FOLDER
    # Works for 2x2, 64x64, 128x128, 256x256, or any other size
    # ------------------------------------------------------

    input_path = os.path.join(
        base_dir,
        INPUT_IMAGE_NAME
    )

    rgb_img = load_image_as_rgb(input_path)

    print("\nLoaded image:")
    print(input_path)
    print("Shape:", rgb_img.shape)

    # --------------------------------------------------
    # All operations below use SCALE (user-defined above)
    # and rgb_img's ACTUAL size (any HxW) automatically.
    # --------------------------------------------------

    qc, upscaled = mcqi_upscale(
        rgb_img,
        scale=SCALE,
        max_circuit_size=MAX_CIRCUIT_SIZE
    )

    vertical_img = vertical_stretch(
        rgb_img,
        scale=SCALE
    )

    horizontal_img = horizontal_stretch(
        rgb_img,
        scale=SCALE
    )

    replicated_img = image_replication(
        rgb_img
    )

    # ------------------------------------------------------
    # OUTPUT PATHS (same folder as script)
    # ------------------------------------------------------

    original_path = os.path.join(base_dir, "MCQI_Original.tif")
    upscaled_path = os.path.join(base_dir, "MCQI_Upscaled.tif")
    vertical_path = os.path.join(base_dir, "MCQI_VerticalStretch.tif")
    horizontal_path = os.path.join(base_dir, "MCQI_HorizontalStretch.tif")
    replication_path = os.path.join(base_dir, "MCQI_Replication.tif")
    comparison_path = os.path.join(base_dir, "MCQI_Comparison.tif")
    all_results_path = os.path.join(base_dir, "MCQI_AllResults.png")

    # ------------------------------------------------------
    # SAVE ALL OUTPUTS AS TIFF
    # ------------------------------------------------------

    save_image(original_path, rgb_img)
    save_image(upscaled_path, upscaled)
    save_image(vertical_path, vertical_img)
    save_image(horizontal_path, horizontal_img)
    save_image(replication_path, replicated_img)

    # ------------------------------------------------------
    # DISPLAY ALL RESULTS (Original + 4 transformations)
    # ------------------------------------------------------

    fig, ax = plt.subplots(1, 5, figsize=(22, 5))

    ax[0].imshow(rgb_img)
    ax[0].set_title(f"Original\n{rgb_img.shape[0]}x{rgb_img.shape[1]}")
    ax[0].axis("off")

    ax[1].imshow(upscaled)
    ax[1].set_title(
        f"MCQI Upscaled (x{SCALE})\n{upscaled.shape[0]}x{upscaled.shape[1]}"
    )
    ax[1].axis("off")

    ax[2].imshow(vertical_img)
    ax[2].set_title(
        f"Vertical Stretch (x{SCALE})\n{vertical_img.shape[0]}x{vertical_img.shape[1]}"
    )
    ax[2].axis("off")

    ax[3].imshow(horizontal_img)
    ax[3].set_title(
        f"Horizontal Stretch (x{SCALE})\n{horizontal_img.shape[0]}x{horizontal_img.shape[1]}"
    )
    ax[3].axis("off")

    ax[4].imshow(replicated_img)
    ax[4].set_title(
        f"Replication\n{replicated_img.shape[0]}x{replicated_img.shape[1]}"
    )
    ax[4].axis("off")

    plt.tight_layout()

    # Save the combined figure as PNG (for viewing) ...
    plt.savefig(all_results_path, dpi=200, bbox_inches="tight")

    # ... and ALSO save a 2-panel comparison TIFF (Original vs Upscaled)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    ax2[0].imshow(rgb_img)
    ax2[0].set_title("Original")
    ax2[0].axis("off")
    ax2[1].imshow(upscaled)
    ax2[1].set_title(f"MCQI Upscaled (x{SCALE})")
    ax2[1].axis("off")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, format="tiff")
    plt.close(fig2)

    # Now SHOW the all-results figure on screen
    plt.figure(fig.number)
    plt.show()

    print("\nFinal Shapes:")
    print("Original :", rgb_img.shape)
    print(f"Upscaled (x{SCALE}) :", upscaled.shape)
    print(f"Vertical Stretch (x{SCALE}) :", vertical_img.shape)
    print(f"Horizontal Stretch (x{SCALE}) :", horizontal_img.shape)
    print("Replication :", replicated_img.shape)

    print("\nAll files saved in folder:")
    print(base_dir)

    print("\nSaved Files:")
    print("MCQI_Circuit.png")
    print("MCQI_Original.tif")
    print("MCQI_Upscaled.tif")
    print("MCQI_VerticalStretch.tif")
    print("MCQI_HorizontalStretch.tif")
    print("MCQI_Replication.tif")
    print("MCQI_Comparison.tif")
    print("MCQI_AllResults.png")