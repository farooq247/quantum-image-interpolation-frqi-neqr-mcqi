# FRQI_Image_Upscaling.py
# Educational FRQI (Flexible Representation of Quantum Images) Upscaling
# Qiskit 1.4.5 Compatible
#
# Works with ANY size TIFF/PNG/JPG grayscale image (64x64, 128x128, ...)
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
# LOAD IMAGE AS GRAYSCALE  -- works for ANY size
# ==========================================================

def load_image_as_grayscale(path):
    """
    Loads an image of ANY size and returns it as a 2D uint8
    grayscale array (FRQI encodes a single intensity channel).
    """

    if HAVE_TIFFFILE and path.lower().endswith((".tif", ".tiff")):
        img = tifffile.imread(path)
    else:
        from PIL import Image as PILImage
        img = np.array(PILImage.open(path))

    img = np.array(img)

    # RGB(A) -> grayscale by averaging channels
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)

    # Normalize to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        img -= img.min()
        if img.max() > 0:
            img = img / img.max() * 255.0
        img = img.astype(np.uint8)

    return img.astype(np.uint8)


# ==========================================================
# PAD TO NEXT POWER OF 2  -- works for ANY size
# ==========================================================

def pad_power2(img):
    """
    Pads a 2D grayscale image (any size) up to the next
    power-of-2 dimensions, so FRQI position qubits map cleanly.
    """

    h, w = img.shape

    hp = 2 ** int(np.ceil(np.log2(h))) if h > 1 else 1
    wp = 2 ** int(np.ceil(np.log2(w))) if w > 1 else 1

    hp = max(hp, 1)
    wp = max(wp, 1)

    out = np.zeros((hp, wp), dtype=np.uint8)
    out[:h, :w] = img

    return out


# ==========================================================
# FRQI UPSCALING  -- works for ANY input size and ANY scale
# ==========================================================

def frqi_upscale(gray_img, scale=2, save_circuit=True, max_circuit_size=8):
    """
    Works for ANY input image size (e.g. 2x2, 64x64, 128x128, 256x256, ...)
    and ANY power-of-2 scale factor (1, 2, 4, 8, ...).

    FRQI encoding:
        - One color qubit "C" stores pixel intensity as an angle:
              |pixel> = cos(theta)|0> + sin(theta)|1>,  theta = (I/255)*(pi/2)
        - "row"/"col" position qubits are put into superposition
          (Hadamard) to represent all pixel coordinates simultaneously.
        - Controlled-RY gates (controlled on row/col basis states)
          rotate the color qubit by the angle theta for that pixel.

    max_circuit_size : maximum H and W (in pixels) used when BUILDING
                        the quantum circuit diagram. If the input image
                        is larger than this, it is downsampled ONLY for
                        circuit construction/visualization.
                        The classical upscaling below ALWAYS uses the
                        full-resolution image, regardless of size.
    """

    gray_img = np.array(gray_img, dtype=np.uint8)

    H, W = gray_img.shape

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
    # ------------------------------------------------------

    circuit_img = gray_img

    if H > max_circuit_size or W > max_circuit_size:

        step_h = max(1, math.ceil(H / max_circuit_size))
        step_w = max(1, math.ceil(W / max_circuit_size))

        circuit_img = gray_img[::step_h, ::step_w]

        print("=" * 60)
        print("NOTE: Image too large for full circuit construction.")
        print(f"Original size : {H} x {W}")
        print(f"Circuit built using downsampled size : "
              f"{circuit_img.shape[0]} x {circuit_img.shape[1]}")
        print("=" * 60)

    cH, cW = circuit_img.shape

    row_bits = max(1, math.ceil(math.log2(cH)))
    col_bits = max(1, math.ceil(math.log2(cW)))

    er_bits = int(math.log2(scale))
    ec_bits = int(math.log2(scale))

    # ------------------------------------------------------
    # REGISTERS
    # ------------------------------------------------------

    C = QuantumRegister(1, "C")  # single FRQI color/intensity qubit

    row = QuantumRegister(row_bits, "row")
    col = QuantumRegister(col_bits, "col")

    regs = [C]

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
    print("STEP-1 : FRQI REGISTERS")
    print("=" * 60)

    print("Color/Intensity Qubit : C")
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
    # FRQI INTENSITY ENCODING (uses circuit_img, NOT full image)
    # ------------------------------------------------------

    for r in range(cH):

        for c in range(cW):

            intensity = int(circuit_img[r, c])

            theta = (intensity / 255.0) * (np.pi / 2)

            r_bin = format(r, f"0{row_bits}b")
            c_bin = format(c, f"0{col_bits}b")

            # Activate controls

            for i, bit in enumerate(reversed(r_bin)):
                if bit == "0":
                    qc.x(row[i])

            for i, bit in enumerate(reversed(c_bin)):
                if bit == "0":
                    qc.x(col[i])

            # Controlled RY for intensity qubit

            cry = RYGate(2 * theta).control(len(controls))
            qc.append(
                cry,
                controls + [C[0]]
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
    print("STEP-3 : FRQI ENCODING COMPLETE")
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
                "FRQI_Circuit.png"
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
        np.repeat(gray_img, scale, axis=0),
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
# SAVE IMAGE AS TIFF (grayscale)  -- works for any size
# ==========================================================

def save_image(path, img):
    """
    Saves the given 2D grayscale array (any size) as a TIFF file (.tif).
    Uses tifffile if available, otherwise falls back to PIL.
    """

    img = img.astype(np.uint8)

    if HAVE_TIFFFILE:
        tifffile.imwrite(path, img)
    else:
        Image.fromarray(img, mode="L").save(path, format="TIFF")


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

    gray_img = load_image_as_grayscale(input_path)
    gray_img = pad_power2(gray_img)

    print("\nLoaded image:")
    print(input_path)
    print("Shape (after power-of-2 padding):", gray_img.shape)

    # --------------------------------------------------
    # All operations below use SCALE (user-defined above)
    # and gray_img's ACTUAL size (any HxW) automatically.
    # --------------------------------------------------

    qc, upscaled = frqi_upscale(
        gray_img,
        scale=SCALE,
        max_circuit_size=MAX_CIRCUIT_SIZE
    )

    vertical_img = vertical_stretch(
        gray_img,
        scale=SCALE
    )

    horizontal_img = horizontal_stretch(
        gray_img,
        scale=SCALE
    )

    replicated_img = image_replication(
        gray_img
    )

    # ------------------------------------------------------
    # OUTPUT PATHS (same folder as script)
    # ------------------------------------------------------

    original_path = os.path.join(base_dir, "FRQI_Original.tif")
    upscaled_path = os.path.join(base_dir, "FRQI_Upscaled.tif")
    vertical_path = os.path.join(base_dir, "FRQI_VerticalStretch.tif")
    horizontal_path = os.path.join(base_dir, "FRQI_HorizontalStretch.tif")
    replication_path = os.path.join(base_dir, "FRQI_Replication.tif")
    comparison_path = os.path.join(base_dir, "FRQI_Comparison.tif")
    all_results_path = os.path.join(base_dir, "FRQI_AllResults.png")

    # ------------------------------------------------------
    # SAVE ALL OUTPUTS AS TIFF
    # ------------------------------------------------------

    save_image(original_path, gray_img)
    save_image(upscaled_path, upscaled)
    save_image(vertical_path, vertical_img)
    save_image(horizontal_path, horizontal_img)
    save_image(replication_path, replicated_img)

    # ------------------------------------------------------
    # DISPLAY ALL RESULTS (Original + 4 transformations)
    # ------------------------------------------------------

    fig, ax = plt.subplots(1, 5, figsize=(22, 5))

    ax[0].imshow(gray_img, cmap="gray")
    ax[0].set_title(f"Original\n{gray_img.shape[0]}x{gray_img.shape[1]}")
    ax[0].axis("off")

    ax[1].imshow(upscaled, cmap="gray")
    ax[1].set_title(
        f"FRQI Upscaled (x{SCALE})\n{upscaled.shape[0]}x{upscaled.shape[1]}"
    )
    ax[1].axis("off")

    ax[2].imshow(vertical_img, cmap="gray")
    ax[2].set_title(
        f"Vertical Stretch (x{SCALE})\n{vertical_img.shape[0]}x{vertical_img.shape[1]}"
    )
    ax[2].axis("off")

    ax[3].imshow(horizontal_img, cmap="gray")
    ax[3].set_title(
        f"Horizontal Stretch (x{SCALE})\n{horizontal_img.shape[0]}x{horizontal_img.shape[1]}"
    )
    ax[3].axis("off")

    ax[4].imshow(replicated_img, cmap="gray")
    ax[4].set_title(
        f"Replication\n{replicated_img.shape[0]}x{replicated_img.shape[1]}"
    )
    ax[4].axis("off")

    plt.tight_layout()

    # Save the combined figure as PNG (for viewing) ...
    plt.savefig(all_results_path, dpi=200, bbox_inches="tight")

    # ... and ALSO save a 2-panel comparison TIFF (Original vs Upscaled)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    ax2[0].imshow(gray_img, cmap="gray")
    ax2[0].set_title("Original")
    ax2[0].axis("off")
    ax2[1].imshow(upscaled, cmap="gray")
    ax2[1].set_title(f"FRQI Upscaled (x{SCALE})")
    ax2[1].axis("off")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, format="tiff")
    plt.close(fig2)

    # Now SHOW the all-results figure on screen
    plt.figure(fig.number)
    plt.show()

    print("\nFinal Shapes:")
    print("Original :", gray_img.shape)
    print(f"Upscaled (x{SCALE}) :", upscaled.shape)
    print(f"Vertical Stretch (x{SCALE}) :", vertical_img.shape)
    print(f"Horizontal Stretch (x{SCALE}) :", horizontal_img.shape)
    print("Replication :", replicated_img.shape)

    print("\nAll files saved in folder:")
    print(base_dir)

    print("\nSaved Files:")
    print("FRQI_Circuit.png")
    print("FRQI_Original.tif")
    print("FRQI_Upscaled.tif")
    print("FRQI_VerticalStretch.tif")
    print("FRQI_HorizontalStretch.tif")
    print("FRQI_Replication.tif")
    print("FRQI_Comparison.tif")
    print("FRQI_AllResults.png")