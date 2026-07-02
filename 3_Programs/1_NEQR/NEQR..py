# NEQR_Image_Upscaling.py
# Educational NEQR (Novel Enhanced Quantum Representation) Upscaling
# Qiskit 1.4.5 Compatible
#
# Works with ANY size TIFF/PNG/JPG grayscale image (64x64, 128x128, ...)
# User sets the upscale factor (SCALE) below.
#
# NEQR encoding:
#   - Multiple gray-value qubits g0,g1,...,gN store binary gray value
#     using MCX (multi-controlled X) gates.
#   - row/col position qubits put into superposition via Hadamard.
#   - e_row/e_col expansion qubits handle upscaling.
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
# How much the image should be zoomed/upscaled.
# Must be a power of 2 (1, 2, 4, 8, ...).
SCALE = 2

# Max H/W used when BUILDING the quantum circuit (for visualization only).
# Kept at 4 for NEQR since MCX gates grow much faster than RY gates.
MAX_CIRCUIT_SIZE = 4

# Whether to draw and save step-by-step circuit diagrams
DRAW_CIRCUITS = True


# ==========================================================
# LOAD IMAGE AS GRAYSCALE  -- works for ANY size
# ==========================================================

def load_image_as_grayscale(path):
    """
    Loads a TIFF/PNG/JPG image of ANY size and returns it
    as a 2D uint8 grayscale array.
    """

    if HAVE_TIFFFILE and path.lower().endswith((".tif", ".tiff")):
        img = tifffile.imread(path)
    else:
        from PIL import Image as PILImage
        img = np.array(PILImage.open(path))

    img = np.array(img)

    # RGB(A) -> grayscale
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)

    # Normalize to uint8
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
    power-of-2 dimensions so NEQR position qubits map cleanly.
    """

    h, w = img.shape

    hp = 2 ** int(np.ceil(np.log2(h))) if h > 1 else 1
    wp = 2 ** int(np.ceil(np.log2(w))) if w > 1 else 1

    out = np.zeros((hp, wp), dtype=np.uint8)
    out[:h, :w] = img

    return out


# ==========================================================
# NEQR UPSCALING  -- works for ANY input size and ANY scale
# ==========================================================

def neqr_upscale(gray_img, scale=2, save_circuit=True,
                 max_circuit_size=4, draw_circuits=True):
    """
    Works for ANY input image size (64x64, 128x128, ...) and
    ANY power-of-2 scale factor (1, 2, 4, 8, ...).

    NEQR encoding:
      - num_color_qubits = ceil(log2(max_gray_val + 1)) binary qubits
        for the gray value (g0, g1, ...).
      - row/col position qubits go into superposition (Hadamard).
      - MCX gates controlled on row/col basis states flip the
        corresponding color qubits to encode that pixel's binary value.
      - e_row/e_col expansion qubits + Hadamard handle upscaling.
      - Circuit is downsampled for visualization only;
        classical upscaling runs on the full-resolution image.
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
    # (works for ANY size: 64x64, 128x128, 256x256, ...)
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

    num_row_qubits = max(1, math.ceil(math.log2(cH)))
    num_col_qubits = max(1, math.ceil(math.log2(cW)))

    scale_row_qubits = int(math.log2(scale))
    scale_col_qubits = int(math.log2(scale))

    max_val = int(np.max(circuit_img))
    num_color_qubits = max(1, math.ceil(math.log2(max_val + 1))) if max_val > 0 else 1

    # ------------------------------------------------------
    # STEP-1 : REGISTERS
    # ------------------------------------------------------

    e_col_reg = QuantumRegister(scale_col_qubits, name="e_col") if scale_col_qubits > 0 else None
    col_reg   = QuantumRegister(num_col_qubits, name="col")
    e_row_reg = QuantumRegister(scale_row_qubits, name="e_row") if scale_row_qubits > 0 else None
    row_reg   = QuantumRegister(num_row_qubits, name="row")
    color_reg = QuantumRegister(num_color_qubits, name="g")

    regs = []
    if e_col_reg: regs.append(e_col_reg)
    regs.append(col_reg)
    if e_row_reg: regs.append(e_row_reg)
    regs.append(row_reg)
    regs.append(color_reg)

    qc = QuantumCircuit(*regs)

    print("=" * 60)
    print("STEP-1 : NEQR REGISTERS")
    print("=" * 60)
    print("Gray Qubits   : g0 ... gN  (binary-encoded gray value)")
    print("Position Qubits : row, col")
    print(f"num_color_qubits = {num_color_qubits}")
    print(f"row_bits = {num_row_qubits}, col_bits = {num_col_qubits}")
    print(f"e_row_bits = {scale_row_qubits}, e_col_bits = {scale_col_qubits}")

    if draw_circuits:
        qc_step1 = QuantumCircuit(col_reg, row_reg, color_reg)
        print("\nStep-1 Circuit (initial registers):")
        print(qc_step1.draw(output="text", idle_wires=True, fold=120))
        try:
            p = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NEQR_Step1_Circuit.png"
            )
            qc_step1.draw(output="mpl", idle_wires=True).savefig(p, bbox_inches="tight")
            plt.close()
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Step-1 save failed: {e}")

    qc.barrier()

    # ------------------------------------------------------
    # STEP-2 : HADAMARD ON ROW/COL + NEQR ORACLE (MCX)
    # ------------------------------------------------------

    for i in range(num_col_qubits):
        qc.h(col_reg[i])

    for i in range(num_row_qubits):
        qc.h(row_reg[i])

    qc.barrier()

    control_qubits = (
        [row_reg[i] for i in range(num_row_qubits)]
        + [col_reg[i] for i in range(num_col_qubits)]
    )

    for r in range(cH):

        for c in range(cW):

            val = int(circuit_img[r, c])

            if val == 0:
                continue

            r_bin = format(r, f"0{num_row_qubits}b")
            c_bin = format(c, f"0{num_col_qubits}b")

            # Activate controls
            for i, bit in enumerate(reversed(r_bin)):
                if bit == "0":
                    qc.x(row_reg[i])
            for i, bit in enumerate(reversed(c_bin)):
                if bit == "0":
                    qc.x(col_reg[i])

            # MCX for each '1' bit in the binary gray value
            v_bin = format(val, f"0{num_color_qubits}b")
            for i, bit in enumerate(reversed(v_bin)):
                if bit == "1":
                    qc.mcx(control_qubits, color_reg[i])

            # Restore controls
            for i, bit in enumerate(reversed(r_bin)):
                if bit == "0":
                    qc.x(row_reg[i])
            for i, bit in enumerate(reversed(c_bin)):
                if bit == "0":
                    qc.x(col_reg[i])

            qc.barrier()

    qc.barrier()

    print("=" * 60)
    print("STEP-2 : HADAMARD + NEQR ORACLE (MCX ENCODING) COMPLETE")
    print("=" * 60)

    if draw_circuits:
        print(qc.draw(output="text", idle_wires=False, fold=120))
        try:
            p = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NEQR_Step2_Circuit.png"
            )
            qc.draw(output="mpl", idle_wires=False).savefig(p, bbox_inches="tight")
            plt.close()
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Step-2 save failed: {e}")

    # ------------------------------------------------------
    # STEP-3 : e_row / e_col EXPANSION QUBITS
    # ------------------------------------------------------

    print("=" * 60)
    print("STEP-3 : e_row / e_col EXPANSION QUBITS ADDED")
    print("=" * 60)

    if draw_circuits:
        print(qc.draw(output="text", idle_wires=True, fold=120))
        try:
            p = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NEQR_Step3_Circuit.png"
            )
            qc.draw(output="mpl", idle_wires=True).savefig(p, bbox_inches="tight")
            plt.close()
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Step-3 save failed: {e}")

    # ------------------------------------------------------
    # STEP-4 : HADAMARD ON e_row / e_col
    # ------------------------------------------------------

    if scale_col_qubits > 0:
        for i in range(scale_col_qubits):
            qc.h(e_col_reg[i])

    if scale_row_qubits > 0:
        for i in range(scale_row_qubits):
            qc.h(e_row_reg[i])

    qc.barrier()

    print("=" * 60)
    print("STEP-4 : HADAMARD ON e_row / e_col COMPLETE")
    print("=" * 60)

    if draw_circuits:
        print(qc.draw(output="text", idle_wires=True, fold=120))
        try:
            p = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NEQR_Step4_Circuit.png"
            )
            qc.draw(output="mpl", idle_wires=True).savefig(p, bbox_inches="tight")
            plt.close()
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Step-4 save failed: {e}")

    # ------------------------------------------------------
    # SAVE FULL CIRCUIT
    # ------------------------------------------------------

    if save_circuit:
        try:
            circuit_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "NEQR_Circuit.png"
            )
            fig = qc.draw(output="mpl", fold=120)
            fig.savefig(circuit_path, bbox_inches="tight")
            plt.close(fig)
            print("\nFull circuit saved:")
            print(circuit_path)
        except Exception as e:
            print("\nFull circuit save failed:", e)

    # ------------------------------------------------------
    # STEP-5 : UPSCALING (full-resolution, any size)
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
        from PIL import Image as PILImage
        PILImage.fromarray(img, mode="L").save(path, format="TIFF")


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

    qc, upscaled = neqr_upscale(
        gray_img,
        scale=SCALE,
        save_circuit=True,
        max_circuit_size=MAX_CIRCUIT_SIZE,
        draw_circuits=DRAW_CIRCUITS
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

    original_path    = os.path.join(base_dir, "NEQR_Original.tif")
    upscaled_path    = os.path.join(base_dir, "NEQR_Upscaled.tif")
    vertical_path    = os.path.join(base_dir, "NEQR_VerticalStretch.tif")
    horizontal_path  = os.path.join(base_dir, "NEQR_HorizontalStretch.tif")
    replication_path = os.path.join(base_dir, "NEQR_Replication.tif")
    comparison_path  = os.path.join(base_dir, "NEQR_Comparison.tif")
    all_results_path = os.path.join(base_dir, "NEQR_AllResults.png")

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
        f"NEQR Upscaled (x{SCALE})\n{upscaled.shape[0]}x{upscaled.shape[1]}"
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

    # Save the combined figure as PNG (for viewing)
    plt.savefig(all_results_path, dpi=200, bbox_inches="tight")

    # Save a 2-panel comparison TIFF (Original vs Upscaled)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    ax2[0].imshow(gray_img, cmap="gray")
    ax2[0].set_title("Original")
    ax2[0].axis("off")
    ax2[1].imshow(upscaled, cmap="gray")
    ax2[1].set_title(f"NEQR Upscaled (x{SCALE})")
    ax2[1].axis("off")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, format="tiff")
    plt.close(fig2)

    # Show all-results figure on screen
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
    print("NEQR_Circuit.png")
    print("NEQR_Step1_Circuit.png")
    print("NEQR_Step2_Circuit.png")
    print("NEQR_Step3_Circuit.png")
    print("NEQR_Step4_Circuit.png")
    print("NEQR_Original.tif")
    print("NEQR_Upscaled.tif")
    print("NEQR_VerticalStretch.tif")
    print("NEQR_HorizontalStretch.tif")
    print("NEQR_Replication.tif")
    print("NEQR_Comparison.tif")
    print("NEQR_AllResults.png")