# Quantum Image Interpolation using FRQI, NEQR and MCQI for Grayscale and Color Images

> **Conference Research Project**
>
> Implementation of **Quantum Image Interpolation** using **Novel Enhanced Quantum Representation (NEQR)**, **Flexible Representation of Quantum Images (FRQI)**, and **Multi-Channel Quantum Image (MCQI)** for grayscale and color image scaling.

---

## 📖 Overview

Quantum Image Processing (QIP) is an emerging field that combines quantum computing with digital image processing to perform image operations efficiently using quantum circuits.

This project presents a quantum image interpolation framework capable of performing **multi-directional image scaling** using:

- **NEQR (Novel Enhanced Quantum Representation)**
- **FRQI (Flexible Representation of Quantum Images)**
- **MCQI (Multi-Channel Quantum Image Representation)**

The proposed implementation performs image interpolation by manipulating **quantum position qubits**, producing different spatial transformations such as:

- Zooming
- Horizontal Stretching
- Vertical Stretching
- Image Replication

The implementation has been developed using **Python** and **Qiskit**.

---

# 📑 Abstract

Classical image interpolation methods require increasing computational resources as image resolution grows. Quantum Image Processing provides an alternative framework by encoding image information into quantum states.

This project proposes an efficient quantum image interpolation method using three quantum image representation models:

- NEQR
- FRQI
- MCQI

The methodology enlarges low-resolution grayscale and color images by introducing additional position qubits and rearranging their positions through quantum gates. Experimental results demonstrate successful generation of zoomed, vertically stretched, horizontally stretched and replicated images while preserving image quality.

---

# ✨ Features

- Quantum Image Representation
- Image Upscaling (2×)
- Quantum Circuit Generation
- Multi-directional Image Scaling
- Image Replication
- Horizontal Stretching
- Vertical Stretching
- Zoomed Image Generation
- Support for Grayscale Images
- Support for Color Images
- Implemented using Qiskit
- Research Paper Included

---

# 🧠 Quantum Image Models

## 1. NEQR (Novel Enhanced Quantum Representation)

NEQR represents grayscale pixel values directly using basis states.

### Advantages

- Exact grayscale representation
- Simple measurement
- Suitable for image processing operations

Outputs

- Zoomed Image
- Horizontal Stretch
- Vertical Stretch
- Image Replication

---

## 2. FRQI (Flexible Representation of Quantum Images)

FRQI represents grayscale values using rotation angles.

Advantages

- Compact representation
- Efficient quantum storage
- Suitable for quantum image transformations

Outputs

- Zoomed Image
- Horizontal Stretch
- Vertical Stretch
- Image Replication

---

## 3. MCQI (Multi-Channel Quantum Image)

MCQI extends FRQI for RGB images.

Advantages

- Supports color images
- Separate R, G and B channels
- Efficient color image processing

Outputs

- Zoomed Image
- Horizontal Stretch
- Vertical Stretch
- Image Replication

---

# 📂 Repository Structure

```
quantum-image-interpolation-frqi-neqr-mcqi

│
├── 3_Programs
│   ├── 1_NEQR
│   ├── 2_FRQI
│   └── 3_MCQI
│
├── Conference Paper (.pdf/.doc)
│
├── Presentation (.pptx)
│
├── Input Image
│
├── README.md
│
└── Additional Documentation
```

---

# ⚙️ Requirements

- Python 3.10+
- Qiskit
- NumPy
- Pillow
- Matplotlib

Install the dependencies:

```bash
pip install qiskit numpy matplotlib pillow
```

---

# ▶️ Running the Programs

## NEQR

Navigate to

```
3_Programs/1_NEQR/
```

Run

```bash
python NEQR.py
```

---

## FRQI

Navigate to

```
3_Programs/2_FRQI/
```

Run

```bash
python FRQI.py
```

---

## MCQI

Navigate to

```
3_Programs/3_MCQI/
```

Run

```bash
python MCQI.py
```

---

# 📷 Input Image

The project uses a **64 × 64 Lena grayscale image** as the input for experimentation.

Example:

```
Original Image
        ↓
Quantum Encoding
        ↓
Quantum Interpolation
        ↓
Output Images
```

---

# 🔬 Methodology

The proposed framework follows these stages:

1. Read input image.
2. Encode image using NEQR, FRQI or MCQI.
3. Apply Hadamard gates to position qubits.
4. Introduce additional position qubits.
5. Rearrange quantum position qubits.
6. Generate enlarged image.
7. Produce:

- Zoomed Image
- Horizontal Stretch
- Vertical Stretch
- Image Replication

---

# 📊 Results

The implementation successfully generates:

✅ Zoomed Image

✅ Horizontal Stretch

✅ Vertical Stretch

✅ Image Replication

for all three quantum image models.

---

# 📸 Sample Outputs

The repository contains generated outputs including:

- Original Image
- Quantum Circuit
- Zoomed Image
- Horizontal Stretch
- Vertical Stretch
- Image Replication
- Comparison Image

---

# ⚛️ Quantum Circuits

Quantum circuit diagrams are generated for

- NEQR
- FRQI
- MCQI

using Qiskit.

---

# 📄 Conference Paper

The complete conference paper is included in this repository.

**Title**

**Image Interpolation for Multi-Directional Scaling using FRQI & NEQR for Gray Images and MCQI for Color Images**

---

# 🛠 Technologies Used

- Python
- Qiskit
- Quantum Computing
- Quantum Image Processing
- NumPy
- Pillow
- Matplotlib

---

# 📚 Applications

- Quantum Image Processing
- Image Enlargement
- Quantum Computer Vision
- Medical Imaging
- Multimedia Processing
- Image Transformation
- Quantum Machine Learning

---

# 👨‍💻 Authors

- **U.S.N. Raju**
- **Makkena Veda Sri**
- **Rudrarapu Akshitha**
- **Shaik Farooq Afrooz**
- **Chakravaram Naga Hrithikesh**

---

# 📧 Contact

**Shaik Farooq Afrooz**

B.Tech Computer Science & Engineering (AI & ML)

JK Lakshmipat University, Jaipur, India

GitHub: https://github.com/farooq247

---

## 🙏 Acknowledgement

The authors express their sincere gratitude to **National Institute of Technology Warangal (NIT Warangal)** for providing the research opportunity and guidance that supported this work in Quantum Image Processing.

---

## 🌟 Repository Status

**Actively Maintained**

Research Area:

- Quantum Computing
- Quantum Image Processing
- Quantum Algorithms
- Image Interpolation
- Quantum Computer Vision
