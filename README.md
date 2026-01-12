# VICEROY 2026: Hyperdimensional Computing for EW-Resilient Command Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference: VICEROY 2026](https://img.shields.io/badge/Conference-VICEROY%202026-green.svg)]()

> **Symposium Poster Companion Repository**  
> "Cognitive Resilience at the Edge: Using Hyperdimensional Computing (HDC) to Secure Autonomous Wingmen (CCA) Against Spectrum Jamming"

---

## 📋 Executive Summary

This repository contains the complete simulation code for our VICEROY 2026 symposium poster demonstrating that **Hyperdimensional Computing (HDC)** provides **graceful degradation** against electronic warfare (EW) jamming attacks, outperforming traditional deep learning approaches.

### Key Results at a Glance

| Scenario | Attack Type | HDC Accuracy | MLP Accuracy | Interpretation |
|----------|-------------|--------------|--------------|----------------|
| **A** (σ²=0) | None (clean) | 100% | 98.7% | Both work well |
| **A** (σ²=5) | Russian Broadband | **100%** | ~57% | HDC resilient |
| **B** (int=4) | US Precision | ~49% | ~22% | Both degraded |
| **B** (int=20) | US Precision (extreme) | ~22% | ~23% | **Both fail** |

---

## 🔬 Why Does HDC Achieve 100% in Scenario A?

We anticipated skepticism about the 100% accuracy. This is **not too good to be true**—it's a direct consequence of the mathematics.

### The Corrected Signal-to-Noise Analysis

**Previous versions incorrectly claimed "noise > signal." This is FALSE.**

#### Signal Energy Calculation
Each class centroid is generated as: `c ~ N(0, 3²)` per dimension

$$\mathbb{E}[\|c\|^2] = 3^2 \times 50 = 450$$

$$\mathbb{E}[\|c\|] = \sqrt{450} \approx 21.2$$

#### Noise Energy Calculation (at σ² = 5)
Noise is AWGN: `ε ~ N(0, σ²)` per dimension

$$\mathbb{E}[\|\varepsilon\|^2] = \sigma^2 \times 50 = 5 \times 50 = 250$$

#### Signal-to-Noise Ratio

$$\text{SNR} = \frac{\text{Signal Energy}}{\text{Noise Energy}} = \frac{450}{250} = 1.8$$

**The signal is STRONGER than the noise!**

### Why the MLP Fails (Despite SNR > 1)

The MLP fails NOT because noise overwhelms the signal, but because:

1. **Distribution Shift**: Noise shifts inputs away from the training distribution. The MLP's learned decision boundaries are invalid in this out-of-distribution (OOD) region.

2. **Feature-Specific Learning**: MLP neurons learn to respond to specific input features. Corrupting those features breaks the learned representations.

3. **No Implicit Normalization**: MLPs propagate the raw magnitude of noisy inputs through all layers, potentially causing saturation or numerical issues.

### Why HDC Survives

1. **Distributed Representation**: All 10,000 dimensions encode all features. There's no single point of failure.

2. **Binary Quantization (The Hardware Limiter Effect)**:
   ```
   MLP sees:  value = 500.0 → activations explode/saturate
   HDC sees:  value = 500.0 → sign(500) = +1 → normal operation
   ```
   The `sign()` function acts like a 1-bit ADC or limiter circuit in RF hardware.

3. **Prototype Averaging**: Bundling k=140 training samples improves prototype SNR by √k ≈ 12×. Training noise cancels; signal adds constructively.

4. **Scaler (AGC Equivalent)**: `StandardScaler` normalizes inputs, analogous to Automatic Gain Control in real RF systems.

---

## 📊 Honest Assessment: Strengths & Weaknesses

### ✅ Strengths of HDC

| Strength | Evidence | Mechanism |
|----------|----------|-----------|
| **Graceful degradation** | Accuracy drops smoothly, not catastrophically | Distributed representation prevents single points of failure |
| **Noise resilience** | 100% accuracy at σ²=5 (Scenario A) | SNR > 1 + prototype averaging + sign() clipping |
| **Binary robustness** | sign() clips extreme values to ±1 | Acts as hardware limiter, prevents numerical instability |
| **Simple training** | No backprop, no hyperparameter tuning | Just matrix multiplication and summation |
| **Interpretable** | Cosine similarity to prototypes | Direct geometric intuition |

### ❌ Weaknesses of HDC

| Weakness | Evidence | Implication |
|----------|----------|-------------|
| **Not immune to extreme noise** | ~22% accuracy at intensity=20 (Scenario B) | Fails at ~random guess under concentrated attack |
| **Precision jamming vulnerability** | Drops below 50% at intensity≈4 | Targeted attacks more effective than broadband |
| **High memory footprint** | D=10,000 dimensions per prototype | 5 classes × 10,000 × 4 bytes = 200KB |
| **Dataset-dependent 100%** | Works because classes are well-separated | May not achieve 100% on harder classification tasks |
| **No adversarial testing** | Only tested random noise | Optimized adversarial attacks not evaluated |

### ⚠️ Limitations of This Study

1. **Synthetic dataset**: Real RF signatures may have different statistical properties
2. **i.i.d. noise assumption**: Real jamming may have temporal/spectral structure
3. **No adversarial attacks**: We tested random noise, not optimized perturbations
4. **Fixed architecture**: We did not tune D, projection matrix, or encoding schemes
5. **Single experimental setup**: Results may vary with different parameters

---

## 📁 Repository Structure

```
VICEROY_2026_HDC_Sim/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── viceroy_hdc_sim.py                 # V1: Original simulation (bit-flip noise)
├── viceroy_hdc_v2.py                  # V2: Dual-doctrine simulation
├── viceroy_hdc_v3.py                  # V3: Scientific rigor update (RECOMMENDED)
│
├── viceroy_2026_hdc_results.png       # V1 output
├── viceroy_2026_hdc_results.pdf       # V1 output (print)
├── viceroy_2026_v2_dual_doctrine.png  # V2 output
├── viceroy_2026_v2_dual_doctrine.pdf  # V2 output (print)
├── viceroy_2026_v3_dual_doctrine.png  # V3 output (LATEST)
└── viceroy_2026_v3_dual_doctrine.pdf  # V3 output (print, LATEST)
```

---

## 🔄 Version History

### V3 (Current - Scientific Rigor Update)
- **Corrected SNR Analysis**: Signal energy (~450) > Noise energy (~250) at σ²=5
- **RNG Isolation**: Local `RandomState` prevents experimental coupling
- **Deterministic Encoding**: No random tie-breaking in `sign()`
- **Unified Scaler Logic**: `encode()` respects `is_fitted` state
- **Proper Notation**: σ² = variance, σ = standard deviation

### V2
- Random Projection architecture
- Dual-doctrine EW scenarios
- Fair comparison (same noisy tensor to both models)
- Input normalization for HDC

### V1
- Original bit-flip noise model
- Basic HDC vs MLP comparison

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/CisnerosCodes/VICEROY_2026_HDC_Sim.git
cd VICEROY_2026_HDC_Sim

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Simulation

```bash
# Run V3 (RECOMMENDED - scientifically rigorous)
python viceroy_hdc_v3.py

# Run V2 (dual doctrine, previous version)
python viceroy_hdc_v2.py

# Run V1 (original, bit-flip model)
python viceroy_hdc_sim.py
```

---

## 📐 Technical Details

### HDC Architecture (V3)

```python
class HDCLearnerV3:
    """
    Random Projection HDC with proper input scaling.
    
    Encoding: h = sign(M @ normalize(x))
    - M ∈ ℝ^(D×n), M[i,j] ~ N(0, 1/D)
    - normalize() = StandardScaler (fitted on training data)
    - sign() = deterministic bipolar quantization
    """
```

### EW Attack Models

| Scenario | Model | Parameters | Real-World Analog |
|----------|-------|------------|-------------------|
| A | Broadband AWGN | σ² ∈ [0, 5] on all features | Krasukha-4 area denial |
| B | Precision sweep | 10× intensity on 20% of features, rotating | AN/ALQ-249 surgical jamming |

### Key Mathematical Properties

| Property | Formula | Implication |
|----------|---------|-------------|
| Signal Energy | ‖c‖² = 9 × 50 = 450 | Strong class separation |
| Noise Energy | σ² × 50 | At σ²=5: 250 |
| SNR at σ²=5 | 450/250 = 1.8 | Signal dominates |
| Prototype SNR boost | √k ≈ 12× | Bundling improves robustness |

---

## 🎤 Talk Track for Presentation

When presenting this work, use these scientifically accurate talking points:

### On the 100% Accuracy
> "HDC achieves 100% accuracy at σ²=5 because the signal is actually **stronger** than the noise—SNR is about 1.8. The MLP fails not because the signal is buried, but because noise shifts the inputs away from its training distribution."

### On the sign() Function
> "The `sign()` function in HDC acts like a **hardware limiter**. When the MLP sees a corrupted value of 500, its activations explode. When HDC sees 500, it simply outputs +1 and continues normally. This is analogous to a 1-bit ADC in RF systems."

### On Input Normalization
> "We use `StandardScaler` to normalize inputs before projection. This is equivalent to **Automatic Gain Control (AGC)** in real RF receivers—it ensures the random projection operates on properly scaled data."

### On Honest Limitations
> "HDC is not immune to noise. In Scenario B with intensity above 10, HDC also fails—it just degrades more gracefully than the MLP. This is **graceful degradation**, not immunity."

---

## 📚 References

1. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation." *Cognitive Computation*.

2. Rahimi, A., et al. (2016). "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing." *ISLPED*.

3. Johnson, W. B., & Lindenstrauss, J. (1984). "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*.

4. Imani, M., et al. (2019). "A Framework for Collaborative Learning in Secure High-Dimensional Space." *IEEE CLOUD*.

---

## 🤝 Contributing

This is a symposium demonstration project. For questions or collaboration inquiries, please open an issue.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{viceroy2026hdc,
  author = {Cisneros, Adrian},
  title = {VICEROY 2026: HDC for EW-Resilient Command Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/CisnerosCodes/VICEROY_2026_HDC_Sim}
}
```

---

*UNCLASSIFIED // FOR OFFICIAL USE ONLY*
