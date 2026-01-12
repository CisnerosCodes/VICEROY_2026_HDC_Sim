# VICEROY 2026: Hyperdimensional Computing for EW-Resilient Command Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference: VICEROY 2026](https://img.shields.io/badge/Conference-VICEROY%202026-green.svg)]()

> **Symposium Poster Companion Repository**  
> "Cognitive Resilience at the Edge: Using Hyperdimensional Computing (HDC) to Secure Autonomous Wingmen (CCA) Against Spectrum Jamming"

---

## 📋 Executive Summary

This repository contains the complete simulation code for our VICEROY 2026 symposium poster demonstrating that **Hyperdimensional Computing (HDC)** provides **graceful degradation** (not immunity) against electronic warfare (EW) jamming attacks, outperforming traditional deep learning approaches.

### Key Results at a Glance

| Scenario | Attack Type | HDC Accuracy | MLP Accuracy | Interpretation |
|----------|-------------|--------------|--------------|----------------|
| **A** (σ²=0) | None (clean) | 100% | 98.7% | Both work well |
| **A** (σ²=5) | Russian Broadband | **100%** | 57% | HDC resilient |
| **B** (int=4) | US Precision | 49% | 22% | Both degraded |
| **B** (int=20) | US Precision (extreme) | 22% | 23% | **Both fail** |

---

## 🔬 Why Does HDC Achieve 100% in Scenario A? (Mathematical Explanation)

We anticipated skepticism about the 100% accuracy under broadband noise. This is **not too good to be true**—it's a direct consequence of the mathematics. Here's the rigorous explanation:

### The Setup

- **Input dimension**: n = 50 features
- **Hypervector dimension**: D = 10,000
- **Encoding**: `h = sign(M @ x)` where M ∈ ℝ^(D×n), M[i,j] ~ N(0, 1/√D)
- **Noise model** (Scenario A): x_noisy = x + ε, where ε ~ N(0, σ²I)

### Why It Works: The Signal-to-Noise Ratio Argument

For a clean input x and noise ε, the projection is:

```
M @ x_noisy = M @ x + M @ ε
              ↑         ↑
           signal     noise
```

**Key insight**: Both signal and noise are projected through the SAME random matrix M.

#### Signal Term: ||M @ x||²
- Expected value: E[||M @ x||²] = ||x||² (variance-normalized projection)
- For our dataset: ||x|| ≈ 3.0 (class centroids scaled by 3.0)

#### Noise Term: ||M @ ε||²  
- Expected value: E[||M @ ε||²] = σ² × n (sum of n independent Gaussians)
- At σ² = 5.0: E[||M @ ε||²] = 5.0 × 50 = 250

**Wait—the noise magnitude is LARGER than the signal!** So why does HDC still work?

### The Sign Function: The Unsung Hero

The critical step is `sign(M @ x_noisy)`. The sign function acts as a **majority vote** across dimensions:

1. Each dimension d of the projection is: `(M @ x)[d] + (M @ ε)[d]`

2. The signal component `(M @ x)[d]` has **consistent direction** across all samples of the same class (because they're projected through the same M from similar inputs)

3. The noise component `(M @ ε)[d]` is **random and independent** for each sample

4. When we **bundle (sum) many training samples** to form the class prototype:
   - Signal components **add constructively** (all point same direction)
   - Noise components **cancel out** (random directions average to ~0)

### Quantitative Bound

For k training samples per class, the prototype's signal-to-noise ratio improves by √k:

```
SNR_prototype ≈ √k × SNR_single_sample
```

With k = 140 samples per class (700 training / 5 classes):
- √140 ≈ 12× improvement in prototype SNR
- Even if single-sample SNR < 1, prototype SNR >> 1

### Why the MLP Fails

The MLP does NOT benefit from this averaging effect at inference time:
1. Each test sample is classified individually
2. No "prototype averaging" to cancel noise
3. ReLU activations can saturate or explode with noisy inputs
4. Learned weights are optimized for clean data distribution

### The Limit of HDC Robustness

HDC's 100% accuracy in Scenario A is **dataset-dependent**. It works because:
1. Our classes are well-separated (centroid distance >> intra-class variance)
2. The noise variance (σ² = 5) is still within the regime where sign() voting works
3. We have enough training samples for good prototype averaging

**At higher noise levels, HDC would also fail.** The 100% is not magic—it's the sweet spot of our experimental parameters.

---

## 📊 Honest Assessment: Strengths & Weaknesses

### ✅ Strengths of HDC

| Strength | Evidence | Mechanism |
|----------|----------|-----------|
| **Graceful degradation** | Accuracy drops smoothly, not catastrophically | Distributed representation prevents single points of failure |
| **Noise averaging** | 100% accuracy at σ²=5 (Scenario A) | Random projection + prototype bundling cancels i.i.d. noise |
| **Binary robustness** | sign() clips extreme values | Prevents numerical instability that affects MLPs |
| **Simple training** | No backprop, no hyperparameter tuning | Just matrix multiplication and summation |
| **Interpretable** | Cosine similarity to prototypes | Direct geometric intuition |

### ❌ Weaknesses of HDC

| Weakness | Evidence | Implication |
|----------|----------|-------------|
| **Not immune to extreme noise** | 22% accuracy at intensity=20 (Scenario B) | Fails at ~random guess under concentrated attack |
| **Precision jamming vulnerability** | Drops below 50% at intensity=4 | Targeted attacks are more effective than broadband |
| **High memory footprint** | D=10,000 dimensions per prototype | 5 classes × 10,000 × 4 bytes = 200KB (acceptable but larger than MLP) |
| **Binary quantization loses information** | sign() discards magnitude | May underperform on tasks requiring fine-grained distinctions |
| **Projection matrix must be shared** | Training and inference need same M | Requires secure distribution of the projection matrix |

### ⚠️ Limitations of This Study

1. **Synthetic dataset**: Real RF signatures may have different statistical properties
2. **i.i.d. noise assumption**: Real jamming may have temporal/spectral structure
3. **No adversarial attacks**: We tested random noise, not optimized adversarial perturbations
4. **Fixed architecture**: We did not tune D, the projection matrix distribution, or encoding schemes
5. **Single random seed**: Results may vary slightly with different random initializations

---

## 🏗️ Repository Structure

```
VICEROY_2026_HDC_Sim/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── viceroy_hdc_sim.py                 # V1: Original simulation (bit-flip noise)
├── viceroy_hdc_v2.py                  # V2: Dual-doctrine simulation (RECOMMENDED)
├── viceroy_2026_hdc_results.png       # V1 output visualization
├── viceroy_2026_hdc_results.pdf       # V1 output (print quality)
├── viceroy_2026_v2_dual_doctrine.png  # V2 output visualization
└── viceroy_2026_v2_dual_doctrine.pdf  # V2 output (print quality)
```

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
# Run V2 (recommended - dual doctrine comparison)
python viceroy_hdc_v2.py

# Run V1 (original bit-flip noise simulation)
python viceroy_hdc_sim.py
```

### Expected Output

The simulation will:
1. Run verification tests to validate HDC mathematical properties
2. Train both HDC and MLP models on clean data
3. Test both models under increasing noise levels
4. Generate publication-quality visualizations (PNG + PDF)
5. Print detailed performance summaries

---

## 📐 Technical Details

### HDC Architecture (V2)

```python
class HDCLearnerV2:
    """
    Random Projection HDC with input normalization.
    
    Encoding: h = sign(M @ normalize(x))
    - M ∈ ℝ^(D×n), M[i,j] ~ N(0, 1/√D)
    - normalize() = StandardScaler (zero mean, unit variance)
    - sign() = bipolar quantization to {-1, +1}
    """
```

### EW Attack Models

| Scenario | Model | Parameters | Real-World Analog |
|----------|-------|------------|-------------------|
| A | Broadband AWGN | σ² ∈ [0, 5] on all features | Krasukha-4 area denial |
| B | Precision sweep | 10× intensity on 20% of features, rotating | AN/ALQ-249 surgical jamming |

### Why Random Projection?

The **Johnson-Lindenstrauss Lemma** guarantees that random projection approximately preserves distances:

> For any ε > 0 and n points, a random projection into D = O(log(n)/ε²) dimensions preserves all pairwise distances within factor (1±ε).

For our D = 10,000, this provides excellent distance preservation, meaning similar inputs produce similar hypervectors.

---

## 📈 Reproducing Our Results

The simulation uses fixed random seeds for reproducibility:

```python
np.random.seed(2026)  # Main simulation seed
np.random.seed(42)    # Class centroid generation
```

Expected results (may vary ±2% due to MLP training stochasticity):

**Scenario A (Broadband)**:
- HDC: 100% (σ²=0) → 100% (σ²=5)
- MLP: 98.7% (σ²=0) → 57% (σ²=5)

**Scenario B (Precision)**:
- HDC: 100% (int=0) → 22% (int=20)
- MLP: 98.7% (int=0) → 23% (int=20)

---

## 📚 References

1. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation*.

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

## 📞 Contact

**VICEROY 2026 Symposium Poster Session**  
*DoD/Academic Partnership Initiative*

---

*UNCLASSIFIED // FOR OFFICIAL USE ONLY*
