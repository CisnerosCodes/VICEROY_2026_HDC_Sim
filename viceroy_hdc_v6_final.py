"""
================================================================================
VICEROY 2026 SYMPOSIUM - HYPERDIMENSIONAL COMPUTING SIMULATION V6 FINAL
================================================================================
Title: "Cognitive Resilience at the Edge: HDC Robustness Against Adversarial
        Electronic Warfare Doctrines"

Author: Senior Defense Research Scientist
Date: February 2026
Version: 6.0 (DEPLOYMENT READY - Tower 2 Hardware)
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY

DEPLOYMENT TARGET:
------------------
- Hardware: Dell Optiplex i5-4590 CPU ("Tower 2")
- Python: 3.12 (CPU-only, no CUDA)
- Dataset: RadioML 2016.10A Tactical Subset

CHANGELOG V5 → V6:
------------------
1. ARGPARSE CLI: Simplified hyperparameter control
   --dim, --epochs, --output

2. LATENCY OPTIMIZATION:
   - Pre-computed RFF weights (W) and bias (b) at initialization
   - Vectorized inference path
   - Dimension management: input --dim D → internal 2*D (Sin+Cos)

3. JSON BENCHMARK OUTPUT:
   - Structured metrics for paper data points
   - Accuracy, Training Time, Inference Latency (μs)

4. TACTICAL SUBSET ONLY:
   - 5 digital modulations: BPSK, QPSK, 8PSK, QAM16, QAM64
   - Focused on drone C2 link defense scenario

USAGE:
------
  Sniper Run (High Accuracy):
    python viceroy_hdc_v6_final.py --dim 10000 --epochs 20

  Scout Run (High Speed):
    python viceroy_hdc_v6_final.py --dim 2000 --epochs 5

================================================================================
"""

import argparse
import json
import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

RADIOML_PATH = "./.data/RML2016.10a_dict.pkl"
RADIOML_HIGH_SNR_THRESHOLD = 10  # Train on SNR >= +10dB

# Tactical subset: 5 digital modulations for drone C2 defense
TACTICAL_MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']


# =============================================================================
# RADIOML 2016.10A TACTICAL SUBSET LOADER
# =============================================================================

def load_radioml_tactical(dataset_path=RADIOML_PATH, verbose=True):
    """
    Load RadioML 2016.10A dataset - TACTICAL SUBSET ONLY.
    
    Filters to 5 digital modulations relevant to drone C2 links:
    - BPSK, QPSK, 8PSK, QAM16, QAM64
    
    Returns:
        dict with X_train, y_train, X_test_by_snr, y_test_by_snr, metadata
        Returns None if dataset not found
    """
    if not os.path.exists(dataset_path):
        if verbose:
            print("=" * 70)
            print("ERROR: RadioML 2016.10A Dataset Not Found")
            print("=" * 70)
            print(f"  Expected path: {os.path.abspath(dataset_path)}")
            print()
            print("  Download from DeepSig and place at:")
            print(f"    {dataset_path}")
            print("=" * 70)
        return None
    
    if verbose:
        print("Loading RadioML 2016.10A Tactical Subset...")
    
    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    # Parse available SNR levels
    snr_levels = sorted(set(snr for (mod, snr) in data_dict.keys()))
    
    # Filter to tactical modulations only
    available_mods = set(mod for (mod, snr) in data_dict.keys())
    tactical_mods = [m for m in TACTICAL_MODULATIONS if m in available_mods]
    
    if verbose:
        print(f"  Tactical modulations: {tactical_mods}")
        print(f"  SNR range: {min(snr_levels)}dB to {max(snr_levels)}dB")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(tactical_mods)
    
    # Split by SNR
    X_train_list = []
    y_train_list = []
    X_test_by_snr = {}
    y_test_by_snr = {}
    
    for snr in snr_levels:
        X_snr_list = []
        y_snr_list = []
        
        for mod in tactical_mods:
            if (mod, snr) in data_dict:
                samples = data_dict[(mod, snr)]  # Shape: (N, 2, 128)
                n_samples = samples.shape[0]
                
                # Flatten IQ to 256-dim vector
                samples_flat = samples.reshape(n_samples, -1)
                
                X_snr_list.append(samples_flat)
                y_snr_list.append([mod] * n_samples)
        
        if X_snr_list:
            X_snr = np.vstack(X_snr_list)
            y_snr = np.concatenate(y_snr_list)
            y_snr_encoded = label_encoder.transform(y_snr)
            
            X_test_by_snr[snr] = X_snr
            y_test_by_snr[snr] = y_snr_encoded
            
            # High-SNR samples for training
            if snr >= RADIOML_HIGH_SNR_THRESHOLD:
                X_train_list.append(X_snr)
                y_train_list.append(y_snr_encoded)
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Shuffle training data
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    if verbose:
        print(f"  Training samples (SNR >= {RADIOML_HIGH_SNR_THRESHOLD}dB): {len(X_train)}")
        print(f"  Feature dimension: {X_train.shape[1]} (2 × 128 IQ)")
        print(f"  Classes: {len(tactical_mods)}")
        print()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test_by_snr': X_test_by_snr,
        'y_test_by_snr': y_test_by_snr,
        'label_encoder': label_encoder,
        'snr_levels': snr_levels,
        'modulations': tactical_mods,
        'num_classes': len(tactical_mods)
    }


# =============================================================================
# DSP PREPROCESSING: FFT MAGNITUDE (PHASE-INVARIANT)
# =============================================================================

def dsp_preprocess(X):
    """
    Convert raw IQ samples to FFT Magnitude Spectrum.
    
    This makes encoding SHIFT-INVARIANT (immune to phase/time offsets).
    
    Args:
        X: Raw IQ samples, shape (n_samples, 256)
           Layout: first 128 = I channel, last 128 = Q channel
           
    Returns:
        X_dsp: Magnitude spectrum, shape (n_samples, 128)
    """
    if X.shape[1] != 256:
        return X  # Pass through if wrong dimension
    
    n_samples = X.shape[0]
    
    # Reshape to (n_samples, 2, 128)
    X_iq = X.reshape(n_samples, 2, 128)
    
    # Extract I and Q channels
    I = X_iq[:, 0, :]  # (n_samples, 128)
    Q = X_iq[:, 1, :]  # (n_samples, 128)
    
    # Convert to complex signal
    signal = I + 1j * Q  # (n_samples, 128)
    
    # Compute FFT along time axis
    spectrum = np.fft.fft(signal, axis=1)  # (n_samples, 128)
    
    # Extract magnitude (phase-invariant)
    magnitude = np.abs(spectrum)  # (n_samples, 128)
    
    return magnitude


# =============================================================================
# HDC-RFF: RANDOM FOURIER FEATURES ENCODER (OPTIMIZED)
# =============================================================================

class HDCLearnerV6_RFF:
    """
    Hyperdimensional Computing Classifier V6 - Random Fourier Features.
    
    OPTIMIZATION NOTES:
    -------------------
    1. Pre-computed W (weights) and b (bias) at initialization
    2. Internal dimension = 2 * D (Sin + Cos concatenation)
    3. Vectorized batch operations for inference
    4. Continuous prototype updates (Perceptron-style)
    
    RFF KERNEL APPROXIMATION:
    -------------------------
    The RFF transform approximates an RBF kernel:
        k(x, y) ≈ φ(x)·φ(y)
    where:
        φ(x) = [cos(γ·x·W + b), sin(γ·x·W + b)]
        
    γ (gamma) controls kernel bandwidth:
        - High γ: Narrow kernel, captures fine details
        - Low γ: Wide kernel, smoother decision boundaries
    """
    
    def __init__(self, input_dim, rff_dim=10000, num_classes=5, gamma=1.0, seed=42):
        """
        Initialize HDC-RFF Learner.
        
        Args:
            input_dim: Dimension of input features (128 after DSP)
            rff_dim: RFF projection dimension D (internal = 2*D)
            num_classes: Number of classes
            gamma: RBF kernel bandwidth parameter
            seed: Random seed for reproducibility
        """
        self.rff_dim = rff_dim
        self.internal_dim = 2 * rff_dim  # Sin + Cos
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.seed = seed
        
        # Initialize RNG
        self.rng = np.random.RandomState(seed)
        
        # Pre-compute RFF projection matrix and bias (LATENCY OPTIMIZATION)
        # W ~ N(0, 1) scaled by gamma
        self.W = self.rng.randn(input_dim, rff_dim).astype(np.float32) * gamma
        # b ~ Uniform(0, 2π)
        self.b = self.rng.uniform(0, 2 * np.pi, rff_dim).astype(np.float32)
        
        # Scaler for input normalization
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Class prototypes (continuous, not binarized)
        self.prototypes = None
        self.class_labels = None
        
    def rff_encode(self, X):
        """
        Encode batch using Random Fourier Features.
        
        Args:
            X: Input features, shape (n_samples, input_dim)
            
        Returns:
            H: RFF encoding, shape (n_samples, 2*rff_dim)
        """
        # Project: Z = X @ W + b
        Z = X @ self.W + self.b  # (n_samples, rff_dim)
        
        # Concatenate cos and sin (approximates RBF kernel)
        H = np.concatenate([np.cos(Z), np.sin(Z)], axis=1)  # (n_samples, 2*rff_dim)
        
        # Normalize to unit length
        H = H / np.sqrt(self.rff_dim)
        
        return H.astype(np.float32)
    
    def train(self, X, y, epochs=10, learning_rate=1.0, verbose=True):
        """
        Train using Online Iterative Learning (Perceptron-style).
        
        For each misclassified sample:
            prototype[correct] += h
            prototype[predicted] -= h
            
        Args:
            X: Training features (raw IQ, 256-dim)
            y: Training labels
            epochs: Number of training passes
            learning_rate: Update magnitude (default 1.0)
            verbose: Print progress
        """
        # DSP preprocessing: IQ → FFT Magnitude
        X_dsp = dsp_preprocess(X)
        
        # Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X_dsp)
        self.is_fitted = True
        
        # Encode all samples
        H = self.rff_encode(X_scaled)
        
        # Get unique classes
        self.class_labels = np.unique(y)
        n_classes = len(self.class_labels)
        
        # Initialize prototypes by bundling (one-shot baseline)
        self.prototypes = np.zeros((n_classes, self.internal_dim), dtype=np.float32)
        for i, label in enumerate(self.class_labels):
            mask = y == label
            self.prototypes[i] = np.mean(H[mask], axis=0)
        
        # Iterative refinement (Perceptron-style)
        n_samples = len(y)
        
        for epoch in range(epochs):
            # Shuffle training order
            shuffle_idx = self.rng.permutation(n_samples)
            H_shuffled = H[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            errors = 0
            
            for i in range(n_samples):
                h = H_shuffled[i]
                true_label = y_shuffled[i]
                true_idx = np.where(self.class_labels == true_label)[0][0]
                
                # Predict
                similarities = self.prototypes @ h
                pred_idx = np.argmax(similarities)
                
                if pred_idx != true_idx:
                    errors += 1
                    # Update prototypes
                    self.prototypes[true_idx] += learning_rate * h
                    self.prototypes[pred_idx] -= learning_rate * h
            
            # Normalize prototypes
            norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
            self.prototypes = self.prototypes / np.maximum(norms, 1e-8)
            
            if verbose and (epoch + 1) % 2 == 0:
                acc = 1.0 - errors / n_samples
                print(f"    Epoch {epoch+1}/{epochs}: Training Acc = {acc*100:.1f}%")
    
    def predict(self, X):
        """
        Predict class labels for input samples.
        
        Args:
            X: Input features (raw IQ, 256-dim)
            
        Returns:
            predictions: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")
        
        # DSP preprocessing
        X_dsp = dsp_preprocess(X)
        
        # Scale
        X_scaled = self.scaler.transform(X_dsp)
        
        # RFF encode
        H = self.rff_encode(X_scaled)
        
        # Compute similarities to all prototypes
        similarities = H @ self.prototypes.T  # (n_samples, n_classes)
        
        # Predict
        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_labels[pred_indices]
        
        return predictions
    
    def predict_single(self, x):
        """
        Predict single sample (for latency benchmarking).
        
        Args:
            x: Single input vector (256-dim)
            
        Returns:
            prediction: Predicted class label
        """
        x = x.reshape(1, -1)
        return self.predict(x)[0]
    
    def accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# =============================================================================
# STEEL MAN MLP (UNCHANGED FROM V5)
# =============================================================================

class SteelManMLP:
    """
    Steel Man Multi-Layer Perceptron - Strongest Possible Baseline.
    
    Architecture: (256, 128, 64)
    L2 Regularization: α=0.01
    Adversarial Training: 50% clean + 50% noisy (σ²=1.0)
    """
    
    def __init__(self, seed=42):
        """Initialize Steel Man MLP."""
        self.scaler = StandardScaler()
        self.seed = seed
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.is_trained = False
        self.adversarial_noise_var = 1.0
        
    def train(self, X, y, adversarial=True, verbose=True):
        """
        Train Steel Man MLP with adversarial augmentation.
        
        Args:
            X: Training features (raw IQ, 256-dim)
            y: Training labels
            adversarial: Use adversarial training (default True)
            verbose: Print progress
        """
        # DSP preprocessing
        X_dsp = dsp_preprocess(X)
        
        rng = np.random.RandomState(self.seed)
        
        if adversarial:
            # Generate noisy copies
            noise_std = np.sqrt(self.adversarial_noise_var)
            X_noisy = X_dsp + rng.randn(*X_dsp.shape) * noise_std
            
            # Combine 50/50
            X_augmented = np.vstack([X_dsp, X_noisy])
            y_augmented = np.concatenate([y, y])
            
            # Shuffle
            shuffle_idx = rng.permutation(len(X_augmented))
            X_augmented = X_augmented[shuffle_idx]
            y_augmented = y_augmented[shuffle_idx]
            
            # Fit scaler on clean data
            self.scaler.fit(X_dsp)
            X_scaled = self.scaler.transform(X_augmented)
        else:
            X_scaled = self.scaler.fit_transform(X_dsp)
            y_augmented = y
        
        if verbose:
            print("    Training Steel Man MLP...")
        
        self.model.fit(X_scaled, y_augmented)
        self.is_trained = True
        
        if verbose:
            print(f"    MLP converged in {self.model.n_iter_} iterations")
    
    def predict(self, X):
        """Predict class labels."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        X_dsp = dsp_preprocess(X)
        X_scaled = self.scaler.transform(X_dsp)
        return self.model.predict(X_scaled)
    
    def predict_single(self, x):
        """Predict single sample."""
        x = x.reshape(1, -1)
        return self.predict(x)[0]
    
    def accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# =============================================================================
# BENCHMARK SUITE
# =============================================================================

def benchmark_training_time(model_class, X_train, y_train, model_kwargs, train_kwargs, n_runs=3):
    """
    Benchmark training time.
    
    Returns:
        mean_time: Average training time in seconds
        std_time: Standard deviation
    """
    times = []
    
    for _ in range(n_runs):
        model = model_class(**model_kwargs)
        start = time.perf_counter()
        model.train(X_train, y_train, **train_kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def benchmark_inference_latency(model, X_test, n_samples=1000):
    """
    Benchmark single-sample inference latency.
    
    Returns:
        mean_latency_us: Average latency in microseconds
        std_latency_us: Standard deviation
    """
    # Select random samples
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    
    latencies = []
    
    for idx in indices:
        x = X_test[idx]
        start = time.perf_counter()
        _ = model.predict_single(x)
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to microseconds
    
    return np.mean(latencies), np.std(latencies)


def benchmark_accuracy_by_snr(model, X_test_by_snr, y_test_by_snr):
    """
    Compute accuracy at each SNR level.
    
    Returns:
        acc_by_snr: Dict mapping SNR → accuracy
    """
    acc_by_snr = {}
    for snr in sorted(X_test_by_snr.keys()):
        acc = model.accuracy(X_test_by_snr[snr], y_test_by_snr[snr])
        acc_by_snr[snr] = acc
    return acc_by_snr


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point with argparse CLI."""
    
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='VICEROY 2026 V6 - HDC-RFF vs Steel Man MLP Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Sniper Run (High Accuracy):
    python viceroy_hdc_v6_final.py --dim 10000 --epochs 20
    
  Scout Run (High Speed):
    python viceroy_hdc_v6_final.py --dim 2000 --epochs 5
        """
    )
    
    parser.add_argument('--dim', type=int, default=10000,
                        help='RFF projection dimension D (internal = 2*D). Default: 10000')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training passes. Default: 10')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='RFF kernel bandwidth (gamma). Default: 1.0')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file for results. Default: results.json')
    
    args = parser.parse_args()
    
    verbose = True  # Always verbose for deployment debugging
    
    # =========================================================================
    # HEADER
    # =========================================================================
    if verbose:
        print()
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "  VICEROY 2026 V6 FINAL - TOWER 2 DEPLOYMENT".center(68) + "║")
        print("║" + "  HDC-RFF vs Steel Man MLP Benchmark".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        print("Configuration:")
        print(f"  --dim    = {args.dim} (internal = {2*args.dim})")
        print(f"  --epochs = {args.epochs}")
        print(f"  --gamma  = {args.gamma}")
        print(f"  --output = {args.output}")
        print()
    
    # =========================================================================
    # LOAD DATASET
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("PHASE 1: Loading RadioML 2016.10A Tactical Subset")
        print("=" * 70)
    
    data = load_radioml_tactical(verbose=verbose)
    
    if data is None:
        print("FATAL: RadioML dataset not found. Cannot proceed.")
        return 1
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test_by_snr = data['X_test_by_snr']
    y_test_by_snr = data['y_test_by_snr']
    snr_levels = data['snr_levels']
    num_classes = data['num_classes']
    
    # Combine high-SNR test data for overall accuracy
    high_snr_levels = [snr for snr in snr_levels if snr >= RADIOML_HIGH_SNR_THRESHOLD]
    X_test_high = np.vstack([X_test_by_snr[snr] for snr in high_snr_levels])
    y_test_high = np.concatenate([y_test_by_snr[snr] for snr in high_snr_levels])
    
    # Input dimension after DSP preprocessing
    input_dim = 128  # FFT Magnitude output
    
    # =========================================================================
    # TRAIN HDC-RFF
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("PHASE 2: Training HDC-RFF Model")
        print("=" * 70)
        print(f"  RFF Dimension: {args.dim} → Internal: {2*args.dim}")
        print(f"  Epochs: {args.epochs}")
        print()
    
    hdc_start = time.perf_counter()
    hdc = HDCLearnerV6_RFF(
        input_dim=input_dim,
        rff_dim=args.dim,
        num_classes=num_classes,
        gamma=args.gamma,
        seed=42
    )
    hdc.train(X_train, y_train, epochs=args.epochs, verbose=verbose)
    hdc_train_time = time.perf_counter() - hdc_start
    
    if verbose:
        print(f"  HDC-RFF training complete: {hdc_train_time:.2f}s")
        print()
    
    # =========================================================================
    # TRAIN STEEL MAN MLP
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("PHASE 3: Training Steel Man MLP")
        print("=" * 70)
        print("  Architecture: (256, 128, 64)")
        print("  L2 Regularization: α=0.01")
        print("  Adversarial Training: 50% clean + 50% noisy")
        print()
    
    mlp_start = time.perf_counter()
    mlp = SteelManMLP(seed=42)
    mlp.train(X_train, y_train, adversarial=True, verbose=verbose)
    mlp_train_time = time.perf_counter() - mlp_start
    
    if verbose:
        print(f"  Steel Man MLP training complete: {mlp_train_time:.2f}s")
        print()
    
    # =========================================================================
    # BENCHMARK: ACCURACY BY SNR
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("PHASE 4: Accuracy by SNR")
        print("=" * 70)
        print()
        print(f"  {'SNR (dB)':<10} {'HDC-RFF':<12} {'Steel MLP':<12} {'Δ':<10}")
        print(f"  {'-'*8:<10} {'-'*10:<12} {'-'*10:<12} {'-'*8:<10}")
    
    hdc_acc_by_snr = benchmark_accuracy_by_snr(hdc, X_test_by_snr, y_test_by_snr)
    mlp_acc_by_snr = benchmark_accuracy_by_snr(mlp, X_test_by_snr, y_test_by_snr)
    
    if verbose:
        for snr in snr_levels:
            hdc_acc = hdc_acc_by_snr[snr] * 100
            mlp_acc = mlp_acc_by_snr[snr] * 100
            delta = hdc_acc - mlp_acc
            print(f"  {snr:<10} {hdc_acc:<12.1f} {mlp_acc:<12.1f} {delta:+.1f}")
        print()
    
    # Compute aggregate metrics
    high_snr_hdc = np.mean([hdc_acc_by_snr[snr] for snr in high_snr_levels])
    high_snr_mlp = np.mean([mlp_acc_by_snr[snr] for snr in high_snr_levels])
    
    low_snr_levels = [snr for snr in snr_levels if snr <= 0]
    low_snr_hdc = np.mean([hdc_acc_by_snr[snr] for snr in low_snr_levels])
    low_snr_mlp = np.mean([mlp_acc_by_snr[snr] for snr in low_snr_levels])
    
    # =========================================================================
    # BENCHMARK: INFERENCE LATENCY
    # =========================================================================
    if verbose:
        print("=" * 70)
        print("PHASE 5: Inference Latency Benchmark")
        print("=" * 70)
        print("  Measuring single-sample inference time...")
        print()
    
    hdc_latency_mean, hdc_latency_std = benchmark_inference_latency(hdc, X_test_high)
    mlp_latency_mean, mlp_latency_std = benchmark_inference_latency(mlp, X_test_high)
    
    if verbose:
        print(f"  HDC-RFF:      {hdc_latency_mean:.1f} ± {hdc_latency_std:.1f} μs/sample")
        print(f"  Steel MLP:    {mlp_latency_mean:.1f} ± {mlp_latency_std:.1f} μs/sample")
        print(f"  Speedup:      {mlp_latency_mean/hdc_latency_mean:.2f}× (HDC vs MLP)")
        print()
    
    # =========================================================================
    # ENERGY PROXY (CPU TIME)
    # =========================================================================
    # Use training time as energy proxy (CPU-bound workload)
    hdc_energy_proxy = hdc_train_time
    mlp_energy_proxy = mlp_train_time
    
    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "rff_dim": args.dim,
            "internal_dim": 2 * args.dim,
            "epochs": args.epochs,
            "gamma": args.gamma,
            "dataset": "RadioML 2016.10A Tactical Subset",
            "modulations": TACTICAL_MODULATIONS
        },
        "hdc_rff": {
            "model": f"HDC-RFF (D={args.dim})",
            "accuracy_high_snr_percent": round(high_snr_hdc * 100, 2),
            "accuracy_low_snr_percent": round(low_snr_hdc * 100, 2),
            "training_time_seconds": round(hdc_train_time, 2),
            "inference_latency_us": round(hdc_latency_mean, 1),
            "accuracy_by_snr": {str(k): round(v * 100, 2) for k, v in hdc_acc_by_snr.items()}
        },
        "steel_mlp": {
            "model": "Steel Man MLP (256-128-64)",
            "accuracy_high_snr_percent": round(high_snr_mlp * 100, 2),
            "accuracy_low_snr_percent": round(low_snr_mlp * 100, 2),
            "training_time_seconds": round(mlp_train_time, 2),
            "inference_latency_us": round(mlp_latency_mean, 1),
            "accuracy_by_snr": {str(k): round(v * 100, 2) for k, v in mlp_acc_by_snr.items()}
        },
        "comparison": {
            "accuracy_delta_high_snr": round((high_snr_hdc - high_snr_mlp) * 100, 2),
            "accuracy_delta_low_snr": round((low_snr_hdc - low_snr_mlp) * 100, 2),
            "training_speedup": round(mlp_train_time / hdc_train_time, 2),
            "inference_speedup": round(mlp_latency_mean / hdc_latency_mean, 2),
            "energy_efficiency_ratio": round(mlp_energy_proxy / hdc_energy_proxy, 2)
        }
    }
    
    # Save JSON results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("=" * 70)
        print("PHASE 6: Final Summary")
        print("=" * 70)
        print()
        print(f"  {'Metric':<30} {'HDC-RFF':<15} {'Steel MLP':<15}")
        print(f"  {'-'*28:<30} {'-'*13:<15} {'-'*13:<15}")
        print(f"  {'Accuracy (High-SNR %)':<30} {high_snr_hdc*100:<15.1f} {high_snr_mlp*100:<15.1f}")
        print(f"  {'Accuracy (Low-SNR %)':<30} {low_snr_hdc*100:<15.1f} {low_snr_mlp*100:<15.1f}")
        print(f"  {'Training Time (s)':<30} {hdc_train_time:<15.2f} {mlp_train_time:<15.2f}")
        print(f"  {'Inference Latency (μs)':<30} {hdc_latency_mean:<15.1f} {mlp_latency_mean:<15.1f}")
        print()
        print(f"  Training Speedup:    HDC is {mlp_train_time/hdc_train_time:.1f}× faster")
        print(f"  Inference Speedup:   {'HDC' if hdc_latency_mean < mlp_latency_mean else 'MLP'} is "
              f"{max(hdc_latency_mean, mlp_latency_mean)/min(hdc_latency_mean, mlp_latency_mean):.1f}× faster")
        print()
        print(f"  Results saved to: {args.output}")
        print()
    
    # =========================================================================
    # FINAL BANNER
    # =========================================================================
    if verbose:
        print("╔" + "═" * 68 + "╗")
        print("║" + "  VICEROY 2026 V6 BENCHMARK COMPLETE".center(68) + "║")
        print("║" + f"  HDC-RFF (D={args.dim}) vs Steel Man MLP".center(68) + "║")
        print("╚" + "═" * 68 + "╝")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())
