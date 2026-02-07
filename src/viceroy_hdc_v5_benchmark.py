"""
================================================================================
VICEROY 2026 SYMPOSIUM - HYPERDIMENSIONAL COMPUTING SIMULATION V5
================================================================================
Title: "Cognitive Resilience at the Edge: HDC Robustness Against Adversarial
        Electronic Warfare Doctrines"

Author: Senior Defense Research Scientist
Date: February 2026
Version: 5.0 (Final Benchmark - RFF + Iterative Learning)
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY

CHANGELOG V4 → V5:
------------------
1. DATASET STRATEGY: "Tactical Subset"
   - Filter to 5 Digital Modulations: BPSK, QPSK, 8PSK, QAM16, QAM64
   - Removes high-variance analog signals (AM-DSB, AM-SSB, WBFM, etc.)
   - Realistic "Drone Defense" scenario
   
2. HDC ARCHITECTURE: Random Fourier Features (RFF)
   - Encoder: H = Concat(Cos(X @ P), Sin(X @ P))
   - Approximates RBF kernel for non-linear decision boundaries
   - Continuous output (floats) for expressivity
   
3. TRAINING: Online/Iterative Perceptron-Style
   - Initialize prototypes as zeros
   - Multi-epoch training with error-driven updates
   - Prototype[Target] += Encoded_X on misclassification
   - Prototype[Predicted] -= Encoded_X on misclassification
   
4. BENCHMARK SUITE: "Drone Battery"
   - Metric A: Training Energy (wall-clock time)
   - Metric B: Inference Latency (μs per sample)
   - Metric C: Day 0 Adaptation Rate (learning new class quickly)

HARDWARE TARGET:
----------------
- Dell Optiplex i5-4590 (Tower 2 - "Drone Hardware")
- CPU-only (AVX2, no GPU)
- Python 3.12, scikit-learn, numpy, pickle
================================================================================
"""

import numpy as np
import pickle
import os
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL RANDOM SEED
# =============================================================================
np.random.seed(2026)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

D = 10000  # Hypervector dimensionality
RFF_GAMMA = 0.1  # RBF kernel bandwidth parameter

# RadioML configuration
RADIOML_PATH = os.path.join(os.path.dirname(__file__), "..", ".data", "RML2016.10a_dict.pkl")
RADIOML_HIGH_SNR_THRESHOLD = 10  # Train on SNR >= +10dB

# Tactical Subset: 5 Digital Modulations Only
TACTICAL_MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']

# Iterative training config
ITERATIVE_EPOCHS = 10
ITERATIVE_LEARNING_RATE = 1.0


# =============================================================================
# RADIOML TACTICAL SUBSET LOADER
# =============================================================================

def load_radioml_tactical(dataset_path=RADIOML_PATH):
    """
    Load RadioML 2016.10A dataset filtered to Tactical Subset.
    
    TACTICAL SUBSET:
    ----------------
    Only 5 Digital Modulations: BPSK, QPSK, 8PSK, QAM16, QAM64
    
    Rationale:
    - Digital modulations are the primary threat in drone C2 links
    - Removes high-variance analog signals (AM, FM) that confuse models
    - Creates a realistic "Drone Defense" classification scenario
    
    Returns dict with X_train, y_train, X_test_by_snr, etc.
    """
    if not os.path.exists(dataset_path):
        print("=" * 70)
        print("WARNING: RadioML 2016.10A Dataset Not Found")
        print("=" * 70)
        print(f"  Expected path: {os.path.abspath(dataset_path)}")
        print("  Falling back to SYNTHETIC DATA...")
        print("=" * 70)
        return None
    
    print("Loading RadioML 2016.10A (Tactical Subset)...")
    
    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    # Filter to tactical modulations only
    all_mods = sorted(set(mod for (mod, snr) in data_dict.keys()))
    tactical_mods = [m for m in TACTICAL_MODULATIONS if m in all_mods]
    
    if len(tactical_mods) != len(TACTICAL_MODULATIONS):
        missing = set(TACTICAL_MODULATIONS) - set(tactical_mods)
        print(f"  WARNING: Missing modulations: {missing}")
    
    snr_levels = sorted(set(snr for (mod, snr) in data_dict.keys()))
    
    print(f"  Tactical Modulations: {tactical_mods}")
    print(f"  SNR range: {min(snr_levels)}dB to {max(snr_levels)}dB")
    
    # Encode labels (tactical subset only)
    label_encoder = LabelEncoder()
    label_encoder.fit(tactical_mods)
    
    # Build training and test sets
    X_train_list = []
    y_train_list = []
    X_test_by_snr = {}
    y_test_by_snr = {}
    
    for snr in snr_levels:
        X_snr_list = []
        y_snr_list = []
        
        for mod in tactical_mods:
            if (mod, snr) in data_dict:
                samples = data_dict[(mod, snr)]  # (N, 2, 128)
                n_samples = samples.shape[0]
                samples_flat = samples.reshape(n_samples, -1)  # (N, 256)
                
                X_snr_list.append(samples_flat)
                y_snr_list.append([mod] * n_samples)
        
        if X_snr_list:
            X_snr = np.vstack(X_snr_list)
            y_snr = np.concatenate(y_snr_list)
            y_snr_encoded = label_encoder.transform(y_snr)
            
            X_test_by_snr[snr] = X_snr
            y_test_by_snr[snr] = y_snr_encoded
            
            if snr >= RADIOML_HIGH_SNR_THRESHOLD:
                X_train_list.append(X_snr)
                y_train_list.append(y_snr_encoded)
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Shuffle training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    print(f"  Training samples (SNR >= {RADIOML_HIGH_SNR_THRESHOLD}dB): {len(X_train)}")
    print(f"  Classes: {len(tactical_mods)} tactical modulations")
    print()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test_by_snr': X_test_by_snr,
        'y_test_by_snr': y_test_by_snr,
        'label_encoder': label_encoder,
        'snr_levels': snr_levels,
        'modulations': tactical_mods
    }


# =============================================================================
# SYNTHETIC TACTICAL DATASET (FALLBACK)
# =============================================================================

def generate_synthetic_tactical(n_samples_per_class=1000, feature_dim=256):
    """
    Generate synthetic dataset mimicking tactical modulation signatures.
    
    Used when RadioML is unavailable for quick verification.
    """
    print("Generating Synthetic Tactical Dataset...")
    
    rng = np.random.RandomState(2026)
    
    X_list = []
    y_list = []
    
    # Simulate 5 modulation classes with distinct spectral signatures
    for class_idx, mod_name in enumerate(TACTICAL_MODULATIONS):
        # Create class-specific frequency pattern
        base_freq = 5 + class_idx * 10  # Different center frequencies
        bandwidth = 3 + class_idx * 2   # Different bandwidths
        
        for _ in range(n_samples_per_class):
            # Generate IQ-like signal with class-specific characteristics
            t = np.linspace(0, 1, 128)
            phase = rng.uniform(0, 2 * np.pi)  # Random phase offset
            
            # I channel
            I = np.cos(2 * np.pi * base_freq * t + phase) * (1 + 0.3 * rng.randn(128))
            # Q channel  
            Q = np.sin(2 * np.pi * base_freq * t + phase) * (1 + 0.3 * rng.randn(128))
            
            # Add class-specific noise structure
            I += rng.randn(128) * 0.1 * (class_idx + 1)
            Q += rng.randn(128) * 0.1 * (class_idx + 1)
            
            sample = np.concatenate([I, Q])  # (256,)
            X_list.append(sample)
            y_list.append(class_idx)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Shuffle
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2026, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {len(TACTICAL_MODULATIONS)}")
    print()
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'modulations': TACTICAL_MODULATIONS,
        'mode': 'synthetic'
    }


# =============================================================================
# DSP PREPROCESSING (FROM V4)
# =============================================================================

def dsp_preprocess(X):
    """
    DSP Preprocessing: Convert raw IQ to FFT Magnitude Spectrum.
    
    Makes encoding shift-invariant (phase/time offset immune).
    
    Args:
        X: Raw IQ samples, shape (n_samples, 256)
        
    Returns:
        X_dsp: Magnitude spectrum, shape (n_samples, 128)
    """
    if X.shape[1] != 256:
        return X
    
    n_samples = X.shape[0]
    X_iq = X.reshape(n_samples, 2, 128)
    
    I = X_iq[:, 0, :]
    Q = X_iq[:, 1, :]
    
    signal = I + 1j * Q
    spectrum = np.fft.fft(signal, axis=1)
    magnitude = np.abs(spectrum)
    
    return magnitude


# =============================================================================
# V5 HDC ARCHITECTURE: RANDOM FOURIER FEATURES (RFF)
# =============================================================================

class HDCLearnerV5_RFF:
    """
    Hyperdimensional Computing Classifier V5 - Random Fourier Features.
    
    KEY INNOVATION: Non-Linear Encoding via RFF
    -------------------------------------------
    Traditional HDC: H = sign(X @ P)  → Linear decision boundaries
    
    RFF HDC: H = Concat(Cos(γ * X @ P), Sin(γ * X @ P))
    
    This approximates a Radial Basis Function (RBF) kernel:
        K(x, y) ≈ H(x) · H(y)
    
    Where γ (gamma) controls the kernel bandwidth.
    
    WHY RFF WORKS:
    --------------
    - Bochner's theorem: Any shift-invariant kernel can be approximated
      by random Fourier features
    - RBF kernel provides non-linear decision boundaries
    - Still maintains HDC's distributed representation benefits
    - Continuous output (floats) for better expressivity
    
    TRAINING: Iterative Perceptron-Style
    ------------------------------------
    Instead of one-shot bundling, we use error-driven updates:
    
    For each misclassified sample:
        Prototype[Target] += learning_rate * Encoded_X
        Prototype[Predicted] -= learning_rate * Encoded_X
    
    This actively pushes class boundaries apart.
    """
    
    def __init__(self, input_dim, dimensions=10000, num_classes=5,
                 gamma=0.1, use_dsp=True):
        """
        Initialize RFF-HDC learner.
        
        Args:
            input_dim: Raw input dimension (256 for RadioML IQ)
            dimensions: Hypervector dimension D (output will be 2*D due to cos/sin)
            num_classes: Number of classes
            gamma: RBF kernel bandwidth parameter
            use_dsp: Whether to apply FFT magnitude preprocessing
        """
        self.D = dimensions
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.use_dsp = use_dsp
        
        # DSP reduces 256→128
        self.dsp_output_dim = input_dim // 2 if use_dsp and input_dim == 256 else input_dim
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Random Fourier projection matrix: P ~ N(0, 2*gamma)
        # Factor of 2*gamma comes from RBF kernel approximation theory
        self.projection_matrix = np.random.randn(self.D, self.dsp_output_dim) * np.sqrt(2 * self.gamma)
        
        # Random bias for complete RFF formulation
        self.bias = np.random.uniform(0, 2 * np.pi, self.D)
        
        # Class prototypes (learned during training)
        self.class_prototypes = None
        self.class_labels = None
        
    def encode_rff(self, X):
        """
        Encode batch using Random Fourier Features.
        
        RFF Formula:
            z(x) = sqrt(2/D) * [cos(P @ x + b), sin(P @ x + b)]
        
        This provides an unbiased estimate of the RBF kernel:
            K(x, y) = exp(-gamma * ||x - y||^2) ≈ z(x) · z(y)
        
        Args:
            X: Input features, shape (n_samples, dsp_output_dim)
            
        Returns:
            H: RFF encoding, shape (n_samples, 2*D)
        """
        # Linear projection: (n_samples, D)
        projection = X @ self.projection_matrix.T + self.bias
        
        # Trigonometric features (non-linear)
        cos_features = np.cos(projection)
        sin_features = np.sin(projection)
        
        # Concatenate: (n_samples, 2*D)
        H = np.concatenate([cos_features, sin_features], axis=1)
        
        # Scale factor for unbiased kernel approximation
        H = H * np.sqrt(2.0 / self.D)
        
        return H.astype(np.float32)
    
    def train(self, X, y, epochs=ITERATIVE_EPOCHS, lr=ITERATIVE_LEARNING_RATE,
              verbose=True):
        """
        Train using iterative Perceptron-style updates.
        
        ALGORITHM:
        ----------
        1. Initialize all prototypes to zero
        2. For each epoch:
           a. Shuffle training data
           b. For each sample (x, target):
              - Encode: h = RFF(x)
              - Predict: pred = argmax(h · prototypes)
              - If pred != target:
                  prototype[target] += lr * h
                  prototype[pred] -= lr * h
        3. Normalize prototypes (optional)
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training passes
            lr: Learning rate for updates
            verbose: Print progress
        """
        # DSP preprocessing
        if self.use_dsp:
            X_dsp = dsp_preprocess(X)
        else:
            X_dsp = X
        
        # Fit and apply scaler
        X_scaled = self.scaler.fit_transform(X_dsp)
        self.is_fitted = True
        
        # Determine classes
        self.class_labels = sorted(np.unique(y))
        n_classes = len(self.class_labels)
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        
        # Initialize prototypes to zero: shape (n_classes, 2*D)
        self.class_prototypes = np.zeros((n_classes, 2 * self.D), dtype=np.float32)
        
        # Encode all training samples once (for efficiency)
        H_all = self.encode_rff(X_scaled)
        n_samples = len(X)
        
        if verbose:
            print(f"    Iterative training: {epochs} epochs, {n_samples} samples")
        
        for epoch in range(epochs):
            # Shuffle training order
            shuffle_idx = np.random.permutation(n_samples)
            errors = 0
            
            for i in shuffle_idx:
                h = H_all[i]  # (2*D,)
                target = y[i]
                target_idx = label_to_idx[target]
                
                # Predict via dot product similarity
                similarities = self.class_prototypes @ h  # (n_classes,)
                pred_idx = np.argmax(similarities)
                
                # Update on error (Perceptron rule)
                if pred_idx != target_idx:
                    self.class_prototypes[target_idx] += lr * h
                    self.class_prototypes[pred_idx] -= lr * h
                    errors += 1
            
            accuracy = 1.0 - errors / n_samples
            if verbose and (epoch + 1) % 2 == 0:
                print(f"      Epoch {epoch + 1}/{epochs}: Acc={accuracy * 100:.1f}%")
        
        if verbose:
            print(f"    Training complete. Final accuracy: {accuracy * 100:.1f}%")
    
    def train_single_pass(self, X, y):
        """
        One-shot training (bundling) for comparison with iterative.
        
        Useful for Day 0 Adaptation benchmark.
        """
        if self.use_dsp:
            X_dsp = dsp_preprocess(X)
        else:
            X_dsp = X
        
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_dsp)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X_dsp)
        
        H_all = self.encode_rff(X_scaled)
        
        self.class_labels = sorted(np.unique(y))
        n_classes = len(self.class_labels)
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        
        self.class_prototypes = np.zeros((n_classes, 2 * self.D), dtype=np.float32)
        
        for i in range(len(X)):
            target_idx = label_to_idx[y[i]]
            self.class_prototypes[target_idx] += H_all[i]
    
    def add_class_samples(self, X_new, y_new_label):
        """
        Add samples for a new class (Day 0 Adaptation).
        
        HDC advantage: Can add new class without retraining existing prototypes.
        
        Args:
            X_new: Samples of the new class
            y_new_label: Label for the new class
        """
        if self.use_dsp:
            X_dsp = dsp_preprocess(X_new)
        else:
            X_dsp = X_new
        
        X_scaled = self.scaler.transform(X_dsp)
        H_new = self.encode_rff(X_scaled)
        
        # Bundle new class prototype
        new_prototype = np.sum(H_new, axis=0)
        
        # Add to model
        self.class_labels.append(y_new_label)
        self.class_prototypes = np.vstack([self.class_prototypes, new_prototype])
    
    def predict(self, X):
        """Classify samples by finding nearest prototype."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction.")
        
        if self.use_dsp:
            X_dsp = dsp_preprocess(X)
        else:
            X_dsp = X
        
        X_scaled = self.scaler.transform(X_dsp)
        H = self.encode_rff(X_scaled)
        
        # Similarity via dot product
        similarities = H @ self.class_prototypes.T  # (n_samples, n_classes)
        pred_indices = np.argmax(similarities, axis=1)
        predictions = np.array([self.class_labels[i] for i in pred_indices])
        
        return predictions
    
    def accuracy(self, X, y):
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# =============================================================================
# STEEL MAN MLP (FROM V4)
# =============================================================================

class SteelManMLP:
    """Steel Man MLP with adversarial training (from V4)."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            random_state=2026,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        self.is_trained = False
        self.adversarial_noise_var = 1.0
        
    def _generate_noisy_samples(self, X, noise_variance, rng=None):
        if rng is None:
            rng = np.random
        noise_std = np.sqrt(noise_variance)
        noise = rng.randn(*X.shape) * noise_std
        return X + noise
    
    def train(self, X, y, adversarial=True):
        training_rng = np.random.RandomState(2026)
        
        if adversarial:
            X_noisy = self._generate_noisy_samples(X, self.adversarial_noise_var, rng=training_rng)
            X_augmented = np.vstack([X, X_noisy])
            y_augmented = np.concatenate([y, y])
            shuffle_idx = training_rng.permutation(len(X_augmented))
            X_augmented = X_augmented[shuffle_idx]
            y_augmented = y_augmented[shuffle_idx]
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X_augmented)
        else:
            X_scaled = self.scaler.fit_transform(X)
            y_augmented = y
        
        self.model.fit(X_scaled, y_augmented)
        self.is_trained = True
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# =============================================================================
# BENCHMARK SUITE: "DRONE BATTERY"
# =============================================================================

def benchmark_training_time(model_class, X_train, y_train, model_kwargs=None,
                            train_kwargs=None, n_runs=3):
    """
    Benchmark: Training Energy (wall-clock time).
    
    Returns average training time in seconds.
    """
    times = []
    
    for _ in range(n_runs):
        if model_kwargs:
            model = model_class(**model_kwargs)
        else:
            model = model_class()
        
        start = time.perf_counter()
        if train_kwargs:
            model.train(X_train, y_train, **train_kwargs)
        else:
            model.train(X_train, y_train)
        end = time.perf_counter()
        
        times.append(end - start)
    
    return np.mean(times), np.std(times)


def benchmark_inference_latency(model, X_test, n_samples=1000):
    """
    Benchmark: Inference Latency (μs per sample).
    
    Measures single-sample inference time averaged over n_samples.
    """
    # Select random subset
    if len(X_test) > n_samples:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_subset = X_test[indices]
    else:
        X_subset = X_test
    
    # Warm-up
    _ = model.predict(X_subset[:10])
    
    # Single-sample latency measurement
    latencies = []
    for i in range(len(X_subset)):
        x = X_subset[i:i+1]
        start = time.perf_counter()
        _ = model.predict(x)
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # Convert to μs
    
    return np.mean(latencies), np.std(latencies)


def benchmark_batch_inference(model, X_test, batch_sizes=[1, 10, 100, 1000]):
    """
    Benchmark: Batch inference throughput.
    
    Returns dict mapping batch_size → samples/second.
    """
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue
        
        n_batches = min(100, len(X_test) // batch_size)
        
        start = time.perf_counter()
        for i in range(n_batches):
            _ = model.predict(X_test[i*batch_size:(i+1)*batch_size])
        end = time.perf_counter()
        
        total_samples = n_batches * batch_size
        throughput = total_samples / (end - start)
        results[batch_size] = throughput
    
    return results


def benchmark_day0_adaptation(hdc_model_class, mlp_model_class, X_train, y_train,
                               X_new_class, y_new_class_label, X_test_new,
                               model_kwargs=None, target_accuracy=0.8):
    """
    Benchmark: Day 0 Adaptation Rate.
    
    Scenario: System has learned N-1 classes. A new class appears.
    How quickly can each model learn the new class to target accuracy?
    
    HDC: Add samples to new prototype (near-instant)
    MLP: Requires full retraining (slow)
    
    Returns dict with adaptation times and sample counts.
    """
    results = {}
    
    # Get unique classes excluding the new one
    existing_classes = sorted(set(y_train))
    if y_new_class_label in existing_classes:
        existing_classes.remove(y_new_class_label)
    
    # Filter training data to existing classes only
    mask = np.isin(y_train, existing_classes)
    X_existing = X_train[mask]
    y_existing = y_train[mask]
    
    # =========================================================================
    # HDC Adaptation
    # =========================================================================
    # Step 1: Train on existing classes
    if model_kwargs:
        hdc = hdc_model_class(**model_kwargs)
    else:
        hdc = hdc_model_class(input_dim=X_train.shape[1])
    
    hdc.train(X_existing, y_existing, verbose=False)
    
    # Step 2: Add new class incrementally
    sample_counts = [10, 25, 50, 100, 200]
    hdc_adaptation = []
    
    for n_samples in sample_counts:
        if n_samples > len(X_new_class):
            break
        
        # Clone model state (re-train from scratch for fair comparison)
        if model_kwargs:
            hdc_test = hdc_model_class(**model_kwargs)
        else:
            hdc_test = hdc_model_class(input_dim=X_train.shape[1])
        hdc_test.train(X_existing, y_existing, verbose=False)
        
        # Time the adaptation
        start = time.perf_counter()
        hdc_test.add_class_samples(X_new_class[:n_samples], y_new_class_label)
        adapt_time = time.perf_counter() - start
        
        # Test accuracy on new class
        acc = hdc_test.accuracy(X_test_new, 
                                np.full(len(X_test_new), y_new_class_label))
        
        hdc_adaptation.append({
            'samples': n_samples,
            'time_ms': adapt_time * 1000,
            'accuracy': acc
        })
        
        if acc >= target_accuracy:
            break
    
    results['hdc'] = hdc_adaptation
    
    # =========================================================================
    # MLP Adaptation (requires full retraining)
    # =========================================================================
    mlp_adaptation = []
    
    for n_samples in sample_counts:
        if n_samples > len(X_new_class):
            break
        
        # Combine existing + new class samples
        X_combined = np.vstack([X_existing, X_new_class[:n_samples]])
        y_combined = np.concatenate([y_existing, 
                                      np.full(n_samples, y_new_class_label)])
        
        mlp = mlp_model_class()
        
        # Time full retraining
        start = time.perf_counter()
        mlp.train(X_combined, y_combined, adversarial=False)  # No adversarial for speed
        adapt_time = time.perf_counter() - start
        
        # Test accuracy on new class
        acc = mlp.accuracy(X_test_new, 
                           np.full(len(X_test_new), y_new_class_label))
        
        mlp_adaptation.append({
            'samples': n_samples,
            'time_ms': adapt_time * 1000,
            'accuracy': acc
        })
        
        if acc >= target_accuracy:
            break
    
    results['mlp'] = mlp_adaptation
    
    return results


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_full_benchmark():
    """
    Run the complete "Drone Battery" benchmark suite.
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  VICEROY 2026 SYMPOSIUM - VERSION 5".center(68) + "║")
    print("║" + "  'DRONE BATTERY' BENCHMARK SUITE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  HDC-RFF vs Steel Man MLP".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # =========================================================================
    # Load Dataset
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Loading Tactical Dataset")
    print("=" * 70)
    
    data = load_radioml_tactical()
    
    if data is None:
        data = generate_synthetic_tactical()
        use_radioml = False
    else:
        use_radioml = True
    
    X_train = data['X_train']
    y_train = data['y_train']
    modulations = data['modulations']
    
    if use_radioml:
        X_test_by_snr = data['X_test_by_snr']
        y_test_by_snr = data['y_test_by_snr']
        snr_levels = data['snr_levels']
        # Use +18dB as clean test set
        X_test = X_test_by_snr[18]
        y_test = y_test_by_snr[18]
    else:
        X_test = data['X_test']
        y_test = data['y_test']
        snr_levels = None
    
    feature_dim = X_train.shape[1]
    num_classes = len(modulations)
    
    print(f"  Dataset mode: {'RadioML Tactical' if use_radioml else 'Synthetic'}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Classes: {num_classes} ({modulations})")
    print()
    
    # =========================================================================
    # BENCHMARK A: Training Time
    # =========================================================================
    print("=" * 70)
    print("BENCHMARK A: Training Energy (Wall-Clock Time)")
    print("=" * 70)
    print()
    
    hdc_kwargs = {
        'input_dim': feature_dim,
        'dimensions': D,
        'num_classes': num_classes,
        'gamma': RFF_GAMMA,
        'use_dsp': True
    }
    
    hdc_train_kwargs = {'epochs': ITERATIVE_EPOCHS, 'verbose': False}
    
    print("  Training HDC-RFF (3 runs)...")
    hdc_time_mean, hdc_time_std = benchmark_training_time(
        HDCLearnerV5_RFF, X_train, y_train,
        model_kwargs=hdc_kwargs, train_kwargs=hdc_train_kwargs
    )
    print(f"    HDC-RFF: {hdc_time_mean:.2f} ± {hdc_time_std:.2f} seconds")
    
    print("  Training Steel Man MLP (3 runs)...")
    mlp_time_mean, mlp_time_std = benchmark_training_time(
        SteelManMLP, X_train, y_train, train_kwargs={'adversarial': True}
    )
    print(f"    Steel Man MLP: {mlp_time_mean:.2f} ± {mlp_time_std:.2f} seconds")
    
    print()
    print(f"  ⚡ HDC is {mlp_time_mean / hdc_time_mean:.1f}× faster to train")
    print()
    
    # =========================================================================
    # Train Final Models for Remaining Benchmarks
    # =========================================================================
    print("=" * 70)
    print("PHASE 2: Training Final Models")
    print("=" * 70)
    print()
    
    print("  Training HDC-RFF (V5)...")
    hdc = HDCLearnerV5_RFF(**hdc_kwargs)
    hdc.train(X_train, y_train, epochs=ITERATIVE_EPOCHS, verbose=True)
    
    print()
    print("  Training Steel Man MLP...")
    mlp = SteelManMLP()
    mlp.train(X_train, y_train, adversarial=True)
    print("    MLP training complete.")
    print()
    
    # =========================================================================
    # BENCHMARK B: Inference Latency
    # =========================================================================
    print("=" * 70)
    print("BENCHMARK B: Inference Latency (μs per sample)")
    print("=" * 70)
    print()
    
    hdc_latency_mean, hdc_latency_std = benchmark_inference_latency(hdc, X_test)
    mlp_latency_mean, mlp_latency_std = benchmark_inference_latency(mlp, X_test)
    
    print(f"  HDC-RFF:        {hdc_latency_mean:.1f} ± {hdc_latency_std:.1f} μs/sample")
    print(f"  Steel Man MLP:  {mlp_latency_mean:.1f} ± {mlp_latency_std:.1f} μs/sample")
    print()
    print(f"  ⚡ HDC is {mlp_latency_mean / hdc_latency_mean:.1f}× faster inference")
    print()
    
    # Batch throughput
    print("  Batch Throughput (samples/second):")
    hdc_throughput = benchmark_batch_inference(hdc, X_test)
    mlp_throughput = benchmark_batch_inference(mlp, X_test)
    
    print(f"    {'Batch':<10} {'HDC-RFF':<15} {'Steel MLP':<15} {'Ratio':<10}")
    print(f"    {'-'*8:<10} {'-'*13:<15} {'-'*13:<15} {'-'*8:<10}")
    for batch_size in sorted(hdc_throughput.keys()):
        hdc_tp = hdc_throughput[batch_size]
        mlp_tp = mlp_throughput.get(batch_size, 0)
        ratio = hdc_tp / mlp_tp if mlp_tp > 0 else float('inf')
        print(f"    {batch_size:<10} {hdc_tp:<15.0f} {mlp_tp:<15.0f} {ratio:<10.1f}×")
    print()
    
    # =========================================================================
    # BENCHMARK C: Classification Accuracy
    # =========================================================================
    print("=" * 70)
    print("BENCHMARK C: Classification Accuracy")
    print("=" * 70)
    print()
    
    if use_radioml:
        print("  Accuracy by SNR Level:")
        print(f"    {'SNR (dB)':<10} {'HDC-RFF':<12} {'Steel MLP':<12} {'Δ':<10}")
        print(f"    {'-'*8:<10} {'-'*10:<12} {'-'*10:<12} {'-'*8:<10}")
        
        hdc_acc_by_snr = {}
        mlp_acc_by_snr = {}
        
        for snr in snr_levels:
            X_snr = X_test_by_snr[snr]
            y_snr = y_test_by_snr[snr]
            
            hdc_acc = hdc.accuracy(X_snr, y_snr)
            mlp_acc = mlp.accuracy(X_snr, y_snr)
            
            hdc_acc_by_snr[snr] = hdc_acc
            mlp_acc_by_snr[snr] = mlp_acc
            
            delta = hdc_acc - mlp_acc
            delta_str = f"+{delta*100:.1f}%" if delta >= 0 else f"{delta*100:.1f}%"
            
            print(f"    {snr:<10} {hdc_acc*100:<12.1f} {mlp_acc*100:<12.1f} {delta_str:<10}")
        
        # Summary statistics
        high_snr = [s for s in snr_levels if s >= 10]
        low_snr = [s for s in snr_levels if s <= 0]
        
        hdc_high = np.mean([hdc_acc_by_snr[s] for s in high_snr])
        mlp_high = np.mean([mlp_acc_by_snr[s] for s in high_snr])
        hdc_low = np.mean([hdc_acc_by_snr[s] for s in low_snr])
        mlp_low = np.mean([mlp_acc_by_snr[s] for s in low_snr])
        
        print()
        print(f"  Summary (Tactical Subset - 5 Digital Modulations):")
        print(f"    High-SNR (≥+10dB):  HDC={hdc_high*100:.1f}%, MLP={mlp_high*100:.1f}%")
        print(f"    Low-SNR (≤0dB):     HDC={hdc_low*100:.1f}%, MLP={mlp_low*100:.1f}%")
        print(f"    HDC Degradation:    {(hdc_high - hdc_low)*100:.1f}%")
        print(f"    MLP Degradation:    {(mlp_high - mlp_low)*100:.1f}%")
    else:
        hdc_acc = hdc.accuracy(X_test, y_test)
        mlp_acc = mlp.accuracy(X_test, y_test)
        print(f"    HDC-RFF:        {hdc_acc*100:.1f}%")
        print(f"    Steel Man MLP:  {mlp_acc*100:.1f}%")
    
    print()
    
    # =========================================================================
    # BENCHMARK D: Day 0 Adaptation
    # =========================================================================
    print("=" * 70)
    print("BENCHMARK D: Day 0 Adaptation (Learning New Class)")
    print("=" * 70)
    print()
    print("  Scenario: System trained on 4 classes, learns 5th class incrementally.")
    print()
    
    # Prepare data: Use last class as "new threat"
    unique_classes = sorted(np.unique(y_train))
    new_class_label = unique_classes[-1]
    
    # Filter data
    new_class_mask = y_train == new_class_label
    X_new_class = X_train[new_class_mask]
    
    test_mask = y_test == new_class_label
    X_test_new = X_test[test_mask]
    
    if len(X_new_class) > 0 and len(X_test_new) > 0:
        adaptation_results = benchmark_day0_adaptation(
            HDCLearnerV5_RFF, SteelManMLP,
            X_train, y_train,
            X_new_class, new_class_label, X_test_new,
            model_kwargs=hdc_kwargs,
            target_accuracy=0.8
        )
        
        print(f"  HDC-RFF Adaptation (new class = {modulations[new_class_label]}):")
        print(f"    {'Samples':<10} {'Time (ms)':<12} {'Accuracy':<12}")
        print(f"    {'-'*8:<10} {'-'*10:<12} {'-'*10:<12}")
        for entry in adaptation_results['hdc']:
            print(f"    {entry['samples']:<10} {entry['time_ms']:<12.2f} {entry['accuracy']*100:<12.1f}%")
        
        print()
        print(f"  Steel Man MLP Adaptation (requires full retraining):")
        print(f"    {'Samples':<10} {'Time (ms)':<12} {'Accuracy':<12}")
        print(f"    {'-'*8:<10} {'-'*10:<12} {'-'*10:<12}")
        for entry in adaptation_results['mlp']:
            print(f"    {entry['samples']:<10} {entry['time_ms']:<12.2f} {entry['accuracy']*100:<12.1f}%")
        
        # Compare time to reach 80% accuracy
        hdc_time_to_80 = next((e['time_ms'] for e in adaptation_results['hdc'] 
                               if e['accuracy'] >= 0.8), None)
        mlp_time_to_80 = next((e['time_ms'] for e in adaptation_results['mlp'] 
                               if e['accuracy'] >= 0.8), None)
        
        if hdc_time_to_80 and mlp_time_to_80:
            print()
            print(f"  ⚡ Time to 80% accuracy on new class:")
            print(f"      HDC-RFF:  {hdc_time_to_80:.2f} ms")
            print(f"      MLP:      {mlp_time_to_80:.2f} ms")
            print(f"      HDC is {mlp_time_to_80 / hdc_time_to_80:.0f}× faster for adaptation")
    else:
        print("  Insufficient data for adaptation benchmark.")
    
    print()
    
    # =========================================================================
    # Generate Visualization
    # =========================================================================
    if use_radioml:
        print("=" * 70)
        print("PHASE 3: Generating Visualization")
        print("=" * 70)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
        
        # Plot 1: Accuracy vs SNR
        ax1 = axes[0, 0]
        ax1.plot(snr_levels, [hdc_acc_by_snr[s]*100 for s in snr_levels],
                 'o-', linewidth=2, markersize=6, color='#2E86AB',
                 label='HDC-RFF (V5)')
        ax1.plot(snr_levels, [mlp_acc_by_snr[s]*100 for s in snr_levels],
                 's-', linewidth=2, markersize=6, color='#E94F37',
                 label='Steel Man MLP')
        ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.7)
        ax1.axvline(x=0, color='orange', linestyle='--', alpha=0.5)
        ax1.set_xlabel('SNR (dB)', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Classification Accuracy vs SNR', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Plot 2: Training Time Comparison
        ax2 = axes[0, 1]
        models = ['HDC-RFF', 'Steel MLP']
        times = [hdc_time_mean, mlp_time_mean]
        colors = ['#2E86AB', '#E94F37']
        bars = ax2.bar(models, times, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Training Time (seconds)', fontweight='bold')
        ax2.set_title('Training Energy Comparison', fontweight='bold')
        for bar, t in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Inference Latency
        ax3 = axes[1, 0]
        latencies = [hdc_latency_mean, mlp_latency_mean]
        bars = ax3.bar(models, latencies, color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Latency (μs/sample)', fontweight='bold')
        ax3.set_title('Inference Latency Comparison', fontweight='bold')
        for bar, lat in zip(bars, latencies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{lat:.0f}μs', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Batch Throughput
        ax4 = axes[1, 1]
        batch_sizes = sorted(hdc_throughput.keys())
        hdc_tp = [hdc_throughput[b] for b in batch_sizes]
        mlp_tp = [mlp_throughput.get(b, 0) for b in batch_sizes]
        x = np.arange(len(batch_sizes))
        width = 0.35
        ax4.bar(x - width/2, hdc_tp, width, label='HDC-RFF', color='#2E86AB')
        ax4.bar(x + width/2, mlp_tp, width, label='Steel MLP', color='#E94F37')
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(b) for b in batch_sizes])
        ax4.set_xlabel('Batch Size', fontweight='bold')
        ax4.set_ylabel('Throughput (samples/sec)', fontweight='bold')
        ax4.set_title('Batch Inference Throughput', fontweight='bold')
        ax4.legend()
        ax4.set_yscale('log')
        
        plt.suptitle('VICEROY 2026 V5: HDC-RFF vs Steel Man MLP\n'
                     '(Tactical Subset: 5 Digital Modulations)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig('viceroy_2026_v5_benchmark.png', dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.savefig('viceroy_2026_v5_benchmark.pdf', format='pdf', bbox_inches='tight',
                    facecolor='white')
        print("  ✓ Saved: viceroy_2026_v5_benchmark.png")
        print("  ✓ Saved: viceroy_2026_v5_benchmark.pdf")
        plt.close()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  BENCHMARK SUITE COMPLETE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Summary for Symposium Paper:")
    print("-" * 50)
    print(f"  Training Speed:     HDC is {mlp_time_mean / hdc_time_mean:.1f}× faster")
    print(f"  Inference Latency:  HDC is {mlp_latency_mean / hdc_latency_mean:.1f}× faster")
    if use_radioml:
        print(f"  High-SNR Accuracy:  HDC={hdc_high*100:.1f}% vs MLP={mlp_high*100:.1f}%")
        print(f"  Low-SNR Accuracy:   HDC={hdc_low*100:.1f}% vs MLP={mlp_low*100:.1f}%")
        print(f"  Resilience:         HDC degrades {(hdc_high-hdc_low)*100:.1f}% vs MLP {(mlp_high-mlp_low)*100:.1f}%")
    if hdc_time_to_80 and mlp_time_to_80:
        print(f"  Adaptation:         HDC is ~{mlp_time_to_80/hdc_time_to_80:.0f}× faster for new class")
    else:
        print(f"  Adaptation:         Neither reached 80% target (requires more samples)")
    print()
    print("Hardware Compatibility: CPU-only, Python 3.12, scikit-learn")
    print("Ready for deployment to Tower 2 (Drone Hardware).")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_full_benchmark()
