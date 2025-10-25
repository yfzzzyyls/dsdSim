#!/usr/bin/env python3
"""Profile the acceptance model in detail to find the real bottleneck."""

import time
import sys
import os
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from acceptance.regressor import AcceptanceRegressor

def profile_model_loading():
    """Profile model loading and prediction separately."""

    model_path = REPO_ROOT / "src" / "acceptance" / "llama2_7b_vs_70b.joblib"

    print("=" * 80)
    print("PROFILING MODEL LOADING")
    print("=" * 80)

    # Time model loading
    start = time.perf_counter()
    model = AcceptanceRegressor.from_file(str(model_path))
    load_time = (time.perf_counter() - start) * 1000
    print(f"Model loading time: {load_time:.1f}ms")

    print("\n" + "=" * 80)
    print("PROFILING PREDICTIONS")
    print("=" * 80)

    # Test different scenarios
    contexts = [500, 1000, 1500]
    depths = [2, 4, 6, 8]

    # Feature context for testing
    feature_context = {
        'pending_tokens': 100,
        'queue_depth': 10
    }

    print("\n1. Testing predict_expected_accepts (regressor):")
    print("-" * 40)

    for context in contexts:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            expected, _ = model.predict_expected_accepts(
                float(context),
                feature_context=feature_context
            )
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"  Context={context}: {avg_time:.2f}ms (avg of 50 calls)")

    print("\n2. Testing position_probabilities (classifier):")
    print("-" * 40)

    for depth in depths:
        times = []
        for _ in range(50):
            start = time.perf_counter()
            probs = model.position_probabilities(
                context_length=1000.0,
                depth=depth,
                feature_context=feature_context
            )
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"  Depth={depth}: {avg_time:.2f}ms (avg of 50 calls)")
        print(f"    - Per position: {avg_time/depth:.2f}ms")

    print("\n3. Testing batched vs sequential predictions:")
    print("-" * 40)

    # Get the classifier directly
    classifier = model._select_classifier(feature_context)

    # Get the correct number of features from the model
    n_features = 2  # Based on error message
    if hasattr(classifier, 'n_features_in_'):
        n_features = classifier.n_features_in_

    # Create sample data for 8 positions
    sample_data = np.random.randn(8, n_features).astype(np.float32)

    # Time batched prediction
    times_batched = []
    for _ in range(100):
        start = time.perf_counter()
        if hasattr(classifier, "predict_proba"):
            classifier.predict_proba(sample_data)
        times_batched.append((time.perf_counter() - start) * 1000)

    # Time sequential predictions
    times_sequential = []
    for _ in range(100):
        start = time.perf_counter()
        for row in sample_data:
            if hasattr(classifier, "predict_proba"):
                classifier.predict_proba(row.reshape(1, -1))
        times_sequential.append((time.perf_counter() - start) * 1000)

    print(f"  Batched (8 positions): {np.mean(times_batched):.2f}ms")
    print(f"  Sequential (8×1 position): {np.mean(times_sequential):.2f}ms")
    print(f"  Speedup from batching: {np.mean(times_sequential)/np.mean(times_batched):.1f}x")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Check model complexity
    if hasattr(model, '_regressor'):
        regressor = model._regressor
        print(f"\nRegressor type: {type(regressor).__name__}")
        if hasattr(regressor, 'n_estimators'):
            print(f"  - Number of trees: {regressor.n_estimators}")
        if hasattr(regressor, 'max_depth'):
            print(f"  - Max depth: {regressor.max_depth}")

    if hasattr(classifier, 'n_estimators'):
        print(f"\nClassifier type: {type(classifier).__name__}")
        print(f"  - Number of trees: {classifier.n_estimators}")
        if hasattr(classifier, 'max_depth'):
            print(f"  - Max depth: {classifier.max_depth}")

if __name__ == "__main__":
    profile_model_loading()