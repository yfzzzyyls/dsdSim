#!/usr/bin/env python3
"""Profile the acceptance model performance to understand the slowdown."""

import time
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Load the acceptance model
import joblib
import numpy as np

def profile_acceptance_model():
    """Profile the acceptance model calls."""

    # Load the model
    model_path = REPO_ROOT / "src" / "acceptance" / "llama2_7b_vs_70b.joblib"
    print(f"Loading model from: {model_path}")

    model_data = joblib.load(model_path)
    print(f"Model type: {type(model_data)}")
    print(f"Model keys: {model_data.keys() if hasattr(model_data, 'keys') else 'N/A'}")

    # Simulate acceptance lookups
    depths = [1, 2, 4, 8]
    contexts = [100, 500, 1000, 2000]

    print("\n" + "="*80)
    print("TIMING ACCEPTANCE MODEL CALLS")
    print("="*80)

    # Test regressor if available
    if 'regressor' in model_data:
        regressor = model_data['regressor']
        print(f"\nRegressor type: {type(regressor)}")

        # Create sample features (you'll need to adjust based on actual feature shape)
        # Typically: [context_length, depth, pending_tokens, queue_depth, etc.]
        sample_features = np.array([[1000.0, 4.0, 100.0, 10.0, 0.5, 0.7]])

        # Warm up
        for _ in range(10):
            regressor.predict(sample_features)

        # Time predictions
        times = []
        for _ in range(100):
            start = time.perf_counter()
            regressor.predict(sample_features)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"Regressor prediction time: {avg_time:.3f}ms (avg over 100 calls)")
        print(f"  Min: {np.min(times):.3f}ms, Max: {np.max(times):.3f}ms")

    # Test classifier if available
    if 'classifier' in model_data:
        classifier = model_data['classifier']
        print(f"\nClassifier type: {type(classifier)}")

        sample_features = np.array([[1000.0, 4.0, 100.0, 10.0, 0.5, 0.7]])

        # Warm up
        for _ in range(10):
            classifier.predict_proba(sample_features)

        # Time predictions
        times = []
        for _ in range(100):
            start = time.perf_counter()
            classifier.predict_proba(sample_features)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        print(f"Classifier prediction time: {avg_time:.3f}ms (avg over 100 calls)")
        print(f"  Min: {np.min(times):.3f}ms, Max: {np.max(times):.3f}ms")

    print("\n" + "="*80)
    print("ESTIMATING TOTAL OVERHEAD")
    print("="*80)

    # Estimate for a typical simulation
    drafts = 800
    conversations_per_draft = 1  # Simplified
    calls_per_conversation = 1   # For Spec++ gamma selection

    total_calls = drafts * conversations_per_draft * calls_per_conversation

    if 'regressor' in model_data and 'classifier' in model_data:
        time_per_call = np.mean(times) * 2  # Both models
        total_time_seconds = (total_calls * time_per_call) / 1000

        print(f"Estimated calls per simulation: {total_calls}")
        print(f"Time per acceptance lookup: {time_per_call:.3f}ms")
        print(f"Total acceptance overhead: {total_time_seconds:.1f} seconds")
        print(f"\nWithout caching, this adds {total_time_seconds:.1f}s to simulation time!")

if __name__ == "__main__":
    profile_acceptance_model()