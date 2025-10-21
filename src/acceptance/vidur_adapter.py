"""Adapter to use VIDUR's fast RandomForest models for acceptance prediction."""

import pickle
import numpy as np
from typing import List, Optional, Tuple, Any, Mapping
import joblib


class VidurAcceptanceAdapter:
    """Use VIDUR's execution time model as acceptance predictor."""

    def __init__(self, vidur_model_path: str = 'data/vidur/cache/model_cache/mlp_up_proj_3f09af7d.pkl'):
        """Load VIDUR model and wrap it for acceptance prediction."""
        with open(vidur_model_path, 'rb') as f:
            self.vidur_model = pickle.load(f)

        # Normalization parameters
        self.baseline_exec_time = 10.0  # Baseline execution time in ms
        self.decay_factor = 0.95  # Decay per position

        print(f"Loaded VIDUR model: {self.vidur_model.n_estimators} trees, depth={self.vidur_model.max_depth}")

    def position_probabilities(
        self,
        *,
        context_length: float,
        depth: int,
        default: Optional[float] = None,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """Convert VIDUR execution time to acceptance probabilities."""
        if depth == 0:
            return []

        # Use context length as feature (VIDUR uses 1 feature)
        # Scale context to reasonable range for execution time model
        scaled_context = context_length / 1000.0  # Scale to [0.1, 2.0] range

        # Get execution time prediction
        X = np.array([[scaled_context]], dtype=np.float32)
        exec_time = self.vidur_model.predict(X)[0]

        # Convert execution time to base acceptance probability
        # Lower execution time = higher acceptance
        base_prob = 1.0 / (1.0 + exec_time / self.baseline_exec_time)
        base_prob = max(0.3, min(0.95, base_prob))  # Clamp to reasonable range

        # Apply position-based decay
        probs = []
        for i in range(depth):
            prob = base_prob * (self.decay_factor ** i)
            probs.append(max(0.1, min(0.9, prob)))

        return probs

    def predict_expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, bool]:
        """Predict expected number of accepted tokens."""
        probs = self.position_probabilities(
            context_length=context_length,
            depth=8,
            feature_context=feature_context
        )
        expected = sum(probs)
        return expected, False


def create_vidur_acceptance_model():
    """Create and save a VIDUR-based acceptance model."""

    print("Creating VIDUR-based acceptance model...")
    print("=" * 60)

    # Load VIDUR model
    adapter = VidurAcceptanceAdapter()

    # Test speed
    import time
    times = []
    for i in range(100):
        start = time.perf_counter()
        probs = adapter.position_probabilities(
            context_length=1000.0 + i,
            depth=8
        )
        times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    print(f"\nSpeed test: {avg_time:.2f}ms")
    print(f"Expected speedup: {28.0/avg_time:.1f}x vs original model")

    # Save as joblib format compatible with AcceptanceRegressor
    original = joblib.load('src/acceptance/llama2_7b_vs_70b.joblib')

    # Create a wrapper that looks like our model format
    model_dict = {
        'classifier': adapter.vidur_model,  # Use VIDUR model directly
        'regressor': adapter.vidur_model,   # Same model for both
        'metadata': {
            'name': 'vidur_adapter',
            'type': 'vidur',
            'original_model': 'mlp_up_proj'
        },
        'feature_columns': original.get('feature_columns', {}),
        'spec_tokens': 8,
        '_adapter': adapter  # Keep adapter for custom logic
    }

    joblib.dump(model_dict, 'src/acceptance/vidur_acceptance.joblib')
    print("\nSaved to: src/acceptance/vidur_acceptance.joblib")

    return adapter


if __name__ == "__main__":
    create_vidur_acceptance_model()