"""Fast acceptance model using a smaller RandomForest or simple heuristics."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import time
from typing import Dict, List, Optional, Tuple, Any, Mapping


class FastAcceptanceModel:
    """Fast acceptance model that prioritizes speed over accuracy."""

    def __init__(self):
        # Simple heuristic: acceptance decreases with depth and context
        self.base_rate = 0.8
        self.context_penalty = 0.00001  # Small penalty for longer contexts
        self.depth_decay = 0.92  # Each position is 92% as likely as previous

    def position_probabilities(
        self,
        *,
        context_length: float,
        depth: int,
        default: Optional[float] = None,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """Fast heuristic-based acceptance probabilities."""
        if depth == 0:
            return []

        # Base rate adjusted for context length
        context_factor = max(0.5, 1.0 - self.context_penalty * context_length)
        base = self.base_rate * context_factor

        # Decay with depth
        probs = []
        for i in range(depth):
            prob = base * (self.depth_decay ** i)
            probs.append(max(0.1, min(0.95, prob)))

        return probs

    def predict_expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, bool]:
        """Fast expected accepts prediction."""
        probs = self.position_probabilities(
            context_length=context_length,
            depth=8,
            feature_context=feature_context
        )
        expected = sum(probs)
        return expected, False


class LightweightRandomForestAcceptance:
    """Lightweight RandomForest trained for speed."""

    def __init__(self, n_trees=50, max_depth=8):
        # Much smaller, faster model
        self.classifier = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features='sqrt',
            n_jobs=1,  # Single thread is often faster for small models
            random_state=42
        )
        self.regressor = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            max_features='sqrt',
            n_jobs=1,
            random_state=42
        )
        self._trained = False

    def train_from_existing(self, existing_model_path: str):
        """Train a smaller model by sampling from existing model predictions."""
        import joblib

        # Load existing model
        existing = joblib.load(existing_model_path)

        if 'classifier' not in existing:
            raise ValueError("No classifier in existing model")

        old_clf = existing['classifier']
        old_reg = existing.get('regressor')

        # Generate synthetic training data
        n_samples = 10000

        # Classifier training
        X_clf = np.random.randn(n_samples, old_clf.n_features_in_).astype(np.float32)
        X_clf[:, 0] = np.random.uniform(100, 2000, n_samples)  # Context length
        if old_clf.n_features_in_ > 1:
            X_clf[:, 1] = np.random.uniform(0, 8, n_samples)  # Position

        y_clf = old_clf.predict(X_clf)

        # Train lightweight classifier
        self.classifier.fit(X_clf, y_clf)

        # Regressor training (if exists)
        if old_reg:
            X_reg = np.random.randn(n_samples, old_reg.n_features_in_).astype(np.float32)
            X_reg[:, 0] = np.random.uniform(100, 2000, n_samples)
            y_reg = old_reg.predict(X_reg)
            self.regressor.fit(X_reg, y_reg)

        self._trained = True

    def save(self, path: str):
        """Save the lightweight model."""
        if not self._trained:
            raise ValueError("Model not trained yet")

        model_dict = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'type': 'lightweight',
            'metadata': {
                'n_trees': self.classifier.n_estimators,
                'max_depth': self.classifier.max_depth
            }
        }
        joblib.dump(model_dict, path)


def create_fast_model(existing_model_path: str = 'src/acceptance/llama2_7b_vs_70b.joblib'):
    """Create and test a fast acceptance model."""

    print("Creating fast acceptance model...")
    print("=" * 60)

    # Test heuristic model
    print("\n1. Testing heuristic model:")
    heuristic = FastAcceptanceModel()

    times = []
    for _ in range(100):
        start = time.perf_counter()
        probs = heuristic.position_probabilities(
            context_length=1000.0,
            depth=8
        )
        times.append((time.perf_counter() - start) * 1000)

    print(f"   Speed: {sum(times)/len(times):.3f}ms (vs 28ms for full model)")

    # Create lightweight RF model
    print("\n2. Creating lightweight RandomForest (50 trees, depth=8):")
    lightweight = LightweightRandomForestAcceptance(n_trees=50, max_depth=8)

    try:
        lightweight.train_from_existing(existing_model_path)

        # Test speed
        test_data = np.random.randn(8, 2).astype(np.float32)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            lightweight.classifier.predict_proba(test_data)
            times.append((time.perf_counter() - start) * 1000)

        print(f"   Speed: {sum(times)/len(times):.2f}ms")

        # Save the model
        lightweight.save('src/acceptance/lightweight_acceptance.joblib')
        print("   Saved to: src/acceptance/lightweight_acceptance.joblib")

    except Exception as e:
        print(f"   Error training lightweight model: {e}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("Use the heuristic model for instant responses (0.01ms)")
    print("Or the lightweight RF for better accuracy (2-3ms expected)")


if __name__ == "__main__":
    create_fast_model()