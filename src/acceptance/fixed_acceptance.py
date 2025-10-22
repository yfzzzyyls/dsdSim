"""Fixed acceptance model with realistic probability decay."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np


class FixedAcceptanceModel:
    """Fixed acceptance model with realistic conditional probabilities."""

    def __init__(
        self,
        initial_prob: float = 0.92,
        decay_rate: float = 0.96,
        min_prob: float = 0.25,
        max_prob: float = 0.99,
        spec_tokens: int = 8,
    ) -> None:
        """
        Args:
            initial_prob: Acceptance probability for first token.
            decay_rate: Multiplicative decay per position.
            min_prob: Lower bound applied after decay/context adjustments.
            max_prob: Upper bound applied after decay/context adjustments.
            spec_tokens: Default speculative horizon for expected-accepts queries.
        """
        self.initial_prob = float(initial_prob)
        self.decay_rate = float(decay_rate)
        self.min_prob = float(np.clip(min_prob, 0.0, 1.0))
        self.max_prob = float(np.clip(max_prob, 0.0, 1.0))

        # Speculative horizon used when no explicit depth is provided.
        self.spec_tokens = int(max(1, spec_tokens))
        self.metadata = {
            "name": "fixed_acceptance",
            "type": "fixed",
            "initial_prob": self.initial_prob,
            "decay_rate": self.decay_rate,
        }
        # Mimic sklearn classifier API so AcceptanceRegressor can introspect.
        self.classes_ = np.array([0, 1], dtype=int)

    # ------------------------------------------------------------------ helpers
    def _context_factor(self, context_length: float) -> float:
        if context_length > 2500:
            return 0.92  # 8% reduction for very long contexts
        if context_length > 1500:
            return 0.95  # 5% reduction for long contexts
        return 1.0

    def _prob_for_position(self, context_length: float, position: int) -> float:
        base = self.initial_prob * (self.decay_rate ** max(0, position))
        prob = base * self._context_factor(context_length)
        return float(np.clip(prob, self.min_prob, self.max_prob))

    def _resolve_depth(self, depth: int, feature_context: Optional[Mapping[str, Any]]) -> int:
        resolved = int(max(1, depth))
        if feature_context:
            spec_tokens = feature_context.get("spec_tokens")
            if spec_tokens:
                try:
                    resolved = int(max(1, spec_tokens))
                except (TypeError, ValueError):
                    pass
        return min(resolved, self.spec_tokens)

    # ---------------------------------------------------------------- interface
    def position_probabilities(
        self,
        *,
        context_length: float,
        depth: int,
        default: Optional[float] = None,  # unused but kept for API compatibility
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> List[float]:
        """Return fixed probabilities with exponential decay."""
        context_length = float(context_length)
        depth = self._resolve_depth(depth, feature_context)
        return [self._prob_for_position(context_length, idx) for idx in range(depth)]

    # scikit-style helpers -----------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return [[P(class=0), P(class=1)], ...] for each row."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        context = X[:, 0]
        positions = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        probs = np.clip(
            [
                self._prob_for_position(ctx, int(round(pos)))
                for ctx, pos in zip(context, positions)
            ],
            0.0,
            1.0,
        )
        probs = probs.astype(float)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (0/1) using 0.5 as the decision threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    # Expected accepts ---------------------------------------------------------
    def predict_expected_accepts(
        self,
        context_length: float,
        *,
        feature_context: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, bool]:
        """Predict expected number of accepted tokens for the requested depth."""
        depth = self._resolve_depth(self.spec_tokens, feature_context)
        if feature_context and feature_context.get("spec_tokens"):
            try:
                depth = int(max(1, feature_context["spec_tokens"]))
            except (TypeError, ValueError):
                depth = self._resolve_depth(self.spec_tokens, feature_context)
        probs = self.position_probabilities(
            context_length=context_length,
            depth=depth,
            feature_context=feature_context,
        )
        expected = float(sum(probs))
        return expected, False

    # Convenience --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FixedAcceptanceModel(initial_prob={self.initial_prob}, decay_rate={self.decay_rate})"


def test_fixed_model():
    """Test the fixed acceptance model."""
    model = FixedAcceptanceModel(initial_prob=0.85, decay_rate=0.92)

    print("Fixed Acceptance Model Test")
    print("=" * 60)

    contexts = [500, 1000, 1500, 2000]
    depths = [2, 4, 6, 8]

    for context in contexts:
        print(f"\nContext={context}:")
        for depth in depths:
            probs = model.position_probabilities(
                context_length=float(context),
                depth=depth
            )
            print(f"  Depth {depth}: {[f'{p:.2f}' for p in probs]}")

            # Calculate cumulative acceptance
            cum_accept = 1.0
            for p in probs:
                cum_accept *= p
            print(f"    Cumulative acceptance: {cum_accept:.3f}")

            # What would Spec++ select with threshold=0.9?
            cum = 1.0
            for i, p in enumerate(probs):
                cum *= p
                if 1 - cum > 0.9:
                    print(f"    Spec++ with threshold=0.9 would select gamma={i}")
                    break
            else:
                print(f"    Spec++ with threshold=0.9 would select gamma={depth}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH BROKEN MODEL:")
    print("Broken: [0.26, 0.98, 0.94, 0.95] - First token only 26%!")
    print("Fixed:  [0.85, 0.78, 0.72, 0.66] - Realistic decay")
    print("\nWith fixed model, Spec++ should select gamma=4-5")
    print("This should match or beat static gamma=4 performance")


if __name__ == "__main__":
    test_fixed_model()
