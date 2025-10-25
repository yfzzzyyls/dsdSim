#!/usr/bin/env python3
"""Export the fixed acceptance model to joblib format."""

from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acceptance.fixed_acceptance import FixedAcceptanceModel

# Create model with realistic acceptance rates
# Position 0: 85% (most likely to accept first token)
# Then exponential decay: 78%, 72%, 66%, 61%, 56%, 52%, 48%
model = FixedAcceptanceModel(
    initial_prob=0.93,
    decay_rate=0.97,
    min_prob=0.30,
    max_prob=0.995,
    spec_tokens=8,
)

# Test the model
print("Fixed Acceptance Model - Position Probabilities:")
print("=" * 60)

for context in [500, 1000, 1500]:
    for depth in [2, 4, 6, 8]:
        probs = model.position_probabilities(
            context_length=float(context),
            depth=depth
        )
        print(f"Context={context}, Depth={depth}: {[f'{p:.2f}' for p in probs]}")

output_path = Path(__file__).resolve().parent / "fixed_acceptance.joblib"
print(f"\nExporting to {output_path}...")
# Export as dict to match expected format
model_dict = {
    'metadata': {'name': 'fixed_acceptance', 'type': 'fixed'},
    'spec_tokens': 8,
    'feature_columns': [],
    'regressor': None,  # Not used
    'classifier': model,  # The actual model
    'metrics': {},
    'details_source': 'Fixed acceptance model with realistic rates'
}
joblib.dump(model_dict, output_path)
print("Done! Model exported successfully.")

print("\n" + "=" * 60)
print("Expected behavior with this model:")
print("- Spec++ should select gamma=4-5 (good balance)")
print("- Static gamma=4 should work well (high acceptance)")
print("- Backoff should stabilize around gamma=4-5")
