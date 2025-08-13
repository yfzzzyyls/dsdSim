#!/usr/bin/env python3
"""
Homogeneous draft scaling experiment.
Tests system performance with 1 to N identical draft devices.
Each draft has 24ms generation latency (6ms per token × 4 tokens).
"""

import yaml
import subprocess
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Default draft configuration (all drafts are identical)
DRAFT_TEMPLATE = {
    'role': 'draft',
    'capability': 1.0,
    'generation_latency_ms': 24.0,  # 6ms per token × 4 tokens
    'burst_factor': 1.0,
    'reliability': 0.99
}

# Default connection configuration (all connections are identical)
# Note: With sequential acceptance (stop on first reject), we need ~85% per-token
# probability to achieve ~65% overall acceptance for 4 tokens
# Math: 0.85 + 0.85^2 + 0.85^3 + 0.85^4 = 2.6 accepted / 4 total = 65%
CONNECTION_TEMPLATE = {
    'forward_latency_ms': 20.0,
    'response_latency_ms': 20.0,
    'acceptance_rate': 0.85  # This gives ~65% overall with sequential acceptance
}

def generate_config(num_drafts: int, output_path: str = None) -> dict:
    """Generate config with N identical drafts."""
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), '../configs/homogeneous_draft.yaml')
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add N draft devices (all identical)
    for i in range(num_drafts):
        draft = DRAFT_TEMPLATE.copy()
        draft['id'] = f'd{i}'
        config['devices'].append(draft)
    
    # Create connections (all drafts to all targets, all identical)
    config['connections'] = []
    targets = [d for d in config['devices'] if d['role'] == 'target']
    
    for i in range(num_drafts):
        for target in targets:
            conn = CONNECTION_TEMPLATE.copy()
            conn['draft'] = f'd{i}'
            conn['target'] = target['id']
            config['connections'].append(conn)
    
    # Scale workload with number of drafts
    # Each draft can handle ~10 req/s in blocking mode
    # No cap - let the system find its natural limit
    config['workload']['rate_rps'] = num_drafts * 10  # No cap, linear scaling
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    return config

def run_experiment(num_drafts: int) -> Tuple[float, dict]:
    """Run simulation and return effective tokens/second and full metrics."""
    
    print(f"Testing with {num_drafts} draft(s)...")
    
    # Generate config
    config = generate_config(num_drafts)
    
    # Save temp config
    temp_config = f'temp_config_{num_drafts:02d}_drafts.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run simulation
        cmd = ['python', '../sim.py', '--config', temp_config]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        # Parse output for effective tokens/second
        effective_tps = 0.0
        acceptance_rate = 0.0
        
        for line in result.stdout.split('\n'):
            if "Effective Tokens/Second:" in line:
                effective_tps = float(line.split(":")[1].strip().replace(" tok/s", ""))
            elif "Acceptance Rate:" in line and "TOKEN PERFORMANCE" in result.stdout[max(0, result.stdout.index(line)-100):result.stdout.index(line)]:
                acceptance_rate = float(line.split(":")[1].strip().replace("%", "")) / 100
        
        print(f"  Effective: {effective_tps:.1f} tok/s (acceptance: {acceptance_rate:.1%})")
        
        metrics = {
            'num_drafts': num_drafts,
            'effective_tps': effective_tps,
            'acceptance_rate': acceptance_rate,
            'rate_rps': config['workload']['rate_rps']
        }
        
        return effective_tps, metrics
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_config):
            os.remove(temp_config)

def plot_results(results: List[dict], output_dir: str):
    """Create visualization of scaling results."""
    
    drafts = [r['num_drafts'] for r in results]
    tps = [r['effective_tps'] for r in results]
    
    # Calculate speedup and efficiency
    # The theoretical maximum per draft is ~24.5 tok/s
    # (9.01 req/s * 2.72 accepted tokens per request)
    theoretical_per_draft = 24.5  # Based on our calculations
    
    # Use 1-draft result as baseline for speedup comparison
    baseline = tps[0] if tps[0] > 0 else 13.0
    speedups = [t / baseline for t in tps]
    
    # Efficiency should be calculated against theoretical maximum
    # Efficiency = actual_tps / (num_drafts * theoretical_per_draft)
    efficiencies = [t / (d * theoretical_per_draft) if d > 0 else 0 for t, d in zip(tps, drafts)]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Tokens per second
    ax1.plot(drafts, tps, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=baseline, color='r', linestyle='--', alpha=0.5, label='Baseline (no speculation)')
    ax1.set_xlabel('Number of Drafts')
    ax1.set_ylabel('Effective Tokens/Second')
    ax1.set_title('Scaling: Effective Tokens/Second')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks([d for d in drafts if d % 5 == 0])  # Show only multiples of 5 for clarity
    
    # Add ideal linear scaling line (theoretical maximum)
    ideal_tps = [theoretical_per_draft * d for d in drafts]
    ax1.plot(drafts, ideal_tps, 'g--', alpha=0.3, label='Theoretical max (no bottleneck)')
    
    # Plot 2: Speedup
    ax2.plot(drafts, speedups, 'g-s', linewidth=2, markersize=8)
    ax2.plot(drafts, drafts, 'r--', alpha=0.3, label='Ideal speedup')
    ax2.set_xlabel('Number of Drafts')
    ax2.set_ylabel('Speedup (vs 25 drafts)')  # Updated baseline reference
    ax2.set_title('Speedup Factor')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks([d for d in drafts if d % 5 == 0])  # Show only multiples of 5
    
    # Plot 3: Efficiency
    ax3.plot(drafts, [e * 100 for e in efficiencies], 'r-^', linewidth=2, markersize=8)
    ax3.axhline(y=100, color='g', linestyle='--', alpha=0.3, label='Perfect efficiency')
    ax3.axhline(y=80, color='orange', linestyle='--', alpha=0.3, label='80% efficiency')
    ax3.axhline(y=50, color='blue', linestyle='--', alpha=0.3, label='50% efficiency')
    ax3.set_xlabel('Number of Drafts')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_title('Scaling Efficiency (vs Theoretical Max)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 100])
    ax3.set_xticks([d for d in drafts if d % 5 == 0])  # Show only multiples of 5
    
    plt.suptitle('Homogeneous Draft Scaling Experiment\n(24ms generation, 37ms verification, 70% acceptance)', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scaling_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    return fig

def print_summary(results: List[dict]):
    """Print summary table of results."""
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Theoretical maximum per draft
    theoretical_per_draft = 24.5  # tok/s
    
    # Use 1-draft result as baseline for speedup
    baseline = results[0]['effective_tps'] if results[0]['effective_tps'] > 0 else 13.0
    
    print(f"{'Drafts':>7} | {'Tokens/s':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'Load':>8}")
    print("-" * 70)
    
    for r in results:
        speedup = r['effective_tps'] / baseline
        # Efficiency vs theoretical maximum
        efficiency = (r['effective_tps'] / (r['num_drafts'] * theoretical_per_draft) * 100) if r['num_drafts'] > 0 else 0
        print(f"{r['num_drafts']:7d} | {r['effective_tps']:10.1f} | {speedup:7.2f}x | {efficiency:9.1f}% | {r['rate_rps']:8d}")
    
    # Find saturation point
    max_tps = max(r['effective_tps'] for r in results)
    saturation_threshold = max_tps * 0.95  # 95% of max
    
    saturation_point = None
    for r in results:
        if r['effective_tps'] >= saturation_threshold:
            saturation_point = r['num_drafts']
            break
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"Maximum throughput: {max_tps:.1f} tok/s")
    print(f"Saturation point: ~{saturation_point} drafts (95% of max throughput)")
    print(f"Baseline (1 draft): {baseline:.1f} tok/s")
    print(f"Max speedup: {max_tps/baseline:.2f}x")
    
    # Find efficiency drop point (< 50%)
    theoretical_per_draft = 24.5
    for r in results:
        efficiency = (r['effective_tps'] / (r['num_drafts'] * theoretical_per_draft) * 100) if r['num_drafts'] > 0 else 0
        if efficiency < 50:
            print(f"Efficiency drops below 50% at: {r['num_drafts']} drafts")
            break

def main():
    """Run the homogeneous draft scaling experiment."""
    
    # Create output directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join('/home/external/choral-spec-internal/simulator/results', script_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    print("=" * 70)
    print("HOMOGENEOUS DRAFT SCALING EXPERIMENT")
    print("=" * 70)
    print("Configuration:")
    print("  - Draft generation: 24ms (6ms/token × 4 tokens)")
    print("  - Target verification: 37ms")
    print("  - Network latency: 20ms each way")
    print("  - Acceptance rate: 70%")
    print("  - Simulation duration: 60 seconds")
    print("=" * 70)
    
    # Test points for scaling curve (1 to 120 drafts)
    draft_counts = list(range(1, 121))  # [1, 2, 3, ..., 120]
    
    results = []
    for n in draft_counts:
        tps, metrics = run_experiment(n)
        results.append(metrics)
    
    # Print summary
    print_summary(results)
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        # Redirect print output to file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_summary(results)
        sys.stdout = original_stdout
    print(f"Summary saved to {summary_path}")
    
    # Create plots
    plot_results(results, output_dir)
    
    # Save results to JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()