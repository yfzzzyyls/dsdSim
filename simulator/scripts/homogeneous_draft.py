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

def generate_config(num_drafts: int, output_path: str = None, batch_window_ms: float = 10.0) -> dict:
    """Generate config with N identical drafts and specified batch window."""
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), '../configs/homogeneous_draft.yaml')
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch window for all targets
    for device in config['devices']:
        if device.get('role') == 'target':
            device['batch_window_ms'] = batch_window_ms
    
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

def run_experiment(num_drafts: int, batch_window_ms: float = 10.0) -> Tuple[float, dict]:
    """Run simulation and return effective tokens/second and full metrics."""
    
    print(f"Testing with {num_drafts} draft(s)...")
    
    # Generate config with specified batch window
    config = generate_config(num_drafts, batch_window_ms=batch_window_ms)
    
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

def calculate_theoretical_per_draft(gamma: int = 4) -> float:
    """Calculate theoretical max tokens/s per draft based on config."""
    # Load template to get actual parameters
    template_path = os.path.join(os.path.dirname(__file__), '../configs/homogeneous_draft.yaml')
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get target parameters
    target_config = next((d for d in config['devices'] if d['role'] == 'target'), None)
    if not target_config:
        return 24.5  # fallback
    
    # Calculate typical decode batch latency
    decode_latency_per_token = target_config.get('decode_latency_per_token', 9.25)
    decode_batch_latency = gamma * decode_latency_per_token  # 37ms for gamma=4
    batch_window = target_config.get('batch_window_ms', 10.0)
    
    # Total batch cycle time
    batch_cycle = decode_batch_latency + batch_window  # 47ms
    
    # Requests per second per draft (no queueing)
    # Draft takes: 24ms gen + 20ms forward + batch_cycle + 20ms response = 64ms + batch_cycle
    draft_cycle = 24.0 + 20.0 + batch_cycle + 20.0  # 111ms for our params
    req_per_s = 1000.0 / draft_cycle  # ~9.01 req/s
    
    # Tokens per request (85% per-token acceptance, sequential)
    acceptance_rate = 0.85
    tokens_per_req = sum(acceptance_rate**k for k in range(1, gamma+1))  # ~2.71
    
    return req_per_s * tokens_per_req

def plot_results(results: List[dict], output_dir: str):
    """Create visualization of scaling results."""
    
    drafts = [r['num_drafts'] for r in results]
    tps = [r['effective_tps'] for r in results]
    
    # Calculate speedup and efficiency
    theoretical_per_draft = calculate_theoretical_per_draft(gamma=4)
    
    # Use 1-draft result as baseline for speedup comparison
    baseline = tps[0] if tps[0] > 0 else 13.0
    speedups = [t / baseline for t in tps]
    
    # Efficiency should be calculated against theoretical maximum
    # Efficiency = actual_tps / (num_drafts * theoretical_per_draft)
    efficiencies = [t / (d * theoretical_per_draft) if d > 0 else 0 for t, d in zip(tps, drafts)]
    
    # Create figure with single plot
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot Tokens per second
    ax1.plot(drafts, tps, 'b-o', linewidth=0.6, markersize=4, label='Actual throughput')
    # Removed baseline line since it's always near zero
    ax1.set_xlabel('Number of Drafts')
    ax1.set_ylabel('Effective Tokens/Second')
    ax1.set_title('Scaling: Effective Tokens/Second')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([d for d in drafts if d % 10 == 0])  # Show multiples of 10 for clarity
    
    # Add ideal linear scaling line (theoretical maximum) with proper label
    ideal_tps = [theoretical_per_draft * d for d in drafts]
    ax1.plot(drafts, ideal_tps, 'g--', alpha=0.4, linewidth=0.5, label='Theoretical max (no bottleneck)')
    ax1.legend(loc='upper left')
    
    plt.suptitle('Homogeneous Draft Scaling Experiment\n(24ms gen, mixed batch: 37ms decode/50ms prefill, Δ=2ms window, p=0.85/token→E[accept]=2.71/4)', fontsize=13)
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
    
    # Theoretical maximum per draft (calculated from actual config)
    theoretical_per_draft = calculate_theoretical_per_draft(gamma=4)
    
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
    for r in results:
        efficiency = (r['effective_tps'] / (r['num_drafts'] * theoretical_per_draft) * 100) if r['num_drafts'] > 0 else 0
        if efficiency < 50:
            print(f"Efficiency drops below 50% at: {r['num_drafts']} drafts")
            break

def run_batch_window_comparison():
    """Run experiments for both 10ms and 100ms batch windows and create comparison plot."""
    
    print("=" * 70)
    print("BATCH WINDOW COMPARISON EXPERIMENT")
    print("=" * 70)
    
    # Test points for scaling curve (1 to 100 drafts)
    draft_counts = list(range(1, 101))  # [1, 2, 3, ..., 100]
    
    # Run 10ms experiment
    print("\n" + "=" * 70)
    print("RUNNING 10ms BATCH WINDOW")
    print("=" * 70)
    results_10ms = []
    for n in draft_counts:
        tps, metrics = run_experiment(n, batch_window_ms=10.0)
        results_10ms.append(metrics)
    
    # Run 100ms experiment
    print("\n" + "=" * 70)
    print("RUNNING 100ms BATCH WINDOW")
    print("=" * 70)
    results_100ms = []
    for n in draft_counts:
        tps, metrics = run_experiment(n, batch_window_ms=100.0)
        results_100ms.append(metrics)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data
    drafts_10ms = [r['num_drafts'] for r in results_10ms]
    tps_10ms = [r['effective_tps'] for r in results_10ms]
    
    drafts_100ms = [r['num_drafts'] for r in results_100ms]
    tps_100ms = [r['effective_tps'] for r in results_100ms]
    
    # Plot 1: Both curves
    ax1.plot(drafts_10ms, tps_10ms, 'b-', linewidth=2, label='10ms batch window', alpha=0.8)
    ax1.plot(drafts_100ms, tps_100ms, 'r-', linewidth=2, label='100ms batch window', alpha=0.8)
    
    # Mark the step region
    ax1.axvspan(52, 55, alpha=0.1, color='yellow', label='Step region (10ms)')
    
    ax1.set_xlabel('Number of Draft Models', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax1.set_title('Scaling Curves: 10ms vs 100ms Batch Window', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, max(max(tps_10ms), max(tps_100ms)) * 1.1)
    
    # Plot 2: Difference
    min_len = min(len(tps_10ms), len(tps_100ms))
    difference = [tps_100ms[i] - tps_10ms[i] for i in range(min_len)]
    drafts_common = drafts_10ms[:min_len]
    
    ax2.plot(drafts_common, difference, 'g-', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(drafts_common, 0, difference, where=[d > 0 for d in difference], 
                      color='green', alpha=0.3, label='100ms better')
    ax2.fill_between(drafts_common, 0, difference, where=[d <= 0 for d in difference], 
                      color='red', alpha=0.3, label='10ms better')
    
    # Mark regions
    ax2.axvspan(20, 52, alpha=0.1, color='green')
    if max(difference) > 0:
        ax2.text(35, max(difference)*0.8, '100ms advantage\n(better batching)', 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Number of Draft Models', fontsize=12)
    ax2.set_ylabel('Throughput Difference (100ms - 10ms)', fontsize=12)
    ax2.set_title('Performance Advantage: 100ms vs 10ms Batch Window', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = '/home/external/choral-spec-internal/simulator/results'
    plot_path = os.path.join(output_dir, 'scaling_curve_100ms_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    # Print key metrics
    print("\n" + "="*70)
    print("KEY COMPARISON METRICS")
    print("="*70)
    
    # Find the step region metrics
    for draft_count in [50, 52, 53, 55]:
        idx_10ms = next((i for i, r in enumerate(results_10ms) if r['num_drafts'] == draft_count), None)
        idx_100ms = next((i for i, r in enumerate(results_100ms) if r['num_drafts'] == draft_count), None)
        
        if idx_10ms is not None and idx_100ms is not None:
            tp_10ms = results_10ms[idx_10ms]['effective_tps']
            tp_100ms = results_100ms[idx_100ms]['effective_tps']
            diff = tp_100ms - tp_10ms
            print(f"{draft_count:3d} drafts: 10ms={tp_10ms:6.1f} tok/s, 100ms={tp_100ms:6.1f} tok/s, diff={diff:+6.1f}")
    
    # Calculate jump for both
    tp_50_10ms = next((r['effective_tps'] for r in results_10ms if r['num_drafts'] == 50), 0)
    tp_55_10ms = next((r['effective_tps'] for r in results_10ms if r['num_drafts'] == 55), 0)
    tp_50_100ms = next((r['effective_tps'] for r in results_100ms if r['num_drafts'] == 50), 0)
    tp_55_100ms = next((r['effective_tps'] for r in results_100ms if r['num_drafts'] == 55), 0)
    
    print(f"\nStep pattern (50→55 drafts):")
    print(f"  10ms window: {tp_50_10ms:.1f} → {tp_55_10ms:.1f} tok/s (jump of {tp_55_10ms-tp_50_10ms:.1f})")
    print(f"  100ms window: {tp_50_100ms:.1f} → {tp_55_100ms:.1f} tok/s (change of {tp_55_100ms-tp_50_100ms:.1f})")
    
    # Save results for both
    for results, window in [(results_10ms, '10ms'), (results_100ms, '100ms')]:
        json_path = os.path.join(output_dir, f'homogeneous_draft_{window}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{window} results saved to {json_path}")
    
    return plot_path

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
    print("  - Target batch processing (mixed batching):")
    print("    * Decode-only batches: 37ms (4 tokens × 9.25ms/token)")
    print("    * Prefill-containing batches: 50ms (100 tokens × 0.5ms/token)")
    print("    * Batch window (Δ): 2ms")
    print("    * Max batch size: 8")
    print("  - Network latency: 20ms forward, 20ms response")
    print("  - Acceptance: p=0.85 per token → E[accepted]=2.71/4 tokens (67.7%)")
    print("  - Prompt: 100 tokens fixed (prompt_scale_by_capability=false)")
    print("  - Answer: 20 tokens per conversation (5 decode rounds)")
    print("  - Simulation duration: 60 seconds")
    print("=" * 70)
    
    # Test points for scaling curve (1 to 100 drafts)
    draft_counts = list(range(1, 101))  # [1, 2, 3, ..., 100]
    
    results = []
    for n in draft_counts:
        tps, metrics = run_experiment(n, batch_window_ms=2.0)  # Use 2ms batch window
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
    
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Run comparison experiment
        run_batch_window_comparison()
    else:
        # Run standard 10ms experiment
        main()