#!/usr/bin/env python3
"""
Heterogeneous Draft Latency Experiment

This script runs experiments with draft devices having heterogeneous (different) 
generation latencies to simulate real-world deployments where edge devices have
varying computational capabilities.

The draft latencies follow a normal distribution (mean=11ms, std=3ms per token),
clipped to 2-20ms range to avoid extreme values.
"""

import os
import sys
import json
import yaml
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def generate_heterogeneous_config(num_drafts: int, batch_window_ms: float = 10.0) -> dict:
    """Generate config with heterogeneous draft latencies (normal distribution)."""
    import numpy as np
    
    # Base configuration template
    config = {
        'sim_time_ms': 60000,  # 60 seconds
        'warmup_time_ms': 1000,  # 1 second warmup
        'devices': [],
        'connections': [],
        'workload': {
            'arrival': 'poisson',
            'rate_rps': num_drafts * 10  # Scale with drafts
        },
        'routing': {
            'policy': 'jsq',
            'jsq_sample_size': 2
        },
        # Conversation parameters at top level
        'answer_length': 20,  # 20 tokens per answer
        'prompt_length_min': 100,
        'prompt_length_max': 100,  # Fixed prompt length
        'prompt_scale_by_capability': False,
        'mixed_batching': True,
        'seed': 42
    }
    
    # Add heterogeneous draft devices with normal distribution latencies
    for i in range(num_drafts):
        # Normal distribution: mean=11ms, std=3ms per token
        # Clip to 2-20ms range to avoid extreme values
        latency_per_token = float(np.clip(np.random.normal(11.0, 3.0), 2.0, 20.0))
        # Total generation latency for gamma tokens
        total_latency = float(latency_per_token * 4)  # 4 tokens per draft
        
        draft_device = {
            'id': f'draft_{i}',
            'role': 'draft',
            'capability': 1.0,
            'generation_latency_ms': total_latency,  # Total time for all gamma tokens
            'burst_factor': 1.0,
            'reliability': 0.99
        }
        config['devices'].append(draft_device)
    
    # Add target servers (2 identical targets)
    for i in range(2):
        target_device = {
            'id': f'target_{i}',
            'role': 'target',
            'capability': 1.0,
            'batch_size': 8,
            'batch_window_ms': batch_window_ms,
            'decode_latency_per_token': 9.25,
            'prefill_latency_per_token': 0.5
        }
        config['devices'].append(target_device)
    
    # Add connections from all drafts to all targets
    config['connections'] = []
    for i in range(num_drafts):
        for j in range(2):  # 2 targets
            conn = {
                'draft': f'draft_{i}',
                'target': f'target_{j}',
                'forward_latency_ms': 20.0,
                'response_latency_ms': 20.0,
                'acceptance_rate': 0.85
            }
            config['connections'].append(conn)
    
    return config

def run_experiment(num_drafts: int, batch_window_ms: float = 10.0) -> Tuple[Dict, Dict]:
    """Run simulation and return latency metrics and draft latency distribution."""
    
    print(f"Testing with {num_drafts} heterogeneous draft(s)...")
    
    # Generate config with heterogeneous drafts
    config = generate_heterogeneous_config(num_drafts, batch_window_ms=batch_window_ms)
    
    # Extract draft latencies for reporting (convert back to per-token for display)
    draft_latencies = {}
    for device in config['devices']:
        if device['role'] == 'draft':
            # Convert total latency back to per-token latency
            draft_latencies[device['id']] = device['generation_latency_ms'] / 4.0
    
    # Save temp config
    temp_config = f'temp_heterogeneous_{num_drafts:02d}_drafts.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    try:
        # Run simulation
        result = subprocess.run(
            ['python', '../sim.py', '--config', temp_config],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Initialize metrics
        metrics = {
            'num_drafts': num_drafts,
            'effective_tps': 0.0,
            'acceptance_rate': 0.0,
            'avg_latency_ms': 0.0,
            'p50_latency_ms': 0.0,
            'p90_latency_ms': 0.0,
            'p99_latency_ms': 0.0,
            'rate_rps': config['workload']['rate_rps']
        }
        
        # Parse output
        lines = result.stdout.split('\n')
        total_accepted = 0
        draft_rtts = []  # Collect RTT for latency metrics
        
        for i, line in enumerate(lines):
            if "Total Accepted:" in line:
                total_accepted = float(line.split(":")[1].strip())
            elif "Effective Tokens/Second:" in line:
                metrics['effective_tps'] = float(line.split(":")[1].strip().replace(" tok/s", ""))
            elif "Acceptance Rate:" in line and "TOKEN PERFORMANCE" in result.stdout[max(0, result.stdout.index(line)-100):result.stdout.index(line)]:
                metrics['acceptance_rate'] = float(line.split(":")[1].strip().replace("%", "")) / 100
            elif "Avg RTT:" in line:
                # Extract RTT values for latency metrics
                try:
                    rtt = float(line.split(":")[1].strip().replace("ms", ""))
                    draft_rtts.append(rtt)
                except:
                    pass
        
        # Calculate effective TPS manually if it's 0
        if metrics['effective_tps'] == 0.0 and total_accepted > 0:
            # Use actual simulation time (60s - 1s warmup = 59s)
            metrics['effective_tps'] = total_accepted / 59.0
        
        # Calculate latency metrics from RTTs
        if draft_rtts:
            metrics['avg_latency_ms'] = np.mean(draft_rtts)
            metrics['p50_latency_ms'] = np.percentile(draft_rtts, 50)
            metrics['p90_latency_ms'] = np.percentile(draft_rtts, 90)
            metrics['p99_latency_ms'] = np.percentile(draft_rtts, 99)
        
        # Print summary with draft latency distribution
        print(f"  Throughput: {metrics['effective_tps']:.1f} tok/s")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f} ms")
        print(f"  P50/P90/P99: {metrics['p50_latency_ms']:.1f}/{metrics['p90_latency_ms']:.1f}/{metrics['p99_latency_ms']:.1f} ms")
        
        # Show draft latency range
        if draft_latencies:
            latencies_list = list(draft_latencies.values())
            print(f"  Draft latencies: min={min(latencies_list):.1f}, max={max(latencies_list):.1f}, avg={np.mean(latencies_list):.1f} ms/token")
        
    finally:
        # Clean up temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    return metrics, draft_latencies

def plot_results(results: List[Dict], output_dir: str):
    """Create comprehensive plots of the experimental results."""
    
    # Extract data
    drafts = [r['num_drafts'] for r in results]
    throughputs = [r['effective_tps'] for r in results]
    avg_latencies = [r['avg_latency_ms'] for r in results]
    p50_latencies = [r['p50_latency_ms'] for r in results]
    p90_latencies = [r['p90_latency_ms'] for r in results]
    p99_latencies = [r['p99_latency_ms'] for r in results]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Throughput scaling
    ax1.plot(drafts, throughputs, 'b-', marker='o', linewidth=2, markersize=6, label='Actual throughput')
    ax1.set_xlabel('Number of Drafts', fontsize=11)
    ax1.set_ylabel('Effective Tokens/Second', fontsize=11)
    ax1.set_title('Throughput Scaling', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Latency percentiles
    ax2.plot(drafts, avg_latencies, 'g-', marker='s', linewidth=2, markersize=6, label='Average')
    ax2.plot(drafts, p50_latencies, 'b-', marker='^', linewidth=1.5, markersize=5, label='P50')
    ax2.plot(drafts, p90_latencies, 'r-', marker='v', linewidth=1.5, markersize=5, label='P90')
    ax2.plot(drafts, p99_latencies, 'm-', marker='d', linewidth=1.5, markersize=5, label='P99')
    ax2.set_xlabel('Number of Drafts', fontsize=11)
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_title('Request Latency Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Latency vs Throughput tradeoff
    scatter = ax3.scatter(throughputs, avg_latencies, c=drafts, cmap='viridis', s=50, alpha=0.7)
    ax3.set_xlabel('Throughput (tokens/second)', fontsize=11)
    ax3.set_ylabel('Average Latency (ms)', fontsize=11)
    ax3.set_title('Latency vs Throughput Tradeoff', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar to show number of drafts
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Number of Drafts', fontsize=10)
    
    plt.suptitle('Heterogeneous Draft Latency Experiment\\n(Draft latencies: Normal(μ=11ms, σ=3ms), Batch: 2ms window)', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heterogeneous_latency_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\\nPlot saved to {plot_path}")
    
    return fig

def plot_latency_distribution(latency_distributions: List[Dict], output_dir: str):
    """Plot the distribution of draft latencies across experiments."""
    
    # Flatten all latencies
    all_latencies = []
    for dist in latency_distributions:
        all_latencies.extend(list(dist.values()))
    
    if not all_latencies:
        return
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Histogram of draft latencies
    ax1.hist(all_latencies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=np.mean(all_latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_latencies):.1f} ms')
    ax1.axvline(x=np.median(all_latencies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_latencies):.1f} ms')
    ax1.set_xlabel('Draft Generation Latency (ms/token)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Draft Latencies', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'draft_latency_distribution.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Distribution plot saved to {plot_path}")
    
    return fig

def print_summary_table(results: List[Dict]):
    """Print a formatted summary table of results."""
    
    print("\\n" + "=" * 90)
    print("SUMMARY - HETEROGENEOUS LATENCY EXPERIMENT")
    print("=" * 90)
    print(f"{'Drafts':>7} | {'Throughput':>12} | {'Avg Latency':>12} | {'P50':>8} | {'P90':>8} | {'P99':>8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['num_drafts']:>7} | {r['effective_tps']:>8.1f} tok/s | "
              f"{r['avg_latency_ms']:>8.1f} ms | {r['p50_latency_ms']:>6.1f} ms | "
              f"{r['p90_latency_ms']:>6.1f} ms | {r['p99_latency_ms']:>6.1f} ms")

def main():
    """Main experiment runner."""
    
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'heterogeneous_latency'
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    print("=" * 90)
    print("HETEROGENEOUS DRAFT LATENCY EXPERIMENT")
    print("=" * 90)
    print("Configuration:")
    print("  - Draft generation: Normal distribution (mean=11ms, std=3ms, clipped to 2-20ms)")
    print("  - Target batch processing:")
    print("    * Decode-only: 37ms (4 tokens x 9.25ms/token)")
    print("    * Prefill-containing: 50ms (100 tokens x 0.5ms/token)")
    print("    * Batch window: 2ms")
    print("    * Max batch size: 8")
    print("  - Network: 20ms forward, 20ms response")
    print("  - Acceptance: p=0.85 per token")
    print("\\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test points - focus on range where latency changes most
    draft_counts = list(range(5, 101, 5))  # [5, 10, 15, ..., 100]
    
    results = []
    latency_distributions = []
    
    for n in draft_counts:
        metrics, draft_latencies = run_experiment(n, batch_window_ms=2.0)  # Use 2ms batch window
        results.append(metrics)
        latency_distributions.append(draft_latencies)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\nResults saved to {results_file}")
    
    # Save latency distributions
    dist_file = os.path.join(output_dir, 'latency_distributions.json')
    with open(dist_file, 'w') as f:
        json.dump(latency_distributions, f, indent=2)
    
    # Generate plots
    plot_results(results, output_dir)
    plot_latency_distribution(latency_distributions, output_dir)
    
    # Print summary table
    print_summary_table(results)
    
    # Find optimal configuration (best throughput/latency ratio)
    best_ratio = 0
    best_config = None
    for r in results:
        if r['avg_latency_ms'] > 0:
            ratio = r['effective_tps'] / r['avg_latency_ms']
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = r
    
    if best_config:
        print("\\n" + "=" * 90)
        print("OPTIMAL CONFIGURATION")
        print("=" * 90)
        print(f"Best throughput/latency ratio at {best_config['num_drafts']} drafts:")
        print(f"  - Throughput: {best_config['effective_tps']:.1f} tok/s")
        print(f"  - Avg Latency: {best_config['avg_latency_ms']:.1f} ms")
        print(f"  - Efficiency: {best_ratio:.3f} tok/s/ms")

if __name__ == "__main__":
    main()