#!/usr/bin/env python3
"""
Run batching strategy benchmarks.

Compare different batching strategies for long-context inference.
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.strategies.batching import BatchingBenchmark
from benchmarks.visualization.plots import BenchmarkVisualizer


def run_batching_benchmark(args):
    """Run batching strategy benchmark."""
    print("=" * 80)
    print("Batching Strategy Benchmark")
    print("=" * 80)
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map=args.device if args.device == 'cuda' else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize benchmark
    benchmark = BatchingBenchmark(model, tokenizer, device=args.device)
    
    # Generate sample requests
    base_text = "This is a sample text for benchmarking batching strategies. "
    input_texts = [base_text * (i + 1) for i in range(args.num_requests)]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nBenchmarking with {args.num_requests} requests")
    print(f"Batch sizes: {args.batch_sizes}")
    
    # Compare strategies
    results = benchmark.compare_strategies(
        input_texts=input_texts,
        batch_sizes=args.batch_sizes,
        max_new_tokens=50,
    )
    
    # Print results
    print("\nResults:")
    for strategy_name, metrics_list in results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"{'Batch Size':<12} {'Throughput (req/s)':<20} {'Latency (ms)':<15} {'Memory (MB)':<15}")
        print("-" * 62)
        for metrics in metrics_list:
            print(f"{metrics.batch_size:<12} "
                  f"{metrics.throughput_requests_per_sec:<20.2f} "
                  f"{metrics.avg_latency_ms:<15.2f} "
                  f"{metrics.memory_peak_mb:<15.2f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'batching_results.json')
    results_dict = {
        strategy: [m.to_dict() for m in metrics_list]
        for strategy, metrics_list in results.items()
    }
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualizations
    if args.generate_plots:
        print("\nGenerating visualizations...")
        visualizer = BenchmarkVisualizer(args.output_dir)
        
        plot_path = os.path.join(args.output_dir, 'batching_comparison.png')
        visualizer.plot_batching_comparison(
            results,
            title="Batching Strategy Comparison",
            save_path=plot_path,
        )
        print(f"Plot saved to: {plot_path}")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run batching strategy benchmarks',
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt2',
        help='Model name or path',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run on',
    )
    
    parser.add_argument(
        '--num-requests',
        type=int,
        default=16,
        help='Number of requests to benchmark',
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='Batch sizes to test',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results',
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate visualization plots',
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    try:
        run_batching_benchmark(args)
    except Exception as e:
        print(f"\nError running benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
