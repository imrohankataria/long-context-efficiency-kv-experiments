#!/usr/bin/env python3
"""
Run comprehensive KV cache efficiency benchmarks.

This script benchmarks KV cache efficiency across different sequence lengths
and configurations, generating detailed metrics and visualizations.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.core.kv_cache_benchmark import KVCacheBenchmark
from benchmarks.visualization.plots import BenchmarkVisualizer
from benchmarks.utils.cost_analysis import CostAnalyzer


def run_kv_cache_benchmark(args):
    """Run KV cache benchmark."""
    print("=" * 80)
    print("KV Cache Efficiency Benchmark")
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
    benchmark = KVCacheBenchmark(model, tokenizer, device=args.device)
    
    # Sample text
    sample_text = args.input_text or """
    Artificial intelligence has transformed how we process and understand language.
    Long-context models enable processing of extensive documents, but they come with
    significant computational costs that need to be carefully measured and optimized.
    """
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "-" * 80)
    print("Running Scaling Benchmark")
    print("-" * 80)
    
    # Benchmark across sequence lengths
    sequence_lengths = [128, 256, 512, 1024]  # Start with smaller lengths
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    results = benchmark.benchmark_scaling(
        base_text=sample_text,
        sequence_lengths=sequence_lengths,
        tokens_to_generate=50,
    )
    
    # Print results
    print("\nResults:")
    print(f"{'Seq Len':<10} {'KV Cache (MB)':<15} {'Memory (MB)':<15} {'Throughput (tok/s)':<20} {'Latency (ms)':<15}")
    print("-" * 80)
    
    for metrics in results:
        print(f"{metrics.sequence_length:<10} "
              f"{metrics.kv_cache_size_bytes/(1024**2):<15.2f} "
              f"{metrics.memory_allocated_mb:<15.2f} "
              f"{metrics.throughput_tokens_per_sec:<20.2f} "
              f"{metrics.time_per_token_ms:<15.2f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'kv_cache_results.json')
    with open(results_file, 'w') as f:
        json.dump([m.to_dict() for m in results], f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualizations
    if args.generate_plots:
        print("\nGenerating visualizations...")
        visualizer = BenchmarkVisualizer(args.output_dir)
        
        plot_path = os.path.join(args.output_dir, 'kv_cache_scaling.png')
        visualizer.plot_kv_cache_scaling(
            results,
            title=f"KV Cache Efficiency - {args.model_name}",
            save_path=plot_path,
        )
        
        print(f"Plot saved to: {plot_path}")
    
    # Cost analysis
    if args.analyze_costs:
        print("\n" + "-" * 80)
        print("Cost Analysis")
        print("-" * 80)
        
        cost_analyzer = CostAnalyzer(model_name='custom-model', gpu_type='a100-40gb')
        
        for metrics in results:
            estimate = cost_analyzer.calculate_cost(
                input_tokens=metrics.sequence_length,
                output_tokens=50,
            )
            print(f"\nSequence Length: {metrics.sequence_length}")
            print(f"  Cost per request: ${estimate.cost_per_request:.6f}")
            print(f"  Input cost: ${estimate.input_cost:.6f}")
            print(f"  Output cost: ${estimate.output_cost:.6f}")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run KV cache efficiency benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='gpt2',
        help='Model name or path (default: gpt2)',
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to run on',
    )
    
    parser.add_argument(
        '--input-text',
        type=str,
        default=None,
        help='Custom input text for benchmarking',
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
    
    parser.add_argument(
        '--analyze-costs',
        action='store_true',
        help='Perform cost analysis',
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    try:
        run_kv_cache_benchmark(args)
    except Exception as e:
        print(f"\nError running benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
