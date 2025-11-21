#!/usr/bin/env python3
"""
Run comprehensive long-context efficiency benchmarks.

This is the main entry point that runs all benchmarks and generates
a complete analysis of long-context efficiency.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarks.core.kv_cache_benchmark import KVCacheBenchmark
from benchmarks.core.squeezed_attention import SqueezedAttentionBenchmark
from benchmarks.strategies.batching import BatchingBenchmark
from benchmarks.visualization.plots import BenchmarkVisualizer
from benchmarks.utils.cost_analysis import CostAnalyzer, WasteAnalyzer


def run_comprehensive_benchmark(args):
    """Run all benchmarks."""
    print("=" * 80)
    print("LONG-CONTEXT EFFICIENCY COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    print(f"\nModel: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map=args.device if args.device == 'cuda' else None,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizer(args.output_dir)
    
    # Sample text
    sample_text = """
    The field of artificial intelligence has seen remarkable progress in recent years,
    particularly in natural language processing. Large language models with extended
    context windows have enabled new capabilities, but they also introduce significant
    computational challenges. Understanding and optimizing these costs is crucial for
    practical deployment at scale.
    """
    
    all_results = {}
    
    # 1. KV Cache Efficiency Benchmark
    if not args.skip_kv_cache:
        print("\n" + "=" * 80)
        print("1. KV CACHE EFFICIENCY BENCHMARK")
        print("=" * 80)
        
        kv_benchmark = KVCacheBenchmark(model, tokenizer, device=args.device)
        
        sequence_lengths = [128, 256, 512, 1024]
        print(f"Testing sequence lengths: {sequence_lengths}")
        
        kv_results = kv_benchmark.benchmark_scaling(
            base_text=sample_text,
            sequence_lengths=sequence_lengths,
            tokens_to_generate=50,
        )
        
        all_results['kv_cache'] = kv_results
        
        # Visualize
        visualizer.plot_kv_cache_scaling(
            kv_results,
            save_path=os.path.join(args.output_dir, 'kv_cache_scaling.png'),
        )
        
        print("\n✓ KV Cache benchmark complete")
    
    # 2. Squeezed Attention Benchmark
    if not args.skip_attention:
        print("\n" + "=" * 80)
        print("2. SQUEEZED ATTENTION BENCHMARK")
        print("=" * 80)
        
        attention_benchmark = SqueezedAttentionBenchmark(model, tokenizer, device=args.device)
        
        attention_results = attention_benchmark.compare_attention_techniques(
            input_text=sample_text * 10,
            max_new_tokens=50,
        )
        
        all_results['squeezed_attention'] = attention_results
        
        # Visualize
        if attention_results:
            visualizer.plot_attention_comparison(
                attention_results,
                save_path=os.path.join(args.output_dir, 'attention_comparison.png'),
            )
        
        print("\n✓ Squeezed Attention benchmark complete")
    
    # 3. Batching Strategies Benchmark
    if not args.skip_batching:
        print("\n" + "=" * 80)
        print("3. BATCHING STRATEGIES BENCHMARK")
        print("=" * 80)
        
        batching_benchmark = BatchingBenchmark(model, tokenizer, device=args.device)
        
        # Generate sample requests
        input_texts = [sample_text] * 16
        batch_sizes = [1, 2, 4, 8]
        
        print(f"Testing batch sizes: {batch_sizes}")
        
        batching_results = batching_benchmark.compare_strategies(
            input_texts=input_texts,
            batch_sizes=batch_sizes,
            max_new_tokens=50,
        )
        
        all_results['batching'] = batching_results
        
        # Visualize
        visualizer.plot_batching_comparison(
            batching_results,
            save_path=os.path.join(args.output_dir, 'batching_comparison.png'),
        )
        
        print("\n✓ Batching benchmark complete")
    
    # 4. Cost Analysis
    print("\n" + "=" * 80)
    print("4. COST ANALYSIS")
    print("=" * 80)
    
    cost_analyzer = CostAnalyzer(model_name='custom-model', gpu_type='a100-40gb')
    waste_analyzer = WasteAnalyzer()
    
    # Analyze costs for different sequence lengths
    sequence_lengths = [1000, 2000, 4000, 8000, 16000, 32000]
    costs = []
    
    print("\nCost per Request Analysis:")
    print(f"{'Seq Length':<12} {'Input Cost':<12} {'Output Cost':<12} {'Total Cost':<12}")
    print("-" * 48)
    
    for seq_len in sequence_lengths:
        estimate = cost_analyzer.calculate_cost(
            input_tokens=seq_len,
            output_tokens=100,
        )
        costs.append(estimate.cost_per_request)
        print(f"{seq_len:<12} ${estimate.input_cost:<11.6f} "
              f"${estimate.output_cost:<11.6f} ${estimate.total_cost:<11.6f}")
    
    # Visualize cost curve
    visualizer.plot_cost_per_request(
        sequence_lengths,
        costs,
        save_path=os.path.join(args.output_dir, 'cost_per_request.png'),
    )
    
    # Waste analysis
    print("\nCompute Waste Analysis:")
    waste_analysis = waste_analyzer.analyze_attention_waste(
        sequence_length=8000,
        actual_relevant_tokens=1000,
    )
    print(f"  Attention waste: {waste_analysis['waste_percent']:.1f}%")
    
    # Generate summary report
    visualizer.create_summary_report(
        all_results,
        save_path=os.path.join(args.output_dir, 'benchmark_report.md'),
    )
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print("Check the following files:")
    print("  - kv_cache_scaling.png")
    print("  - attention_comparison.png")
    print("  - batching_comparison.png")
    print("  - cost_per_request.png")
    print("  - benchmark_report.md")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive long-context efficiency benchmarks',
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
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results',
    )
    
    parser.add_argument(
        '--skip-kv-cache',
        action='store_true',
        help='Skip KV cache benchmark',
    )
    
    parser.add_argument(
        '--skip-attention',
        action='store_true',
        help='Skip attention benchmark',
    )
    
    parser.add_argument(
        '--skip-batching',
        action='store_true',
        help='Skip batching benchmark',
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    try:
        run_comprehensive_benchmark(args)
    except Exception as e:
        print(f"\nError running benchmarks: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
