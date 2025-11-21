#!/usr/bin/env python3
"""
Quick test script to verify the benchmark setup.

This script runs minimal tests to ensure all components work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from benchmarks.core.kv_cache_benchmark import KVCacheBenchmark, KVCacheMetrics
        from benchmarks.core.squeezed_attention import SqueezedAttentionBenchmark
        from benchmarks.strategies.batching import BatchingBenchmark
        from benchmarks.visualization.plots import BenchmarkVisualizer
        from benchmarks.utils.cost_analysis import CostAnalyzer, WasteAnalyzer
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_cost_analysis():
    """Test cost analysis utilities."""
    print("\nTesting cost analysis...")
    
    try:
        from benchmarks.utils.cost_analysis import CostAnalyzer, WasteAnalyzer
        
        # Test cost analyzer
        analyzer = CostAnalyzer(model_name='custom-model', gpu_type='a100-40gb')
        estimate = analyzer.calculate_cost(input_tokens=1000, output_tokens=100)
        
        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 100
        assert estimate.total_cost > 0
        
        # Test waste analyzer
        waste_analyzer = WasteAnalyzer()
        waste = waste_analyzer.analyze_attention_waste(
            sequence_length=1000,
            actual_relevant_tokens=100,
        )
        
        assert 'waste_percent' in waste
        assert waste['waste_percent'] > 0
        
        print("✓ Cost analysis tests passed")
        return True
    except Exception as e:
        print(f"✗ Cost analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_dataclasses():
    """Test that metrics dataclasses work correctly."""
    print("\nTesting metrics dataclasses...")
    
    try:
        from benchmarks.core.kv_cache_benchmark import KVCacheMetrics
        from benchmarks.core.squeezed_attention import SqueezedAttentionMetrics
        from benchmarks.strategies.batching import BatchingMetrics
        
        # Test KVCacheMetrics
        kv_metrics = KVCacheMetrics(
            sequence_length=1024,
            batch_size=1,
            num_layers=12,
            kv_cache_size_bytes=1024*1024,
            memory_allocated_mb=100.0,
            memory_reserved_mb=120.0,
            time_per_token_ms=10.0,
            throughput_tokens_per_sec=100.0,
            gpu_memory_percent=50.0,
        )
        
        data = kv_metrics.to_dict()
        assert data['sequence_length'] == 1024
        assert data['throughput_tokens_per_sec'] == 100.0
        
        # Test SqueezedAttentionMetrics
        attn_metrics = SqueezedAttentionMetrics(
            technique="sliding_window",
            sequence_length=2048,
            compression_ratio=0.5,
            attention_memory_mb=50.0,
            inference_time_ms=100.0,
            throughput_tokens_per_sec=200.0,
            memory_reduction_percent=50.0,
            speedup_factor=2.0,
        )
        
        data = attn_metrics.to_dict()
        assert data['technique'] == "sliding_window"
        assert data['compression_ratio'] == 0.5
        
        # Test BatchingMetrics
        batch_metrics = BatchingMetrics(
            strategy="static_batching",
            batch_size=8,
            num_requests=32,
            total_time_sec=10.0,
            avg_latency_ms=250.0,
            throughput_requests_per_sec=3.2,
            throughput_tokens_per_sec=320.0,
            memory_peak_mb=200.0,
            gpu_utilization_percent=80.0,
        )
        
        data = batch_metrics.to_dict()
        assert data['strategy'] == "static_batching"
        assert data['batch_size'] == 8
        
        print("✓ Metrics dataclasses tests passed")
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization utilities (without actual plotting)."""
    print("\nTesting visualization setup...")
    
    try:
        from benchmarks.visualization.plots import BenchmarkVisualizer
        from benchmarks.core.kv_cache_benchmark import KVCacheMetrics
        import os
        
        # Create temp directory
        temp_dir = "/tmp/benchmark_test"
        os.makedirs(temp_dir, exist_ok=True)
        
        visualizer = BenchmarkVisualizer(output_dir=temp_dir)
        
        # Create sample metrics
        metrics = KVCacheMetrics(
            sequence_length=1024,
            batch_size=1,
            num_layers=12,
            kv_cache_size_bytes=1024*1024,
            memory_allocated_mb=100.0,
            memory_reserved_mb=120.0,
            time_per_token_ms=10.0,
            throughput_tokens_per_sec=100.0,
            gpu_memory_percent=50.0,
        )
        
        # Test that we can create visualizations (they'll be saved to temp dir)
        visualizer.plot_cost_per_request(
            sequence_lengths=[128, 256, 512],
            costs=[0.001, 0.002, 0.004],
            save_path=os.path.join(temp_dir, "test_cost.png"),
        )
        
        assert os.path.exists(os.path.join(temp_dir, "test_cost.png"))
        
        print("✓ Visualization tests passed")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("BENCHMARK SETUP VERIFICATION")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Cost Analysis", test_cost_analysis()))
    results.append(("Metrics", test_metrics_dataclasses()))
    results.append(("Visualization", test_visualization()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("  1. Run: python examples/run_kv_benchmark.py --generate-plots")
        print("  2. Run: python examples/run_all_benchmarks.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
