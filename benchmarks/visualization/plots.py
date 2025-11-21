"""
Visualization Module for Benchmark Results

Creates comprehensive plots including:
- Cost-per-request curves
- Memory usage graphs
- Throughput comparisons
- Efficiency breakdowns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BenchmarkVisualizer:
    """Create visualizations for benchmark results."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_cost_per_request(
        self,
        sequence_lengths: List[int],
        costs: List[float],
        title: str = "Cost per Request vs Sequence Length",
        save_path: Optional[str] = None,
    ):
        """
        Plot cost per request curve.
        
        Args:
            sequence_lengths: List of sequence lengths
            costs: Corresponding costs
            title: Plot title
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sequence_lengths, costs, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax.set_ylabel('Cost per Request ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(sequence_lengths, costs)):
            ax.annotate(f'${y:.4f}', (x, y), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_memory_usage(
        self,
        sequence_lengths: List[int],
        memory_allocated: List[float],
        memory_reserved: List[float],
        kv_cache_size: List[float],
        title: str = "Memory Usage vs Sequence Length",
        save_path: Optional[str] = None,
    ):
        """
        Plot memory usage breakdown.
        
        Args:
            sequence_lengths: List of sequence lengths
            memory_allocated: Allocated memory (MB)
            memory_reserved: Reserved memory (MB)
            kv_cache_size: KV cache size (MB)
            title: Plot title
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        width = 0.25
        x = np.arange(len(sequence_lengths))
        
        ax.bar(x - width, memory_allocated, width, label='Allocated Memory', alpha=0.8)
        ax.bar(x, memory_reserved, width, label='Reserved Memory', alpha=0.8)
        ax.bar(x + width, kv_cache_size, width, label='KV Cache Size', alpha=0.8)
        
        ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax.set_ylabel('Memory (MB)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequence_lengths)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_throughput_comparison(
        self,
        strategies: List[str],
        throughputs: List[float],
        title: str = "Throughput Comparison Across Strategies",
        save_path: Optional[str] = None,
    ):
        """
        Plot throughput comparison bar chart.
        
        Args:
            strategies: List of strategy names
            throughputs: Corresponding throughputs (tokens/sec)
            title: Plot title
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", len(strategies))
        bars = ax.bar(strategies, throughputs, color=colors, alpha=0.8)
        
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_kv_cache_scaling(
        self,
        metrics_list: List[Any],
        title: str = "KV Cache Efficiency Scaling",
        save_path: Optional[str] = None,
    ):
        """
        Plot KV cache efficiency across sequence lengths.
        
        Args:
            metrics_list: List of KVCacheMetrics objects
            title: Plot title
            save_path: Path to save plot
        """
        seq_lengths = [m.sequence_length for m in metrics_list]
        kv_sizes = [m.kv_cache_size_bytes / (1024**2) for m in metrics_list]  # MB
        throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
        latencies = [m.time_per_token_ms for m in metrics_list]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # KV Cache Size
        axes[0, 0].plot(seq_lengths, kv_sizes, marker='o', linewidth=2, color='#2E86AB')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('KV Cache Size (MB)')
        axes[0, 0].set_title('KV Cache Size vs Sequence Length')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throughput
        axes[0, 1].plot(seq_lengths, throughputs, marker='s', linewidth=2, color='#A23B72')
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Throughput (tokens/sec)')
        axes[0, 1].set_title('Throughput vs Sequence Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Latency
        axes[1, 0].plot(seq_lengths, latencies, marker='^', linewidth=2, color='#F18F01')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Time per Token (ms)')
        axes[1, 0].set_title('Latency vs Sequence Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory efficiency (tokens per MB)
        memory_efficiency = [s / (k if k > 0 else 1) for s, k in zip(seq_lengths, kv_sizes)]
        axes[1, 1].plot(seq_lengths, memory_efficiency, marker='d', linewidth=2, color='#6A994E')
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('Tokens per MB')
        axes[1, 1].set_title('Memory Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_attention_comparison(
        self,
        metrics_dict: Dict[str, Any],
        title: str = "Attention Technique Comparison",
        save_path: Optional[str] = None,
    ):
        """
        Compare different attention techniques.
        
        Args:
            metrics_dict: Dictionary mapping technique names to metrics
            title: Plot title
            save_path: Path to save plot
        """
        techniques = list(metrics_dict.keys())
        memory_reductions = [metrics_dict[t].memory_reduction_percent for t in techniques]
        speedups = [metrics_dict[t].speedup_factor for t in techniques]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Memory reduction
        colors = sns.color_palette("RdYlGn", len(techniques))
        axes[0].barh(techniques, memory_reductions, color=colors, alpha=0.8)
        axes[0].set_xlabel('Memory Reduction (%)', fontsize=12)
        axes[0].set_title('Memory Savings by Technique', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Speedup factor
        axes[1].barh(techniques, speedups, color=colors, alpha=0.8)
        axes[1].set_xlabel('Speedup Factor', fontsize=12)
        axes[1].set_title('Theoretical Speedup', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def plot_batching_comparison(
        self,
        results: Dict[str, List[Any]],
        title: str = "Batching Strategy Comparison",
        save_path: Optional[str] = None,
    ):
        """
        Compare batching strategies.
        
        Args:
            results: Dictionary mapping strategy names to lists of metrics
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for strategy_name, metrics_list in results.items():
            batch_sizes = [m.batch_size for m in metrics_list]
            throughputs = [m.throughput_requests_per_sec for m in metrics_list]
            latencies = [m.avg_latency_ms for m in metrics_list]
            memory_peaks = [m.memory_peak_mb for m in metrics_list]
            
            # Throughput
            axes[0, 0].plot(batch_sizes, throughputs, marker='o', label=strategy_name, linewidth=2)
            
            # Latency
            axes[0, 1].plot(batch_sizes, latencies, marker='s', label=strategy_name, linewidth=2)
            
            # Memory
            axes[1, 0].plot(batch_sizes, memory_peaks, marker='^', label=strategy_name, linewidth=2)
            
            # Efficiency (throughput / memory)
            efficiency = [t / (m if m > 0 else 1) for t, m in zip(throughputs, memory_peaks)]
            axes[1, 1].plot(batch_sizes, efficiency, marker='d', label=strategy_name, linewidth=2)
        
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Throughput (req/sec)')
        axes[0, 0].set_title('Throughput vs Batch Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Avg Latency (ms)')
        axes[0, 1].set_title('Latency vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Memory Usage vs Batch Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Efficiency (req/sec/MB)')
        axes[1, 1].set_title('Memory Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
    
    def create_summary_report(
        self,
        results: Dict[str, Any],
        save_path: str,
    ):
        """
        Create a comprehensive summary report.
        
        Args:
            results: Dictionary of all benchmark results
            save_path: Path to save the report
        """
        with open(save_path, 'w') as f:
            f.write("# Long-Context Efficiency Benchmark Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # Write summary sections for each benchmark type
            if 'kv_cache' in results:
                f.write("### KV Cache Efficiency\n\n")
                f.write("Key findings from KV cache benchmarks...\n\n")
            
            if 'squeezed_attention' in results:
                f.write("### Squeezed Attention\n\n")
                f.write("Analysis of attention compression techniques...\n\n")
            
            if 'batching' in results:
                f.write("### Batching Strategies\n\n")
                f.write("Comparison of different batching approaches...\n\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("See accompanying plots for detailed visualizations.\n")
        
        print(f"Summary report saved to {save_path}")
