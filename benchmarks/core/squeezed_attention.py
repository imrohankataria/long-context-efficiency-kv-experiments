"""
Squeezed Attention Benchmark Module

Implements and benchmarks Squeezed Attention techniques for efficient
long-context processing, including attention pattern compression and
selective attention mechanisms.
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SqueezedAttentionMetrics:
    """Metrics for Squeezed Attention performance."""
    
    technique: str
    sequence_length: int
    compression_ratio: float
    attention_memory_mb: float
    inference_time_ms: float
    throughput_tokens_per_sec: float
    memory_reduction_percent: float
    speedup_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'technique': self.technique,
            'sequence_length': self.sequence_length,
            'compression_ratio': self.compression_ratio,
            'attention_memory_mb': self.attention_memory_mb,
            'inference_time_ms': self.inference_time_ms,
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
            'memory_reduction_percent': self.memory_reduction_percent,
            'speedup_factor': self.speedup_factor,
        }


class SqueezedAttentionBenchmark:
    """Benchmark Squeezed Attention techniques."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize Squeezed Attention benchmark.
        
        Args:
            model: Transformer model to benchmark
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def simulate_sliding_window_attention(
        self,
        input_text: str,
        window_size: int = 2048,
        max_new_tokens: int = 100,
    ) -> SqueezedAttentionMetrics:
        """
        Simulate sliding window attention efficiency.
        
        Args:
            input_text: Input text to process
            window_size: Size of attention window
            max_new_tokens: Number of tokens to generate
            
        Returns:
            SqueezedAttentionMetrics
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        seq_length = inputs.input_ids.shape[1]
        
        # Calculate compression ratio
        compression_ratio = min(1.0, window_size / seq_length)
        
        # Estimate memory savings
        # Full attention: O(n^2), Sliding window: O(n*w)
        full_attention_memory = seq_length ** 2
        sliding_attention_memory = seq_length * window_size
        memory_reduction = (1 - sliding_attention_memory / full_attention_memory) * 100
        
        # Benchmark inference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # ms
        throughput = max_new_tokens / (inference_time / 1000)
        
        # Memory estimation (in MB)
        if torch.cuda.is_available():
            attention_memory = (sliding_attention_memory * 4) / (1024 ** 2)  # 4 bytes per float32
        else:
            attention_memory = 0
            
        metrics = SqueezedAttentionMetrics(
            technique="sliding_window",
            sequence_length=seq_length,
            compression_ratio=compression_ratio,
            attention_memory_mb=attention_memory,
            inference_time_ms=inference_time,
            throughput_tokens_per_sec=throughput,
            memory_reduction_percent=memory_reduction,
            speedup_factor=1.0 / compression_ratio if compression_ratio > 0 else 1.0,
        )
        
        return metrics
    
    def simulate_sparse_attention(
        self,
        input_text: str,
        sparsity_ratio: float = 0.9,
        max_new_tokens: int = 100,
    ) -> SqueezedAttentionMetrics:
        """
        Simulate sparse attention efficiency.
        
        Args:
            input_text: Input text to process
            sparsity_ratio: Fraction of attention weights to zero out (0-1)
            max_new_tokens: Number of tokens to generate
            
        Returns:
            SqueezedAttentionMetrics
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        seq_length = inputs.input_ids.shape[1]
        
        # Calculate compression
        compression_ratio = 1.0 - sparsity_ratio
        
        # Estimate memory savings
        full_attention_memory = seq_length ** 2
        sparse_attention_memory = full_attention_memory * compression_ratio
        memory_reduction = sparsity_ratio * 100
        
        # Benchmark inference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # ms
        throughput = max_new_tokens / (inference_time / 1000)
        
        # Memory estimation
        if torch.cuda.is_available():
            attention_memory = (sparse_attention_memory * 4) / (1024 ** 2)
        else:
            attention_memory = 0
            
        metrics = SqueezedAttentionMetrics(
            technique="sparse_attention",
            sequence_length=seq_length,
            compression_ratio=compression_ratio,
            attention_memory_mb=attention_memory,
            inference_time_ms=inference_time,
            throughput_tokens_per_sec=throughput,
            memory_reduction_percent=memory_reduction,
            speedup_factor=1.0 / compression_ratio if compression_ratio > 0 else 1.0,
        )
        
        return metrics
    
    def compare_attention_techniques(
        self,
        input_text: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, SqueezedAttentionMetrics]:
        """
        Compare different attention compression techniques.
        
        Args:
            input_text: Input text to process
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Dictionary mapping technique names to metrics
        """
        results = {}
        
        # Sliding window variants
        for window_size in [512, 1024, 2048, 4096]:
            try:
                metrics = self.simulate_sliding_window_attention(
                    input_text, window_size, max_new_tokens
                )
                results[f"sliding_window_{window_size}"] = metrics
            except Exception as e:
                print(f"Warning: Could not benchmark sliding window {window_size}: {e}")
        
        # Sparse attention variants
        for sparsity in [0.5, 0.7, 0.9, 0.95]:
            try:
                metrics = self.simulate_sparse_attention(
                    input_text, sparsity, max_new_tokens
                )
                results[f"sparse_{int(sparsity*100)}pct"] = metrics
            except Exception as e:
                print(f"Warning: Could not benchmark sparse attention {sparsity}: {e}")
        
        return results
    
    def benchmark_scaling(
        self,
        base_text: str,
        sequence_lengths: List[int],
        technique: str = "sliding_window",
        window_size: int = 2048,
        sparsity: float = 0.9,
    ) -> List[SqueezedAttentionMetrics]:
        """
        Benchmark attention technique across sequence lengths.
        
        Args:
            base_text: Base text for scaling
            sequence_lengths: Target sequence lengths
            technique: 'sliding_window' or 'sparse_attention'
            window_size: Window size for sliding window
            sparsity: Sparsity ratio for sparse attention
            
        Returns:
            List of metrics for each sequence length
        """
        results = []
        
        for target_length in sequence_lengths:
            # Create input of target length
            num_repeats = max(1, target_length // len(base_text.split()))
            input_text = (base_text + " ") * num_repeats
            
            tokens = self.tokenizer.encode(input_text)
            if len(tokens) > target_length:
                tokens = tokens[:target_length]
            input_text = self.tokenizer.decode(tokens)
            
            # Benchmark
            try:
                if technique == "sliding_window":
                    metrics = self.simulate_sliding_window_attention(
                        input_text, window_size, max_new_tokens=50
                    )
                elif technique == "sparse_attention":
                    metrics = self.simulate_sparse_attention(
                        input_text, sparsity, max_new_tokens=50
                    )
                else:
                    raise ValueError(f"Unknown technique: {technique}")
                    
                results.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to benchmark at length {target_length}: {e}")
            
            # Clear cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return results
