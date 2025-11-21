"""
KV Cache Efficiency Measurement Module

This module provides tools to measure and analyze KV cache efficiency
in transformer models for long-context inference.
"""

import torch
import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class KVCacheMetrics:
    """Metrics for KV cache performance."""
    
    sequence_length: int
    batch_size: int
    num_layers: int
    kv_cache_size_bytes: int
    memory_allocated_mb: float
    memory_reserved_mb: float
    time_per_token_ms: float
    throughput_tokens_per_sec: float
    gpu_memory_percent: float
    cost_per_1k_tokens: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'num_layers': self.num_layers,
            'kv_cache_size_bytes': self.kv_cache_size_bytes,
            'memory_allocated_mb': self.memory_allocated_mb,
            'memory_reserved_mb': self.memory_reserved_mb,
            'time_per_token_ms': self.time_per_token_ms,
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
            'gpu_memory_percent': self.gpu_memory_percent,
            'cost_per_1k_tokens': self.cost_per_1k_tokens,
        }


class KVCacheBenchmark:
    """Benchmark KV cache efficiency for transformer models."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        use_cache: bool = True,
    ):
        """
        Initialize KV cache benchmark.
        
        Args:
            model: Transformer model to benchmark
            tokenizer: Tokenizer for the model
            device: Device to run on ('cuda' or 'cpu')
            use_cache: Whether to use KV caching
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cache = use_cache
        self.model.to(device)
        self.model.eval()
        
    def measure_kv_cache_size(
        self,
        sequence_length: int,
        batch_size: int = 1,
    ) -> int:
        """
        Calculate theoretical KV cache size in bytes.
        
        Args:
            sequence_length: Length of input sequence
            batch_size: Batch size
            
        Returns:
            KV cache size in bytes
        """
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        
        # KV cache stores both keys and values for each layer
        # Size = 2 (K,V) * batch * num_layers * num_heads * seq_len * head_dim * bytes_per_element
        bytes_per_element = 2 if self.model.dtype == torch.float16 else 4
        
        kv_cache_size = (
            2 * batch_size * num_layers * num_heads * sequence_length * head_dim * bytes_per_element
        )
        
        return kv_cache_size
    
    def benchmark_inference(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        num_warmup: int = 3,
    ) -> KVCacheMetrics:
        """
        Benchmark inference with KV caching.
        
        Args:
            input_text: Input text to process
            max_new_tokens: Number of new tokens to generate
            num_warmup: Number of warmup iterations
            
        Returns:
            KVCacheMetrics object with benchmark results
        """
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Warmup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    use_cache=self.use_cache,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        
        # Clear cache and collect garbage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / 1024**2
            
        # Benchmark generation
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
        else:
            memory_after = 0
            memory_reserved = 0
            
        # Calculate metrics
        total_time = end_time - start_time
        output_length = outputs.sequences.shape[1]
        generated_tokens = output_length - input_length
        
        time_per_token_ms = (total_time / generated_tokens) * 1000
        throughput = generated_tokens / total_time
        
        # Calculate KV cache size
        kv_cache_size = self.measure_kv_cache_size(
            sequence_length=output_length,
            batch_size=1,
        )
        
        # GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_percent = (memory_after / (torch.cuda.get_device_properties(0).total_memory / 1024**2)) * 100
        else:
            gpu_memory_percent = 0
        
        metrics = KVCacheMetrics(
            sequence_length=output_length,
            batch_size=1,
            num_layers=self.model.config.num_hidden_layers,
            kv_cache_size_bytes=kv_cache_size,
            memory_allocated_mb=memory_after,
            memory_reserved_mb=memory_reserved,
            time_per_token_ms=time_per_token_ms,
            throughput_tokens_per_sec=throughput,
            gpu_memory_percent=gpu_memory_percent,
        )
        
        return metrics
    
    def compare_with_without_cache(
        self,
        input_text: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, KVCacheMetrics]:
        """
        Compare inference with and without KV caching.
        
        Args:
            input_text: Input text to process
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            Dictionary with 'with_cache' and 'without_cache' metrics
        """
        # Benchmark with cache
        self.use_cache = True
        metrics_with_cache = self.benchmark_inference(input_text, max_new_tokens)
        
        # Benchmark without cache
        self.use_cache = False
        metrics_without_cache = self.benchmark_inference(input_text, max_new_tokens)
        
        # Restore cache setting
        self.use_cache = True
        
        return {
            'with_cache': metrics_with_cache,
            'without_cache': metrics_without_cache,
        }
    
    def benchmark_scaling(
        self,
        base_text: str,
        sequence_lengths: List[int],
        tokens_to_generate: int = 50,
    ) -> List[KVCacheMetrics]:
        """
        Benchmark KV cache efficiency across different sequence lengths.
        
        Args:
            base_text: Base text to repeat for different lengths
            sequence_lengths: List of target sequence lengths
            tokens_to_generate: Number of tokens to generate for each length
            
        Returns:
            List of KVCacheMetrics for each sequence length
        """
        results = []
        
        for target_length in sequence_lengths:
            # Create input of approximately target length
            num_repeats = max(1, target_length // len(base_text.split()))
            input_text = (base_text + " ") * num_repeats
            
            # Truncate to target length (approximately)
            tokens = self.tokenizer.encode(input_text)
            if len(tokens) > target_length:
                tokens = tokens[:target_length]
            input_text = self.tokenizer.decode(tokens)
            
            # Benchmark
            metrics = self.benchmark_inference(input_text, tokens_to_generate)
            results.append(metrics)
            
            # Clear cache between runs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        return results
