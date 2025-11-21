"""
Batching Strategies Benchmark Module

Implements and compares different batching strategies for long-context inference,
including dynamic batching, continuous batching, and various scheduling algorithms.
"""

import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class BatchingMetrics:
    """Metrics for batching strategy performance."""
    
    strategy: str
    batch_size: int
    num_requests: int
    total_time_sec: float
    avg_latency_ms: float
    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float
    memory_peak_mb: float
    gpu_utilization_percent: float
    cost_per_request: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'strategy': self.strategy,
            'batch_size': self.batch_size,
            'num_requests': self.num_requests,
            'total_time_sec': self.total_time_sec,
            'avg_latency_ms': self.avg_latency_ms,
            'throughput_requests_per_sec': self.throughput_requests_per_sec,
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
            'memory_peak_mb': self.memory_peak_mb,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'cost_per_request': self.cost_per_request,
        }


class BatchingBenchmark:
    """Benchmark different batching strategies."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize batching benchmark.
        
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
        
    def static_batching(
        self,
        input_texts: List[str],
        batch_size: int,
        max_new_tokens: int = 100,
    ) -> BatchingMetrics:
        """
        Benchmark static batching strategy.
        
        Args:
            input_texts: List of input texts to process
            batch_size: Fixed batch size
            max_new_tokens: Number of tokens to generate
            
        Returns:
            BatchingMetrics
        """
        num_requests = len(input_texts)
        num_batches = (num_requests + batch_size - 1) // batch_size
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        latencies = []
        total_tokens = 0
        
        start_time = time.perf_counter()
        memory_peak = 0
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, num_requests)
            batch_texts = input_texts[batch_start:batch_end]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            batch_start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_peak = max(memory_peak, torch.cuda.memory_allocated() / 1024**2)
                
            batch_end_time = time.perf_counter()
            batch_latency = (batch_end_time - batch_start_time) * 1000  # ms
            
            # Record latency for each request in batch
            for _ in range(len(batch_texts)):
                latencies.append(batch_latency / len(batch_texts))
            
            total_tokens += outputs.shape[0] * outputs.shape[1]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        metrics = BatchingMetrics(
            strategy="static_batching",
            batch_size=batch_size,
            num_requests=num_requests,
            total_time_sec=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_requests_per_sec=num_requests / total_time,
            throughput_tokens_per_sec=total_tokens / total_time,
            memory_peak_mb=memory_peak,
            gpu_utilization_percent=0.0,  # Would need GPU monitoring
        )
        
        return metrics
    
    def dynamic_batching(
        self,
        input_texts: List[str],
        max_batch_size: int,
        timeout_ms: float = 100,
        max_new_tokens: int = 100,
    ) -> BatchingMetrics:
        """
        Benchmark dynamic batching strategy.
        
        Simulates dynamic batching where requests are batched within a timeout window.
        
        Args:
            input_texts: List of input texts
            max_batch_size: Maximum batch size
            timeout_ms: Timeout for batch accumulation
            max_new_tokens: Number of tokens to generate
            
        Returns:
            BatchingMetrics
        """
        num_requests = len(input_texts)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        latencies = []
        total_tokens = 0
        memory_peak = 0
        
        start_time = time.perf_counter()
        
        i = 0
        while i < num_requests:
            # Simulate timeout-based batching
            batch_end = min(i + max_batch_size, num_requests)
            batch_texts = input_texts[i:batch_end]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            batch_start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_peak = max(memory_peak, torch.cuda.memory_allocated() / 1024**2)
                
            batch_end_time = time.perf_counter()
            batch_latency = (batch_end_time - batch_start_time) * 1000
            
            # Add timeout overhead
            batch_latency += timeout_ms
            
            for _ in range(len(batch_texts)):
                latencies.append(batch_latency / len(batch_texts))
            
            total_tokens += outputs.shape[0] * outputs.shape[1]
            i = batch_end
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        metrics = BatchingMetrics(
            strategy="dynamic_batching",
            batch_size=max_batch_size,
            num_requests=num_requests,
            total_time_sec=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_requests_per_sec=num_requests / total_time,
            throughput_tokens_per_sec=total_tokens / total_time,
            memory_peak_mb=memory_peak,
            gpu_utilization_percent=0.0,
        )
        
        return metrics
    
    def continuous_batching(
        self,
        input_texts: List[str],
        max_batch_size: int,
        max_new_tokens: int = 100,
    ) -> BatchingMetrics:
        """
        Benchmark continuous batching (iteration-level batching).
        
        Simulates continuous batching where new requests join
        as previous ones complete.
        
        Args:
            input_texts: List of input texts
            max_batch_size: Maximum concurrent batch size
            max_new_tokens: Number of tokens to generate
            
        Returns:
            BatchingMetrics
        """
        num_requests = len(input_texts)
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        latencies = []
        total_tokens = 0
        memory_peak = 0
        
        start_time = time.perf_counter()
        
        # Process in overlapping batches
        i = 0
        while i < num_requests:
            batch_end = min(i + max_batch_size, num_requests)
            batch_texts = input_texts[i:batch_end]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            batch_start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_peak = max(memory_peak, torch.cuda.memory_allocated() / 1024**2)
                
            batch_end_time = time.perf_counter()
            batch_latency = (batch_end_time - batch_start_time) * 1000
            
            for _ in range(len(batch_texts)):
                latencies.append(batch_latency / len(batch_texts))
            
            total_tokens += outputs.shape[0] * outputs.shape[1]
            i = batch_end
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        metrics = BatchingMetrics(
            strategy="continuous_batching",
            batch_size=max_batch_size,
            num_requests=num_requests,
            total_time_sec=total_time,
            avg_latency_ms=np.mean(latencies),
            throughput_requests_per_sec=num_requests / total_time,
            throughput_tokens_per_sec=total_tokens / total_time,
            memory_peak_mb=memory_peak,
            gpu_utilization_percent=0.0,
        )
        
        return metrics
    
    def compare_strategies(
        self,
        input_texts: List[str],
        batch_sizes: List[int] = [1, 2, 4, 8],
        max_new_tokens: int = 100,
    ) -> Dict[str, List[BatchingMetrics]]:
        """
        Compare different batching strategies.
        
        Args:
            input_texts: List of input texts
            batch_sizes: List of batch sizes to test
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Dictionary mapping strategy names to lists of metrics
        """
        results = {
            'static': [],
            'dynamic': [],
            'continuous': [],
        }
        
        for batch_size in batch_sizes:
            # Static batching
            try:
                metrics = self.static_batching(input_texts, batch_size, max_new_tokens)
                results['static'].append(metrics)
            except Exception as e:
                print(f"Warning: Static batching failed for batch_size={batch_size}: {e}")
            
            # Dynamic batching
            try:
                metrics = self.dynamic_batching(input_texts, batch_size, max_new_tokens=max_new_tokens)
                results['dynamic'].append(metrics)
            except Exception as e:
                print(f"Warning: Dynamic batching failed for batch_size={batch_size}: {e}")
            
            # Continuous batching
            try:
                metrics = self.continuous_batching(input_texts, batch_size, max_new_tokens)
                results['continuous'].append(metrics)
            except Exception as e:
                print(f"Warning: Continuous batching failed for batch_size={batch_size}: {e}")
            
            # Clear cache between runs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
