"""
Cost Analysis Module

Calculate and analyze costs for long-context inference across
different configurations, models, and cloud providers.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


# Pricing data (USD per 1000 tokens)
# These are example prices - adjust based on actual provider pricing
PRICING = {
    'gpt-4-turbo': {
        'input': 0.01,
        'output': 0.03,
    },
    'gpt-4-32k': {
        'input': 0.06,
        'output': 0.12,
    },
    'claude-2': {
        'input': 0.008,
        'output': 0.024,
    },
    'llama-2-70b': {
        'input': 0.0007,
        'output': 0.0009,
    },
    'custom-model': {
        'input': 0.001,
        'output': 0.002,
    },
}

# GPU instance pricing (USD per hour)
GPU_PRICING = {
    'a100-40gb': 3.67,
    'a100-80gb': 4.10,
    'v100-32gb': 2.48,
    'h100-80gb': 8.00,
    't4-16gb': 0.95,
}


@dataclass
class CostEstimate:
    """Cost estimate for inference."""
    
    model_name: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cost_per_request: float
    gpu_hours: float = 0.0
    gpu_cost: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost,
            'cost_per_request': self.cost_per_request,
            'gpu_hours': self.gpu_hours,
            'gpu_cost': self.gpu_cost,
        }


class CostAnalyzer:
    """Analyze costs for long-context inference."""
    
    def __init__(
        self,
        model_name: str = 'custom-model',
        gpu_type: Optional[str] = None,
    ):
        """
        Initialize cost analyzer.
        
        Args:
            model_name: Name of the model (for pricing lookup)
            gpu_type: Type of GPU instance (for self-hosted costs)
        """
        self.model_name = model_name
        self.gpu_type = gpu_type
        
        if model_name in PRICING:
            self.pricing = PRICING[model_name]
        else:
            self.pricing = PRICING['custom-model']
            
        if gpu_type and gpu_type in GPU_PRICING:
            self.gpu_price_per_hour = GPU_PRICING[gpu_type]
        else:
            self.gpu_price_per_hour = 0.0
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        num_requests: int = 1,
    ) -> CostEstimate:
        """
        Calculate cost for inference.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_requests: Number of requests
            
        Returns:
            CostEstimate object
        """
        # Calculate token costs
        input_cost = (input_tokens / 1000) * self.pricing['input'] * num_requests
        output_cost = (output_tokens / 1000) * self.pricing['output'] * num_requests
        total_cost = input_cost + output_cost
        cost_per_request = total_cost / num_requests if num_requests > 0 else 0
        
        return CostEstimate(
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            cost_per_request=cost_per_request,
        )
    
    def calculate_gpu_cost(
        self,
        inference_time_seconds: float,
        num_requests: int = 1,
    ) -> float:
        """
        Calculate GPU compute cost.
        
        Args:
            inference_time_seconds: Total inference time
            num_requests: Number of requests
            
        Returns:
            Total GPU cost in USD
        """
        hours = inference_time_seconds / 3600
        return hours * self.gpu_price_per_hour
    
    def calculate_total_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        inference_time_seconds: float,
        num_requests: int = 1,
    ) -> CostEstimate:
        """
        Calculate total cost including GPU time.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            inference_time_seconds: Total inference time
            num_requests: Number of requests
            
        Returns:
            CostEstimate with GPU costs included
        """
        estimate = self.calculate_cost(input_tokens, output_tokens, num_requests)
        
        # Add GPU costs if applicable
        if self.gpu_price_per_hour > 0:
            gpu_hours = inference_time_seconds / 3600
            gpu_cost = self.calculate_gpu_cost(inference_time_seconds, num_requests)
            
            estimate.gpu_hours = gpu_hours
            estimate.gpu_cost = gpu_cost
            estimate.total_cost += gpu_cost
            estimate.cost_per_request = estimate.total_cost / num_requests if num_requests > 0 else 0
        
        return estimate
    
    def compare_sequence_lengths(
        self,
        sequence_lengths: List[int],
        output_tokens: int = 100,
    ) -> List[CostEstimate]:
        """
        Compare costs across different sequence lengths.
        
        Args:
            sequence_lengths: List of input sequence lengths
            output_tokens: Number of output tokens
            
        Returns:
            List of CostEstimate objects
        """
        estimates = []
        
        for seq_len in sequence_lengths:
            estimate = self.calculate_cost(seq_len, output_tokens)
            estimates.append(estimate)
        
        return estimates
    
    def cost_efficiency_metrics(
        self,
        cost: float,
        total_tokens: int,
        throughput_tokens_per_sec: float,
    ) -> Dict:
        """
        Calculate cost efficiency metrics.
        
        Args:
            cost: Total cost
            total_tokens: Total number of tokens processed
            throughput_tokens_per_sec: Throughput in tokens/sec
            
        Returns:
            Dictionary of efficiency metrics
        """
        # Cost per token
        cost_per_token = cost / total_tokens if total_tokens > 0 else 0
        
        # Tokens per dollar
        tokens_per_dollar = total_tokens / cost if cost > 0 else 0
        
        # Cost-adjusted throughput (tokens per second per dollar)
        cost_adjusted_throughput = throughput_tokens_per_sec / cost if cost > 0 else 0
        
        return {
            'cost_per_token': cost_per_token,
            'tokens_per_dollar': tokens_per_dollar,
            'cost_adjusted_throughput': cost_adjusted_throughput,
        }
    
    def break_even_analysis(
        self,
        api_cost_per_1k_tokens: float,
        gpu_cost_per_hour: float,
        tokens_per_second: float,
    ) -> Dict:
        """
        Calculate break-even point between API and self-hosting.
        
        Args:
            api_cost_per_1k_tokens: API cost per 1000 tokens
            gpu_cost_per_hour: GPU instance cost per hour
            tokens_per_second: Throughput in tokens/second
            
        Returns:
            Break-even analysis results
        """
        # Tokens per hour at given throughput
        tokens_per_hour = tokens_per_second * 3600
        
        # Cost per hour for API
        api_cost_per_hour = (tokens_per_hour / 1000) * api_cost_per_1k_tokens
        
        # Break-even occurs when costs are equal
        if api_cost_per_hour > 0:
            cost_ratio = gpu_cost_per_hour / api_cost_per_hour
        else:
            cost_ratio = float('inf')
        
        # Monthly costs (assuming 730 hours/month)
        hours_per_month = 730
        monthly_api_cost = api_cost_per_hour * hours_per_month
        monthly_gpu_cost = gpu_cost_per_hour * hours_per_month
        
        return {
            'tokens_per_hour': tokens_per_hour,
            'api_cost_per_hour': api_cost_per_hour,
            'gpu_cost_per_hour': gpu_cost_per_hour,
            'cost_ratio': cost_ratio,
            'monthly_api_cost': monthly_api_cost,
            'monthly_gpu_cost': monthly_gpu_cost,
            'cheaper_option': 'self-hosted' if gpu_cost_per_hour < api_cost_per_hour else 'api',
            'monthly_savings': abs(monthly_api_cost - monthly_gpu_cost),
        }


class WasteAnalyzer:
    """Analyze where long-context models waste compute."""
    
    @staticmethod
    def analyze_attention_waste(
        sequence_length: int,
        actual_relevant_tokens: int,
    ) -> Dict:
        """
        Analyze wasted attention computation.
        
        Args:
            sequence_length: Total sequence length
            actual_relevant_tokens: Number of actually relevant tokens
            
        Returns:
            Waste analysis results
        """
        # Full attention computes O(n^2) operations
        total_attention_ops = sequence_length ** 2
        necessary_attention_ops = actual_relevant_tokens * sequence_length
        wasted_ops = total_attention_ops - necessary_attention_ops
        waste_percent = (wasted_ops / total_attention_ops) * 100 if total_attention_ops > 0 else 0
        
        return {
            'total_operations': total_attention_ops,
            'necessary_operations': necessary_attention_ops,
            'wasted_operations': wasted_ops,
            'waste_percent': waste_percent,
        }
    
    @staticmethod
    def analyze_kv_cache_waste(
        max_sequence_length: int,
        actual_sequence_length: int,
        batch_size: int,
        num_layers: int,
        hidden_size: int,
    ) -> Dict:
        """
        Analyze wasted KV cache memory.
        
        Args:
            max_sequence_length: Maximum allocated sequence length
            actual_sequence_length: Actual used sequence length
            batch_size: Batch size
            num_layers: Number of layers
            hidden_size: Hidden dimension size
            
        Returns:
            KV cache waste analysis
        """
        # Bytes per element (float16 = 2 bytes)
        bytes_per_element = 2
        
        # Total allocated (K and V)
        total_allocated = 2 * batch_size * num_layers * max_sequence_length * hidden_size * bytes_per_element
        
        # Actually used
        actually_used = 2 * batch_size * num_layers * actual_sequence_length * hidden_size * bytes_per_element
        
        # Wasted
        wasted = total_allocated - actually_used
        waste_percent = (wasted / total_allocated) * 100 if total_allocated > 0 else 0
        
        return {
            'total_allocated_mb': total_allocated / (1024 ** 2),
            'actually_used_mb': actually_used / (1024 ** 2),
            'wasted_mb': wasted / (1024 ** 2),
            'waste_percent': waste_percent,
        }
    
    @staticmethod
    def analyze_padding_waste(
        sequence_lengths: List[int],
    ) -> Dict:
        """
        Analyze waste from padding in batched inference.
        
        Args:
            sequence_lengths: List of sequence lengths in batch
            
        Returns:
            Padding waste analysis
        """
        if not sequence_lengths:
            return {
                'max_length': 0,
                'avg_length': 0,
                'total_tokens': 0,
                'padding_tokens': 0,
                'waste_percent': 0,
            }
        
        max_length = max(sequence_lengths)
        avg_length = np.mean(sequence_lengths)
        total_tokens = max_length * len(sequence_lengths)
        actual_tokens = sum(sequence_lengths)
        padding_tokens = total_tokens - actual_tokens
        waste_percent = (padding_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        return {
            'max_length': max_length,
            'avg_length': avg_length,
            'total_tokens': total_tokens,
            'actual_tokens': actual_tokens,
            'padding_tokens': padding_tokens,
            'waste_percent': waste_percent,
        }
