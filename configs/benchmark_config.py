"""
Example configuration for KV cache benchmarks.
"""

# Model configuration
MODEL_CONFIG = {
    'model_name': 'gpt2',  # Start with small model for testing
    'device': 'cuda',  # 'cuda' or 'cpu'
    'use_cache': True,
}

# Benchmark sequences
SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]

# Number of tokens to generate
MAX_NEW_TOKENS = 100

# Batching configurations
BATCH_SIZES = [1, 2, 4, 8]

# Sample text for benchmarking
SAMPLE_TEXT = """
Artificial intelligence and machine learning have revolutionized the field of natural language processing.
Large language models with extended context windows enable unprecedented capabilities in understanding
and generating human-like text. However, these models come with significant computational costs,
particularly when processing long contexts that span thousands or even hundreds of thousands of tokens.
The key challenge lies in efficiently managing the key-value cache, which grows quadratically with
sequence length in traditional attention mechanisms.
"""

# Output directory
OUTPUT_DIR = 'outputs'

# Visualization settings
SAVE_PLOTS = True
PLOT_FORMAT = 'png'  # 'png' or 'pdf'

# Cost analysis settings
COST_CONFIG = {
    'model_name': 'custom-model',
    'gpu_type': 'a100-40gb',
}
