# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-21

### Added
- Initial release of KV cache efficiency benchmarking tools
- Core benchmarking modules:
  - KV cache efficiency measurement
  - Squeezed attention benchmarks (sliding window, sparse attention)
  - Batching strategy comparison (static, dynamic, continuous)
- Cost analysis utilities:
  - Cost per token calculations
  - GPU compute cost analysis
  - Break-even analysis for API vs self-hosting
  - Waste analysis (attention, KV cache, padding)
- Visualization suite:
  - Cost-per-request curves
  - Memory usage graphs
  - Throughput comparisons
  - KV cache scaling plots
  - Batching strategy comparisons
- Example scripts:
  - Comprehensive benchmark runner
  - KV cache-only benchmark
  - Batching strategy benchmark
  - Setup verification tests
- Documentation:
  - Comprehensive README with usage examples
  - Contributing guidelines
  - MIT License
  - Configuration templates

### Features
- Support for multiple model architectures via HuggingFace Transformers
- Configurable sequence lengths and batch sizes
- Automated visualization generation
- JSON output for easy integration
- Extensible metrics classes
- GPU and CPU support

### Benchmarks Included
- KV cache size scaling
- Memory allocation patterns
- Throughput vs sequence length
- Latency measurements
- Attention compression techniques
- Batching efficiency
- Cost analysis across different configurations

## [Unreleased]

### Planned
- Integration with vLLM and Text Generation Inference
- Support for PagedAttention
- Flash Attention benchmarks
- Multi-GPU benchmarking
- Distributed inference analysis
- Real-world workload traces
- Interactive Jupyter notebooks
- Additional model architectures
- Streaming inference benchmarks
