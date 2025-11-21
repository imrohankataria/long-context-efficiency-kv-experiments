from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kv-cache-benchmarks",
    version="0.1.0",
    author="KV Cache Efficiency Team",
    description="Benchmarking tools for KV cache efficiency and long-context inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "psutil>=5.9.0",
        "memory_profiler>=0.61.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "scipy>=1.11.0",
    ],
)
