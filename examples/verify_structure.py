#!/usr/bin/env python3
"""
Verify repository structure and basic imports without requiring heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists


def check_directory_structure():
    """Check that all expected directories exist."""
    print("\n" + "=" * 80)
    print("DIRECTORY STRUCTURE VERIFICATION")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "benchmarks",
        "benchmarks/core",
        "benchmarks/strategies",
        "benchmarks/visualization",
        "benchmarks/utils",
        "configs",
        "examples",
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}/")
        all_exist = all_exist and exists
    
    return all_exist


def check_core_files():
    """Check that all core files exist."""
    print("\n" + "=" * 80)
    print("CORE FILES VERIFICATION")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        ("README.md", "README"),
        ("requirements.txt", "Requirements"),
        ("setup.py", "Setup script"),
        (".gitignore", "Git ignore"),
        ("LICENSE", "License"),
        ("CONTRIBUTING.md", "Contributing guide"),
        ("CHANGELOG.md", "Changelog"),
        ("benchmarks/__init__.py", "Benchmarks package"),
        ("benchmarks/core/__init__.py", "Core package"),
        ("benchmarks/core/kv_cache_benchmark.py", "KV cache benchmark"),
        ("benchmarks/core/squeezed_attention.py", "Squeezed attention"),
        ("benchmarks/strategies/__init__.py", "Strategies package"),
        ("benchmarks/strategies/batching.py", "Batching strategies"),
        ("benchmarks/visualization/__init__.py", "Visualization package"),
        ("benchmarks/visualization/plots.py", "Visualization plots"),
        ("benchmarks/utils/__init__.py", "Utils package"),
        ("benchmarks/utils/cost_analysis.py", "Cost analysis"),
        ("benchmarks/utils/memory_profiler.py", "Memory profiler"),
        ("configs/benchmark_config.py", "Benchmark config"),
        ("examples/run_all_benchmarks.py", "Comprehensive benchmark script"),
        ("examples/run_kv_benchmark.py", "KV benchmark script"),
        ("examples/run_batching_benchmark.py", "Batching benchmark script"),
        ("examples/test_setup.py", "Setup test script"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        full_path = base_dir / filepath
        all_exist = check_file_exists(full_path, description) and all_exist
    
    return all_exist


def check_python_syntax():
    """Check Python files for syntax errors."""
    print("\n" + "=" * 80)
    print("PYTHON SYNTAX VERIFICATION")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    
    python_files = []
    for pattern in ["benchmarks/**/*.py", "examples/*.py", "configs/*.py"]:
        python_files.extend(base_dir.glob(pattern))
    
    all_valid = True
    errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
            print(f"  ✓ {py_file.relative_to(base_dir)}")
        except SyntaxError as e:
            print(f"  ✗ {py_file.relative_to(base_dir)}: {e}")
            errors.append((py_file, e))
            all_valid = False
    
    return all_valid, errors


def check_readme_content():
    """Check that README has expected sections."""
    print("\n" + "=" * 80)
    print("README CONTENT VERIFICATION")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    readme_path = base_dir / "README.md"
    
    if not readme_path.exists():
        print("  ✗ README.md not found")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    expected_sections = [
        "Features",
        "Quick Start",
        "Installation",
        "Usage",
        "KV Cache",
        "Squeezed Attention",
        "Batching",
        "Cost",
        "Repository Structure",
    ]
    
    all_present = True
    for section in expected_sections:
        present = section.lower() in content.lower()
        status = "✓" if present else "✗"
        print(f"  {status} Section: {section}")
        all_present = all_present and present
    
    # Check for code examples
    has_code_blocks = "```python" in content
    print(f"  {'✓' if has_code_blocks else '✗'} Code examples present")
    
    return all_present and has_code_blocks


def main():
    """Run all verification checks."""
    print("=" * 80)
    print("REPOSITORY VERIFICATION (No Dependencies Required)")
    print("=" * 80)
    
    results = []
    
    # Run checks
    results.append(("Directory Structure", check_directory_structure()))
    results.append(("Core Files", check_core_files()))
    
    syntax_valid, syntax_errors = check_python_syntax()
    results.append(("Python Syntax", syntax_valid))
    
    results.append(("README Content", check_readme_content()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ Repository structure verification complete!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full tests: python examples/test_setup.py")
        print("  3. Run benchmarks: python examples/run_all_benchmarks.py")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        if syntax_errors:
            print("\nSyntax errors found:")
            for filepath, error in syntax_errors:
                print(f"  {filepath}: {error}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
