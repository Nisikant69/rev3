# backend/performance_benchmarking.py
"""
Performance Benchmarking System
Provides comprehensive performance analysis, benchmarking, and regression detection.
"""

import json
import asyncio
import time
import psutil
import subprocess
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import re

from config import DATA_DIR
from utils import detect_language_from_filename


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    CPU_TIME = "cpu_time"
    MEMORY_USAGE = "memory_usage"
    EXECUTION_TIME = "execution_time"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    ALGORITHM_COMPLEXITY = "algorithm_complexity"


class PerformanceImpact(Enum):
    """Performance impact levels."""
    CRITICAL = "critical"  # >100% regression
    HIGH = "high"         # 50-100% regression
    MEDIUM = "medium"     # 20-50% regression
    LOW = "low"          # 5-20% regression
    MINIMAL = "minimal"   # <5% regression
    IMPROVED = "improved" # Performance improvement


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    type: PerformanceMetricType
    timestamp: datetime
    context: Dict[str, Any] = None
    baseline_value: Optional[float] = None
    regression_percentage: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    test_name: str
    file_path: str
    function_name: Optional[str]
    metrics: List[PerformanceMetric]
    overall_impact: PerformanceImpact
    recommendations: List[str]
    timestamp: datetime
    environment: Dict[str, Any] = None


@dataclass
class PerformanceProfile:
    """Performance profile for tracking over time."""
    function_name: str
    file_path: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    trend_data: List[Dict[str, Any]]
    last_updated: datetime


class PerformanceBenchmarking:
    """Performance benchmarking and analysis system."""

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.data_dir = DATA_DIR / "performance" / repo_name.replace("/", "_")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load historical data
        self.profiles = self._load_profiles()
        self.benchmarks = self._load_benchmarks()

        # Benchmark configuration
        self.config = {
            "timeout_seconds": 30,
            "max_iterations": 100,
            "warmup_iterations": 5,
            "confidence_level": 0.95,
            "significance_threshold": 0.05
        }

    def _load_profiles(self) -> Dict[str, PerformanceProfile]:
        """Load performance profiles from storage."""
        profiles_file = self.data_dir / "profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r') as f:
                data = json.load(f)
            return {
                key: PerformanceProfile(**profile_data)
                for key, profile_data in data.items()
            }
        return {}

    def _load_benchmarks(self) -> List[BenchmarkResult]:
        """Load benchmark results from storage."""
        benchmarks_file = self.data_dir / "benchmarks.json"
        if benchmarks_file.exists():
            with open(benchmarks_file, 'r') as f:
                data = json.load(f)
            return [BenchmarkResult(**result) for result in data]
        return []

    def _save_profiles(self):
        """Save performance profiles to storage."""
        profiles_file = self.data_dir / "profiles.json"
        data = {
            key: asdict(profile) for key, profile in self.profiles.items()
        }
        with open(profiles_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_benchmarks(self):
        """Save benchmark results to storage."""
        benchmarks_file = self.data_dir / "benchmarks.json"
        data = [asdict(benchmark) for benchmark in self.benchmarks]
        with open(benchmarks_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    async def analyze_code_performance(self, file_path: str, content: str,
                                     pr_change_type: str = "modified") -> BenchmarkResult:
        """Analyze performance of code changes."""
        language = detect_language_from_filename(file_path)
        function_name = self._extract_function_name(content, language)

        # Performance analysis based on code patterns
        issues = []
        metrics = []
        recommendations = []

        # Analyze algorithmic complexity
        complexity_metrics = self._analyze_algorithmic_complexity(content, language)
        metrics.extend(complexity_metrics)

        # Analyze resource usage patterns
        resource_metrics = self._analyze_resource_usage(content, language)
        metrics.extend(resource_metrics)

        # Analyze potential performance anti-patterns
        anti_patterns = self._detect_performance_anti_patterns(content, language)
        recommendations.extend(anti_patterns)

        # Generate recommendations based on analysis
        code_recommendations = self._generate_performance_recommendations(
            content, language, pr_change_type
        )
        recommendations.extend(code_recommendations)

        # Calculate overall impact
        impact = self._calculate_performance_impact(metrics)

        # Create benchmark result
        result = BenchmarkResult(
            test_name=f"performance_analysis_{function_name or 'general'}",
            file_path=file_path,
            function_name=function_name,
            metrics=metrics,
            overall_impact=impact,
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
            environment={
                "language": language,
                "change_type": pr_change_type,
                "file_size": len(content)
            }
        )

        # Save result
        self.benchmarks.append(result)
        self._save_benchmarks()

        return result

    def _extract_function_name(self, content: str, language: str) -> Optional[str]:
        """Extract the main function name from code."""
        patterns = {
            "Python": r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            "JavaScript": r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
            "TypeScript": r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
            "Java": r'(public|private|protected)?\s*(static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            "C++": r'([a-zA-Z_][a-zA-Z0-9_]*)\s*::\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(|([a-zA-Z_][a-zA-Z0-9_]*)\s+\w+\s*\([^)]*\)\s*\{',
            "Go": r'func\s+\([^)]*\)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        }

        pattern = patterns.get(language, patterns["Python"])
        matches = re.findall(pattern, content)
        if matches:
            # Handle different capture groups
            for match in matches:
                if isinstance(match, tuple):
                    # Take the last non-empty group
                    for group in reversed(match):
                        if group:
                            return group
                else:
                    return match

        return None

    def _analyze_algorithmic_complexity(self, content: str, language: str) -> List[PerformanceMetric]:
        """Analyze algorithmic complexity patterns."""
        metrics = []

        # Look for nested loops (potential O(n²) or worse)
        nested_loops = self._detect_nested_loops(content)
        if nested_loops:
            metrics.append(PerformanceMetric(
                name="algorithmic_complexity",
                value=nested_loops,
                unit="nesting_level",
                type=PerformanceMetricType.ALGORITHM_COMPLEXITY,
                timestamp=datetime.utcnow(),
                context={"pattern": "nested_loops"}
            ))

        # Look for inefficient data structures
        inefficient_structures = self._detect_inefficient_structures(content, language)
        if inefficient_structures:
            metrics.append(PerformanceMetric(
                name="inefficient_structures",
                value=len(inefficient_structures),
                unit="count",
                type=PerformanceMetricType.ALGORITHM_COMPLEXITY,
                timestamp=datetime.utcnow(),
                context={"structures": inefficient_structures}
            ))

        # Look for recursive functions
        recursive_functions = self._detect_recursive_functions(content, language)
        if recursive_functions:
            metrics.append(PerformanceMetric(
                name="recursive_functions",
                value=len(recursive_functions),
                unit="count",
                type=PerformanceMetricType.ALGORITHM_COMPLEXITY,
                timestamp=datetime.utcnow(),
                context={"functions": recursive_functions}
            ))

        return metrics

    def _detect_nested_loops(self, content: str) -> int:
        """Detect nested loop patterns."""
        lines = content.split('\n')
        max_nesting = 0
        current_nesting = 0

        loop_patterns = [
            r'\bfor\s+',
            r'\bwhile\s+',
            r'\bdo\s+.*\bwhile',
            r'\.forEach\s*\(',
            r'\.map\s*\(',
            r'\.filter\s*\(',
            r'\.reduce\s*\(',
        ]

        for line in lines:
            line_stripped = line.strip()

            # Check for loop start
            if any(re.search(pattern, line_stripped) for pattern in loop_patterns):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)

            # Check for loop end (simplified)
            if line_stripped.endswith('}') and current_nesting > 0:
                current_nesting -= 1

        return max_nesting

    def _detect_inefficient_structures(self, content: str, language: str) -> List[str]:
        """Detect potentially inefficient data structures."""
        inefficient = []

        # Language-specific patterns
        if language == "Python":
            # O(n²) operations on lists
            if re.search(r'for\s+\w+\s+in\s+\w+:\s*.*?in\s+\w+', content, re.DOTALL):
                inefficient.append("list_in_list")

            # String concatenation in loops
            if re.search(r'for\s+\w+\s+in.*?\w+\s*\+=.*?"[\']', content, re.DOTALL):
                inefficient.append("string_concatenation_loop")

            # Inefficient membership testing
            if re.search(r'if\s+\w+\s+in\s+range\(', content):
                inefficient.append("range_membership_test")

        elif language in ["JavaScript", "TypeScript"]:
            # Array.includes in large arrays
            if re.search(r'\.includes\s*\(', content):
                inefficient.append("array_includes_large")

            # Array.find on large arrays
            if re.search(r'\.find\s*\(', content):
                inefficient.append("array_find_large")

            # Spread operator with large arrays
            if re.search(r'\[\s*\.\.\.\w+', content):
                inefficient.append("spread_large_array")

        elif language == "Java":
            # ArrayList operations that could be optimized
            if re.search(r'ArrayList<.*>.*\.contains\(', content):
                inefficient.append("arraylist_contains")

            # String concatenation in loops
            if re.search(r'for.*?\w+\s*\+=.*?"\+.*?"', content, re.DOTALL):
                inefficient.append("string_concatenation_loop")

        return inefficient

    def _detect_recursive_functions(self, content: str, language: str) -> List[str]:
        """Detect recursive function calls."""
        recursive_functions = []

        # Extract function names
        function_pattern = r'(?:def|function|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        functions = re.findall(function_pattern, content)

        # Check for self-references
        for func_name in functions:
            if re.search(fr'\b{func_name}\s*\(', content):
                # Check if the call is within the same function (simplified)
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if f'def {func_name}' in line or f'function {func_name}' in line:
                        # Look ahead for self-call within reasonable range
                        for j in range(i+1, min(i+50, len(lines))):
                            if re.search(fr'\b{func_name}\s*\(', lines[j]):
                                recursive_functions.append(func_name)
                                break

        return recursive_functions

    def _analyze_resource_usage(self, content: str, language: str) -> List[PerformanceMetric]:
        """Analyze resource usage patterns."""
        metrics = []

        # Memory usage patterns
        memory_patterns = self._detect_memory_patterns(content, language)
        metrics.append(PerformanceMetric(
            name="memory_patterns",
            value=len(memory_patterns),
            unit="count",
            type=PerformanceMetricType.MEMORY_USAGE,
            timestamp=datetime.utcnow(),
            context={"patterns": memory_patterns}
        ))

        # I/O patterns
        io_patterns = self._detect_io_patterns(content, language)
        metrics.append(PerformanceMetric(
            name="io_patterns",
            value=len(io_patterns),
            unit="count",
            type=PerformanceMetricType.RESPONSE_TIME,
            timestamp=datetime.utcnow(),
            context={"patterns": io_patterns}
        ))

        # CPU-intensive patterns
        cpu_patterns = self._detect_cpu_patterns(content, language)
        metrics.append(PerformanceMetric(
            name="cpu_patterns",
            value=len(cpu_patterns),
            unit="count",
            type=PerformanceMetricType.CPU_TIME,
            timestamp=datetime.utcnow(),
            context={"patterns": cpu_patterns}
        ))

        return metrics

    def _detect_memory_patterns(self, content: str, language: str) -> List[str]:
        """Detect memory usage patterns."""
        patterns = []

        # Large object creation
        if re.search(r'new\s+Array\s*\(\s*\d{4,}', content):
            patterns.append("large_array_creation")

        # Potential memory leaks
        if language == "Python":
            if re.search(r'while\s+True:.*?append\(', content, re.DOTALL):
                patterns.append("potential_memory_leak_loop")

        # Caching patterns
        if re.search(r'cache|Cache|CACHE', content):
            patterns.append("caching_pattern")

        return patterns

    def _detect_io_patterns(self, content: str, language: str) -> List[str]:
        """Detect I/O patterns that could affect performance."""
        patterns = []

        # File I/O
        if re.search(r'open\s*\(|read\s*\(|write\s*\(', content):
            patterns.append("file_io")

        # Network I/O
        if re.search(r'http|fetch|request|socket', content, re.IGNORECASE):
            patterns.append("network_io")

        # Database I/O
        if re.search(r'db|database|sql|query', content, re.IGNORECASE):
            patterns.append("database_io")

        return patterns

    def _detect_cpu_patterns(self, content: str, language: str) -> List[str]:
        """Detect CPU-intensive patterns."""
        patterns = []

        # Heavy computations
        if re.search(r'crypto|encrypt|decrypt|hash', content, re.IGNORECASE):
            patterns.append("cryptographic_operations")

        # Image/video processing
        if re.search(r'image|video|render|process', content, re.IGNORECASE):
            patterns.append("media_processing")

        # Mathematical operations
        if re.search(r'math|calculate|compute|algorithm', content, re.IGNORECASE):
            patterns.append("mathematical_operations")

        return patterns

    def _detect_performance_anti_patterns(self, content: str, language: str) -> List[str]:
        """Detect common performance anti-patterns."""
        anti_patterns = []

        # Global variables in loops
        if re.search(r'for.*?global\s+\w+', content, re.DOTALL):
            anti_patterns.append("global_in_loop")

        # Exception handling in loops
        if re.search(r'try:.*?for.*?except', content, re.DOTALL):
            anti_patterns.append("exception_in_loop")

        # Regex in loops
        if re.search(r'for.*?re\.', content, re.DOTALL):
            anti_patterns.append("regex_in_loop")

        # Database queries in loops
        if re.search(r'for.*?execute\(|for.*?query\(', content, re.DOTALL):
            anti_patterns.append("database_in_loop")

        # Synchronous operations in async contexts
        if language in ["JavaScript", "TypeScript"]:
            if re.search(r'async.*?\.readFileSync\(|await.*?\.readFileSync\(', content):
                anti_patterns.append("sync_in_async")

        # Blocking I/O in event loops
        if language == "Python":
            if re.search(r'async\s+def.*?time\.sleep\(', content):
                anti_patterns.append("blocking_in_async")

        return anti_patterns

    def _generate_performance_recommendations(self, content: str, language: str,
                                             change_type: str) -> List[str]:
        """Generate performance recommendations based on code analysis."""
        recommendations = []

        # General performance recommendations
        if "for" in content and "len(" in content:
            recommendations.append("Consider using list comprehensions or generator expressions for better performance")

        if language == "Python":
            if "dict.keys()" in content and "in" in content:
                recommendations.append("Use direct key lookup instead of 'key in dict.keys()' for better performance")

            if "range(len(" in content:
                recommendations.append("Use enumerate() instead of range(len()) for cleaner and more efficient code")

        elif language in ["JavaScript", "TypeScript"]:
            if "for (let i = 0; i < array.length; i++)" in content:
                recommendations.append("Use for...of or array methods instead of traditional for loops when possible")

            if "===" not in content:
                recommendations.append("Use strict equality (===) instead of loose equality (==) for better performance and type safety")

        elif language == "Java":
            if "String" in content and "+" in content:
                recommendations.append("Use StringBuilder for string concatenation in loops to avoid creating multiple String objects")

        # Algorithm-specific recommendations
        nested_loops = self._detect_nested_loops(content)
        if nested_loops >= 3:
            recommendations.append(f"Consider optimizing the {nested_loops}-level nested loops - O(n^{nested_loops}) complexity detected")

        recursive_functions = self._detect_recursive_functions(content, language)
        if recursive_functions:
            recommendations.append("Consider memoization or iterative approaches for recursive functions to improve performance")

        # Memory usage recommendations
        memory_patterns = self._detect_memory_patterns(content, language)
        if memory_patterns:
            recommendations.append("Review memory usage patterns - consider using generators or streams for large datasets")

        # Change-type specific recommendations
        if change_type == "new":
            recommendations.append("Since this is new code, consider adding performance benchmarks to track future regressions")
        elif change_type == "modified":
            recommendations.append("Compare performance with previous version to detect potential regressions")

        return recommendations

    def _calculate_performance_impact(self, metrics: List[PerformanceMetric]) -> PerformanceImpact:
        """Calculate overall performance impact."""
        if not metrics:
            return PerformanceImpact.MINIMAL

        # Weight different metric types
        impact_scores = {
            PerformanceMetricType.ALGORITHM_COMPLEXITY: 3,
            PerformanceMetricType.CPU_TIME: 2,
            PerformanceMetricType.MEMORY_USAGE: 2,
            PerformanceMetricType.RESPONSE_TIME: 2,
            PerformanceMetricType.EXECUTION_TIME: 1,
            PerformanceMetricType.THROUGHPUT: 1,
            PerformanceMetricType.LATENCY: 2,
            PerformanceMetricType.ERROR_RATE: 3,
        }

        total_score = 0
        for metric in metrics:
            weight = impact_scores.get(metric.type, 1)
            value_score = min(metric.value / 10, 5)  # Normalize and cap at 5
            total_score += weight * value_score

        # Convert score to impact level
        if total_score >= 15:
            return PerformanceImpact.CRITICAL
        elif total_score >= 10:
            return PerformanceImpact.HIGH
        elif total_score >= 5:
            return PerformanceImpact.MEDIUM
        elif total_score >= 2:
            return PerformanceImpact.LOW
        else:
            return PerformanceImpact.MINIMAL

    async def benchmark_function(self, file_path: str, function_name: str,
                                 test_data: Any = None) -> BenchmarkResult:
        """Benchmark a specific function with actual execution."""
        try:
            # Import the module and get the function
            module_path = Path(file_path)
            if not module_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Create a temporary test script
            test_script = self._create_benchmark_script(file_path, function_name, test_data)

            # Run benchmark
            metrics = await self._run_benchmark(test_script)

            # Generate recommendations
            recommendations = self._generate_benchmark_recommendations(metrics)

            # Calculate impact
            impact = self._calculate_performance_impact(metrics)

            result = BenchmarkResult(
                test_name=f"benchmark_{function_name}",
                file_path=file_path,
                function_name=function_name,
                metrics=metrics,
                overall_impact=impact,
                recommendations=recommendations,
                timestamp=datetime.utcnow(),
                environment={"test_data_size": str(test_data) if test_data else "default"}
            )

            # Update profile
            profile_key = f"{file_path}:{function_name}"
            if profile_key not in self.profiles:
                self.profiles[profile_key] = PerformanceProfile(
                    function_name=function_name,
                    file_path=file_path,
                    baseline_metrics={m.name: m.value for m in metrics},
                    current_metrics={m.name: m.value for m in metrics},
                    trend_data=[{
                        "timestamp": datetime.utcnow().isoformat(),
                        "metrics": {m.name: m.value for m in metrics}
                    }],
                    last_updated=datetime.utcnow()
                )
            else:
                profile = self.profiles[profile_key]
                profile.current_metrics = {m.name: m.value for m in metrics}
                profile.trend_data.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {m.name: m.value for m in metrics}
                })
                profile.last_updated = datetime.utcnow()

            self._save_profiles()
            self._save_benchmarks()

            return result

        except Exception as e:
            # Return error result
            error_metric = PerformanceMetric(
                name="benchmark_error",
                value=1,
                unit="error",
                type=PerformanceMetricType.ERROR_RATE,
                timestamp=datetime.utcnow(),
                context={"error": str(e)}
            )

            return BenchmarkResult(
                test_name=f"benchmark_{function_name}",
                file_path=file_path,
                function_name=function_name,
                metrics=[error_metric],
                overall_impact=PerformanceImpact.CRITICAL,
                recommendations=[f"Benchmark failed: {str(e)}"],
                timestamp=datetime.utcnow()
            )

    def _create_benchmark_script(self, file_path: str, function_name: str,
                               test_data: Any) -> str:
        """Create a benchmark script for the function."""
        language = detect_language_from_filename(file_path)

        script_template = f"""
import time
import sys
import traceback
import gc
from statistics import mean, stdev

# Import the module
sys.path.insert(0, '{Path(file_path).parent}')

def run_benchmark():
    try:
        # Import the function
        from {Path(file_path).stem} import {function_name}

        # Prepare test data
        test_data = {repr(test_data) if test_data else 'None'}

        # Warmup
        for _ in range(5):
            try:
                {function_name}(test_data) if test_data is not None else {function_name}()
            except:
                pass

        gc.collect()

        # Benchmark execution
        times = []
        for _ in range(20):
            start = time.perf_counter()
            try:
                {function_name}(test_data) if test_data is not None else {function_name}()
            except Exception as e:
                print(f"ERROR: {{e}}", file=sys.stderr)
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        avg_time = mean(times) if times else 0
        std_dev = stdev(times) if len(times) > 1 else 0

        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # CPU usage
        cpu_percent = process.cpu_percent()

        results = {{
            "execution_time_avg": avg_time,
            "execution_time_std": std_dev,
            "memory_usage_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "iterations": len(times)
        }}

        print("BENCHMARK_RESULTS:", json.dumps(results))

    except Exception as e:
        print("BENCHMARK_ERROR:", str(e), file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    run_benchmark()
"""

        return script_template

    async def _run_benchmark(self, script: str) -> List[PerformanceMetric]:
        """Run the benchmark script and collect metrics."""
        try:
            # Write script to temporary file
            script_file = self.data_dir / "temp_benchmark.py"
            with open(script_file, 'w') as f:
                f.write(script)

            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_file)],
                capture_output=True,
                text=True,
                timeout=self.config["timeout_seconds"],
                cwd=self.data_dir
            )

            # Parse results
            metrics = []
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULTS:'):
                    try:
                        data = json.loads(line.split(':', 1)[1])

                        metrics.extend([
                            PerformanceMetric(
                                name="execution_time_avg",
                                value=data["execution_time_avg"],
                                unit="seconds",
                                type=PerformanceMetricType.EXECUTION_TIME,
                                timestamp=datetime.utcnow()
                            ),
                            PerformanceMetric(
                                name="execution_time_std",
                                value=data["execution_time_std"],
                                unit="seconds",
                                type=PerformanceMetricType.EXECUTION_TIME,
                                timestamp=datetime.utcnow()
                            ),
                            PerformanceMetric(
                                name="memory_usage_mb",
                                value=data["memory_usage_mb"],
                                unit="mb",
                                type=PerformanceMetricType.MEMORY_USAGE,
                                timestamp=datetime.utcnow()
                            ),
                            PerformanceMetric(
                                name="cpu_percent",
                                value=data["cpu_percent"],
                                unit="percent",
                                type=PerformanceMetricType.CPU_TIME,
                                timestamp=datetime.utcnow()
                            )
                        ])
                    except json.JSONDecodeError:
                        pass

            # Clean up
            script_file.unlink(missing_ok=True)

            return metrics

        except subprocess.TimeoutExpired:
            return [PerformanceMetric(
                name="timeout",
                value=self.config["timeout_seconds"],
                unit="seconds",
                type=PerformanceMetricType.EXECUTION_TIME,
                timestamp=datetime.utcnow(),
                context={"error": "timeout"}
            )]

        except Exception as e:
            return [PerformanceMetric(
                name="benchmark_error",
                value=1,
                unit="error",
                type=PerformanceMetricType.ERROR_RATE,
                timestamp=datetime.utcnow(),
                context={"error": str(e)}
            )]

    def _generate_benchmark_recommendations(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []

        for metric in metrics:
            if metric.name == "execution_time_avg" and metric.value > 1.0:
                recommendations.append(f"Function takes {metric.value:.2f} seconds - consider optimization")

            if metric.name == "memory_usage_mb" and metric.value > 100:
                recommendations.append(f"High memory usage ({metric.value:.1f} MB) - check for memory leaks")

            if metric.name == "cpu_percent" and metric.value > 50:
                recommendations.append(f"High CPU usage ({metric.value:.1f}%) - check for inefficient algorithms")

            if metric.name == "execution_time_std" and metric.value > 0.1:
                recommendations.append("High variance in execution times - check for inconsistent performance")

        return recommendations

    def get_performance_trends(self, file_path: str, function_name: str,
                             days: int = 30) -> Dict[str, Any]:
        """Get performance trends for a function over time."""
        profile_key = f"{file_path}:{function_name}"

        if profile_key not in self.profiles:
            return {"error": "No performance data available for this function"}

        profile = self.profiles[profile_key]
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Filter trend data
        recent_trends = [
            point for point in profile.trend_data
            if datetime.fromisoformat(point["timestamp"]) >= cutoff_date
        ]

        if not recent_trends:
            return {"error": f"No performance data in the last {days} days"}

        # Calculate trends
        metrics_trends = {}
        for metric_name in profile.current_metrics.keys():
            values = [point["metrics"].get(metric_name, 0) for point in recent_trends if metric_name in point["metrics"]]

            if len(values) > 1:
                trend = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
                metrics_trends[metric_name] = {
                    "trend_percentage": trend,
                    "current_value": values[-1],
                    "baseline_value": values[0],
                    "values": values,
                    "improvement": trend < -5,  # Improved if more than 5% faster
                    "regression": trend > 5  # Regression if more than 5% slower
                }

        return {
            "function_name": function_name,
            "file_path": file_path,
            "period_days": days,
            "data_points": len(recent_trends),
            "metrics_trends": metrics_trends,
            "overall_trend": self._calculate_overall_trend(metrics_trends),
            "last_updated": profile.last_updated.isoformat()
        }

    def _calculate_overall_trend(self, metrics_trends: Dict[str, Any]) -> str:
        """Calculate overall trend from individual metrics."""
        if not metrics_trends:
            return "no_data"

        regressions = sum(1 for trend in metrics_trends.values() if trend.get("regression", False))
        improvements = sum(1 for trend in metrics_trends.values() if trend.get("improvement", False))

        if regressions > improvements:
            return "regression"
        elif improvements > regressions:
            return "improvement"
        else:
            return "stable"

    def compare_performance(self, file_path: str, function_name: str,
                          compare_to: str = "baseline") -> Dict[str, Any]:
        """Compare current performance with baseline or previous version."""
        profile_key = f"{file_path}:{function_name}"

        if profile_key not in self.profiles:
            return {"error": "No performance data available"}

        profile = self.profiles[profile_key]
        current_metrics = profile.current_metrics

        if compare_to == "baseline":
            compare_metrics = profile.baseline_metrics
            comparison_type = "baseline"
        else:
            # Compare to previous version
            if len(profile.trend_data) < 2:
                return {"error": "Not enough data for comparison"}

            previous_metrics = profile.trend_data[-2]["metrics"]
            compare_metrics = previous_metrics
            comparison_type = "previous"

        comparison = {}
        for metric_name in current_metrics:
            if metric_name in compare_metrics:
                current_val = current_metrics[metric_name]
                compare_val = compare_metrics[metric_name]

                if compare_val != 0:
                    change_percent = ((current_val - compare_val) / compare_val) * 100
                else:
                    change_percent = 0

                comparison[metric_name] = {
                    "current": current_val,
                    "comparison": compare_val,
                    "change_percent": change_percent,
                    "improvement": change_percent < -5,
                    "regression": change_percent > 5,
                    "significant": abs(change_percent) > 10
                }

        return {
            "function_name": function_name,
            "file_path": file_path,
            "comparison_type": comparison_type,
            "metrics": comparison,
            "overall_assessment": self._assess_comparison(comparison)
        }

    def _assess_comparison(self, comparison: Dict[str, Any]) -> str:
        """Assess overall comparison result."""
        if not comparison:
            return "no_data"

        regressions = sum(1 for metric in comparison.values() if metric.get("regression", False))
        improvements = sum(1 for metric in comparison.values() if metric.get("improvement", False))
        significant_changes = sum(1 for metric in comparison.values() if metric.get("significant", False))

        if regressions > 0:
            return "performance_regression"
        elif improvements > 0 and significant_changes > 0:
            return "performance_improvement"
        elif significant_changes > 0:
            return "minor_changes"
        else:
            return "stable"

    def generate_performance_report(self, file_path: str, function_name: Optional[str] = None,
                                   include_trends: bool = True, include_comparisons: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "repo_name": self.repo_name
        }

        if function_name:
            # Function-specific report
            benchmark_results = [b for b in self.benchmarks if b.function_name == function_name and b.file_path == file_path]

            if benchmark_results:
                latest_benchmark = benchmark_results[-1]
                report["benchmark"] = asdict(latest_benchmark)

            if include_trends:
                report["trends"] = self.get_performance_trends(file_path, function_name)

            if include_comparisons:
                report["comparison"] = self.compare_performance(file_path, function_name)

            report["profile"] = asdict(self.profiles.get(f"{file_path}:{function_name}", {})) if f"{file_path}:{function_name}" in self.profiles else None

        else:
            # File-level report
            file_benchmarks = [b for b in self.benchmarks if b.file_path == file_path]
            report["benchmarks"] = [asdict(b) for b in file_benchmarks[-10:]]  # Last 10

            # Aggregate metrics
            all_metrics = []
            for benchmark in file_benchmarks:
                all_metrics.extend(benchmark.metrics)

            report["summary"] = {
                "total_benchmarks": len(file_benchmarks),
                "metrics_summary": self._summarize_metrics(all_metrics),
                "performance_impacts": {impact.value: sum(1 for b in file_benchmarks if b.overall_impact == impact)
                                    for impact in PerformanceImpact}
            }

        return report

    def _summarize_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Summarize metrics across all benchmarks."""
        if not metrics:
            return {}

        summary = {}
        for metric_type in PerformanceMetricType:
            type_metrics = [m for m in metrics if m.type == metric_type]
            if type_metrics:
                values = [m.value for m in type_metrics]
                summary[metric_type.value] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": type_metrics[-1].value
                }

        return summary


# Global performance benchmarking instance
_performance_benchmarking_instances = {}

def get_performance_benchmarking(repo_name: str) -> PerformanceBenchmarking:
    """Get or create performance benchmarking instance for a repository."""
    if repo_name not in _performance_benchmarking_instances:
        _performance_benchmarking_instances[repo_name] = PerformanceBenchmarking(repo_name)
    return _performance_benchmarking_instances[repo_name]