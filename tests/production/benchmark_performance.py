"""
Performance benchmarking for SeaSeeAI production system
"""

import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json

class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def generate_test_data(self, sequence_length=20, num_sequences=100):
        """Generate test data for benchmarking"""
        test_sequences = []
        
        for i in range(num_sequences):
            observations = []
            base_lat = 37.7749 + (i * 0.1)
            base_lon = -122.4194 + (i * 0.1)
            
            for j in range(sequence_length):
                observations.append({
                    'mmsi': 100000000 + i,
                    'timestamp': (datetime.now() - timedelta(hours=sequence_length-j)).isoformat(),
                    'latitude': base_lat + j * 0.001,
                    'longitude': base_lon + j * 0.001,
                    'sog': 10.0 + j * 0.1,
                    'cog': 45.0 + j * 1.0
                })
            
            test_sequences.append(observations)
        
        return test_sequences
    def benchmark_single_predictions(self, num_requests=100):
        """Benchmark single prediction performance"""
        print(f"🚀 Benchmarking {num_requests} single predictions...")
        
        test_sequences = self.generate_test_data(num_sequences=num_requests)
        response_times = []
        success_count = 0
        
        for i, sequence in enumerate(test_sequences):
            prediction_request = {
                'observations': sequence,
                'prediction_horizon': 5
            }
            
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=prediction_request,
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                    success_count += 1
                else:
                    print(f"Request {i} failed with status {response.status_code}")
            
            except Exception as e:
                print(f"Request {i} failed with exception: {e}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_requests} requests")
        
        # Calculate statistics
        if response_times:
            stats = {
                'test_type': 'single_predictions',
                'total_requests': num_requests,
                'successful_requests': success_count,
                'success_rate': success_count / num_requests,
                'mean_response_time': np.mean(response_times),
                'median_response_time': np.median(response_times),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99),
                'min_response_time': np.min(response_times),
                'max_response_time': np.max(response_times),
                'throughput': success_count / sum(response_times) if response_times else 0
            }
            
            self.results.append(stats)
            self._print_stats(stats)
            
            return stats
        else:
            print("❌ No successful requests")
            return None
    def benchmark_batch_predictions(self, batch_sizes=[5, 10, 20]):
        """Benchmark batch prediction performance"""
        print("🚀 Benchmarking batch predictions...")
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Generate batch request
            batch_requests = []
            test_sequences = self.generate_test_data(num_sequences=batch_size)
            
            for sequence in test_sequences:
                batch_requests.append({
                    'observations': sequence,
                    'prediction_horizon': 5
                })
            
            batch_request = {
                'requests': batch_requests
            }
            
            response_times = []
            success_count = 0
            
            for i in range(10):  # Multiple trials
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/predict/batch",
                        json=batch_request,
                        timeout=60
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        response_times.append(end_time - start_time)
                        success_count += 1
                        data = response.json()
                        
                        # Verify all predictions were processed
                        assert len(data['results']) == batch_size
                    
                    else:
                        print(f"Batch request failed with status {response.status_code}")
                
                except Exception as e:
                    print(f"Batch request failed with exception: {e}")
            
            if response_times:
                stats = {
                    'test_type': f'batch_predictions_{batch_size}',
                    'batch_size': batch_size,
                    'total_requests': 10,
                    'successful_requests': success_count,
                    'success_rate': success_count / 10,
                    'mean_response_time': np.mean(response_times),
                    'median_response_time': np.median(response_times),
                    'throughput': (batch_size * success_count) / sum(response_times),
                    'efficiency': (batch_size * success_count) / sum(response_times) / batch_size
                }
                
                self.results.append(stats)
                self._print_stats(stats)
    def benchmark_concurrent_requests(self, concurrent_users=[1, 5, 10, 20]):
        """Benchmark concurrent request handling"""
        print("🚀 Benchmarking concurrent requests...")
        import concurrent.futures
        
        test_sequences = self.generate_test_data(num_sequences=max(concurrent_users))
        
        for num_users in concurrent_users:
            print(f"Testing with {num_users} concurrent users...")
            
            def make_request(user_id):
                sequence = test_sequences[user_id % len(test_sequences)]
                prediction_request = {
                    'observations': sequence,
                    'prediction_horizon': 5
                }
                
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json=prediction_request,
                        timeout=30
                    )
                    end_time = time.time()
                    
                    return {
                        'success': response.status_code == 200,
                        'response_time': end_time - start_time
                    }
                except:
                    return {'success': False, 'response_time': None}
            
            response_times = []
            success_count = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_users)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            for result in results:
                if result['success'] and result['response_time'] is not None:
                    response_times.append(result['response_time'])
                    success_count += 1
            
            if response_times:
                stats = {
                    'test_type': f'concurrent_requests_{num_users}',
                    'concurrent_users': num_users,
                    'total_requests': num_users,
                    'successful_requests': success_count,
                    'success_rate': success_count / num_users,
                    'mean_response_time': np.mean(response_times),
                    'median_response_time': np.median(response_times),
                    'p95_response_time': np.percentile(response_times, 95),
                    'throughput': success_count / max(response_times)
                }
                
                self.results.append(stats)
                self._print_stats(stats)
    def _print_stats(self, stats):
        """Print performance statistics"""
        print(f"\n📊 {stats['test_type'].replace('_', ' ').title()}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Mean Response Time: {stats['mean_response_time']:.3f}s")
        
        if 'median_response_time' in stats:
            print(f"   Median Response Time: {stats['median_response_time']:.3f}s")
        
        if 'p95_response_time' in stats:
            print(f"   95th Percentile: {stats['p95_response_time']:.3f}s")
        
        if 'throughput' in stats:
            print(f"   Throughput: {stats['throughput']:.2f} requests/second")
        
        if 'efficiency' in stats:
            print(f"   Batch Efficiency: {stats['efficiency']:.2f} requests/second/core")
        
        print()
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            print("No benchmark results available")
            return
        
        df = pd.DataFrame(self.results)
        
        print("=" * 60)
        print("🎯 SeaSeeAI Performance Benchmark Report")
        print("=" * 60)
        
        # Display results table
        print("\n📈 Results Summary:")
        print(df.to_string(index=False))
        
        # Create visualizations
        plots_dir = "tests/production/plots"
        import os
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: Response times for different tests
        plt.figure(figsize=(10, 6))
        response_times_data = []
        for result in self.results:
            if 'mean_response_time' in result:
                response_times_data.append({
                    'Test': result['test_type'],
                    'Mean Response Time (s)': result['mean_response_time']
                })
        
        if response_times_data:
            response_df = pd.DataFrame(response_times_data)
            plt.bar(response_df['Test'], response_df['Mean Response Time (s)'])
            plt.title('Mean Response Time by Test Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/response_times.png")
            plt.close()
        
        # Plot 2: Throughput for different tests
        plt.figure(figsize=(10, 6))
        throughput_data = []
        for result in self.results:
            if 'throughput' in result:
                throughput_data.append({
                    'Test': result['test_type'],
                    'Throughput (req/s)': result['throughput']
                })
        
        if throughput_data:
            throughput_df = pd.DataFrame(throughput_data)
            plt.bar(throughput_df['Test'], throughput_df['Throughput (req/s)'])
            plt.title('Throughput by Test Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/throughput.png")
            plt.close()
        
        # Save detailed results to CSV
        df.to_csv("tests/production/benchmark_results.csv", index=False)
        print(f"\n💾 Detailed results saved to: tests/production/benchmark_results.csv")
        print(f"📊 Plots saved to: {plots_dir}/")

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_single_predictions(num_requests=50)  # Reduced for quick test
    benchmark.benchmark_batch_predictions(batch_sizes=[5, 10])
    benchmark.benchmark_concurrent_requests(concurrent_users=[1, 5, 10])
    
    # Generate report
    benchmark.generate_report()
