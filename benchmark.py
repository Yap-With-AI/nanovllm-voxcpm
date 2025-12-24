"""
TTS Benchmark Script - Calls FastAPI server
Measures: TTFB, RTF, XRT, P50/P90/P95 latencies, throughput
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List
import argparse
import httpx


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: int
    text: str
    
    # Timing
    start_time: float = 0.0
    first_chunk_time: float = 0.0
    end_time: float = 0.0
    
    # Audio
    audio_samples: int = 0
    chunk_count: int = 0
    
    @property
    def ttfb(self) -> float:
        """Time to first byte (seconds)"""
        return self.first_chunk_time - self.start_time
    
    @property
    def total_time(self) -> float:
        """Total generation time (seconds)"""
        return self.end_time - self.start_time
    
    @property
    def audio_duration(self) -> float:
        """Audio duration in seconds (16kHz)"""
        return self.audio_samples / 16000
    
    @property
    def rtf(self) -> float:
        """Real-Time Factor (lower is better, <1 means faster than real-time)"""
        if self.audio_duration == 0:
            return float('inf')
        return self.total_time / self.audio_duration
    
    @property
    def xrt(self) -> float:
        """Times real-time (higher is better, >1 means faster than real-time)"""
        if self.total_time == 0:
            return float('inf')
        return self.audio_duration / self.total_time


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    metrics: List[RequestMetrics] = field(default_factory=list)
    total_start_time: float = 0.0
    total_end_time: float = 0.0
    concurrency: int = 0
    
    def add(self, metric: RequestMetrics):
        self.metrics.append(metric)
    
    def percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))
    
    def summary(self) -> dict:
        ttfbs = [m.ttfb for m in self.metrics]
        total_times = [m.total_time for m in self.metrics]
        rtfs = [m.rtf for m in self.metrics]
        xrts = [m.xrt for m in self.metrics]
        audio_durations = [m.audio_duration for m in self.metrics]
        
        total_audio = sum(audio_durations)
        wall_time = self.total_end_time - self.total_start_time
        
        return {
            "requests": len(self.metrics),
            "concurrency": self.concurrency,
            "wall_time_sec": wall_time,
            
            # TTFB stats
            "ttfb_p50": self.percentile(ttfbs, 50),
            "ttfb_p90": self.percentile(ttfbs, 90),
            "ttfb_p95": self.percentile(ttfbs, 95),
            "ttfb_mean": np.mean(ttfbs) if ttfbs else 0,
            "ttfb_min": min(ttfbs) if ttfbs else 0,
            "ttfb_max": max(ttfbs) if ttfbs else 0,
            
            # Total latency stats
            "latency_p50": self.percentile(total_times, 50),
            "latency_p90": self.percentile(total_times, 90),
            "latency_p95": self.percentile(total_times, 95),
            "latency_mean": np.mean(total_times) if total_times else 0,
            
            # RTF stats (lower is better)
            "rtf_p50": self.percentile(rtfs, 50),
            "rtf_p90": self.percentile(rtfs, 90),
            "rtf_p95": self.percentile(rtfs, 95),
            "rtf_mean": np.mean(rtfs) if rtfs else 0,
            
            # XRT stats (higher is better)
            "xrt_p50": self.percentile(xrts, 50),
            "xrt_p90": self.percentile(xrts, 90),
            "xrt_p95": self.percentile(xrts, 95),
            "xrt_mean": np.mean(xrts) if xrts else 0,
            
            # Throughput
            "total_audio_sec": total_audio,
            "throughput_xrt": total_audio / wall_time if wall_time > 0 else 0,
            "requests_per_sec": len(self.metrics) / wall_time if wall_time > 0 else 0,
        }
    
    def print_report(self):
        s = self.summary()
        
        print("\n" + "="*60)
        print("                    TTS BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\n{'CONFIGURATION':-^60}")
        print(f"  Total Requests:      {s['requests']}")
        print(f"  Concurrency:         {s['concurrency']}")
        print(f"  Wall Clock Time:     {s['wall_time_sec']:.2f}s")
        
        print(f"\n{'TIME TO FIRST BYTE (TTFB)':-^60}")
        print(f"  P50:    {s['ttfb_p50']*1000:>8.1f} ms")
        print(f"  P90:    {s['ttfb_p90']*1000:>8.1f} ms")
        print(f"  P95:    {s['ttfb_p95']*1000:>8.1f} ms")
        print(f"  Mean:   {s['ttfb_mean']*1000:>8.1f} ms")
        print(f"  Min:    {s['ttfb_min']*1000:>8.1f} ms")
        print(f"  Max:    {s['ttfb_max']*1000:>8.1f} ms")
        
        print(f"\n{'TOTAL LATENCY (end-to-end)':-^60}")
        print(f"  P50:    {s['latency_p50']:>8.2f} s")
        print(f"  P90:    {s['latency_p90']:>8.2f} s")
        print(f"  P95:    {s['latency_p95']:>8.2f} s")
        print(f"  Mean:   {s['latency_mean']:>8.2f} s")
        
        print(f"\n{'REAL-TIME FACTOR (RTF) - lower is better':-^60}")
        print(f"  P50:    {s['rtf_p50']:>8.3f}x")
        print(f"  P90:    {s['rtf_p90']:>8.3f}x")
        print(f"  P95:    {s['rtf_p95']:>8.3f}x")
        print(f"  Mean:   {s['rtf_mean']:>8.3f}x")
        
        print(f"\n{'TIMES REAL-TIME (XRT) - higher is better':-^60}")
        print(f"  P50:    {s['xrt_p50']:>8.2f}x")
        print(f"  P90:    {s['xrt_p90']:>8.2f}x")
        print(f"  P95:    {s['xrt_p95']:>8.2f}x")
        print(f"  Mean:   {s['xrt_mean']:>8.2f}x")
        
        print(f"\n{'THROUGHPUT':-^60}")
        print(f"  Total Audio Generated:  {s['total_audio_sec']:>8.1f} s")
        print(f"  Aggregate XRT:          {s['throughput_xrt']:>8.2f}x real-time")
        print(f"  Requests/sec:           {s['requests_per_sec']:>8.2f}")
        
        print("\n" + "="*60)
        
        if s['rtf_p95'] < 1.0:
            print("✅ REAL-TIME CAPABLE: P95 RTF < 1.0")
        else:
            print(f"⚠️  NOT REAL-TIME: P95 RTF = {s['rtf_p95']:.2f}x (needs < 1.0)")
        print("="*60 + "\n")


# Sample texts of varying lengths for testing
SAMPLE_TEXTS = [
    # Short (~10-20 tokens)
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of artificial intelligence.",
    
    # Medium (~30-40 tokens)
    "In a world where technology advances at an unprecedented pace, we must adapt and evolve to keep up with the changing times.",
    "The development of neural networks has revolutionized the way we think about machine learning and artificial intelligence.",
    "Climate change poses one of the greatest challenges of our generation, requiring immediate and decisive action from all nations.",
    
    # Longer (~50-60 tokens)
    "Throughout history, humanity has faced countless challenges that seemed insurmountable at the time. Yet through perseverance, innovation, and collective action, we have overcome obstacles that once appeared impossible to surmount.",
    "The intersection of art and technology has given rise to new forms of creative expression that were unimaginable just a few decades ago. From digital paintings to AI-generated music, the boundaries of human creativity continue to expand.",
]


async def run_single_request(
    server_url: str,
    request_id: int,
    text: str,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
) -> RequestMetrics:
    """Run a single TTS request and collect metrics"""
    
    metrics = RequestMetrics(
        request_id=request_id,
        text=text,
    )
    
    first_chunk_received = False
    
    # Create a NEW client for each request (no connection reuse)
    async with httpx.AsyncClient(timeout=300) as client:
        # Start timing RIGHT BEFORE sending the request (not including handshake)
        metrics.start_time = time.perf_counter()
        
        async with client.stream(
            "POST",
            f"{server_url}/generate",
            json={
                "target_text": text,
                "temperature": temperature,
                "cfg_value": cfg_value,
                "max_generate_length": max_generate_length,
            },
        ) as response:
            response.raise_for_status()
            
            async for chunk in response.aiter_bytes():
                if not first_chunk_received:
                    metrics.first_chunk_time = time.perf_counter()
                    first_chunk_received = True
                
                # Each chunk is raw float32 audio
                metrics.audio_samples += len(chunk) // 4
                metrics.chunk_count += 1
    
    metrics.end_time = time.perf_counter()
    
    return metrics


async def run_batch(
    server_url: str,
    texts: List[str],
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    start_id: int = 0,
) -> List[RequestMetrics]:
    """Run a batch of concurrent requests"""
    
    # Each request creates its own client (no connection reuse)
    tasks = [
        run_single_request(
            server_url, start_id + i, text, 
            max_generate_length, temperature, cfg_value
        )
        for i, text in enumerate(texts)
    ]
    
    return await asyncio.gather(*tasks)


async def run_benchmark(
    server_url: str,
    num_requests: int,
    concurrency: int,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    texts: List[str] = None,
) -> BenchmarkResults:
    """Run the full benchmark"""
    
    if texts is None:
        texts = SAMPLE_TEXTS
    
    results = BenchmarkResults(concurrency=concurrency)
    results.total_start_time = time.perf_counter()
    
    # Process requests in batches of `concurrency`
    request_id = 0
    remaining = num_requests
    
    while remaining > 0:
        batch_size = min(concurrency, remaining)
        batch_texts = [texts[i % len(texts)] for i in range(request_id, request_id + batch_size)]
        
        print(f"Running batch: {request_id + 1}-{request_id + batch_size} of {num_requests}...")
        
        batch_metrics = await run_batch(
            server_url, batch_texts, max_generate_length, temperature, cfg_value, request_id
        )
        
        for m in batch_metrics:
            results.add(m)
            print(f"  Request {m.request_id + 1}: "
                  f"TTFB={m.ttfb*1000:.0f}ms, "
                  f"Total={m.total_time:.2f}s, "
                  f"Audio={m.audio_duration:.2f}s, "
                  f"RTF={m.rtf:.3f}x")
        
        request_id += batch_size
        remaining -= batch_size
    
    results.total_end_time = time.perf_counter()
    
    return results


async def main_async(args):
    print(f"Connecting to server at {args.server}...")
    
    # Check server health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{args.server}/health", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: Cannot connect to server at {args.server}")
            print(f"Make sure the FastAPI server is running:")
            print(f"  cd fastapi && uvicorn app:app --host 0.0.0.0 --port 8000")
            raise SystemExit(1)
    
    print("Server ready!\n")
    
    # Warmup
    if args.warmup > 0:
        print(f"Warming up with {args.warmup} requests...")
        await run_benchmark(
            args.server,
            num_requests=args.warmup,
            concurrency=min(args.warmup, args.concurrency),
            max_generate_length=args.max_generate_length,
            temperature=args.temperature,
            cfg_value=args.cfg_value,
        )
        print("Warmup complete.\n")
    
    # Run benchmark
    print(f"Starting benchmark: {args.num_requests} requests, concurrency={args.concurrency}")
    results = await run_benchmark(
        args.server,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_generate_length=args.max_generate_length,
        temperature=args.temperature,
        cfg_value=args.cfg_value,
    )
    
    results.print_report()


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark - calls FastAPI server")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="FastAPI server URL")
    parser.add_argument("--num-requests", type=int, default=32,
                        help="Total number of requests to run")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent requests")
    parser.add_argument("--max-generate-length", type=int, default=400,
                        help="Max audio tokens per request (~15s)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--cfg-value", type=float, default=1.5,
                        help="CFG value for classifier-free guidance")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup requests (server already warm)")
    
    args = parser.parse_args()
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
