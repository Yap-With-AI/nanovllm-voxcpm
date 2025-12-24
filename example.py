"""
Streaming TTS Example - Calls the FastAPI server
"""

import numpy as np
import soundfile as sf
import time
import argparse
import httpx


def main(args):
    print(f"Connecting to server at {args.server}...")
    
    # Check server health
    try:
        with httpx.Client() as client:
            resp = client.get(f"{args.server}/health", timeout=5)
            resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {args.server}")
        print(f"Make sure the FastAPI server is running:")
        print(f"  cd fastapi && uvicorn app:app --host 0.0.0.0 --port 8000")
        raise SystemExit(1)
    
    print(f"Server ready!\n")
    print(f"Generating audio for: \"{args.text[:80]}{'...' if len(args.text) > 80 else ''}\"")
    print(f"Temperature: {args.temperature}, CFG: {args.cfg_value}")
    print("-" * 60)

    chunks = []
    chunk_count = 0
    first_chunk_time = None
    
    # Stream audio from server (fresh connection, no reuse)
    with httpx.Client(timeout=300) as client:
        # Start timing RIGHT BEFORE sending the request (not including handshake)
        start_time = time.perf_counter()
        
        with client.stream(
            "POST",
            f"{args.server}/generate",
            json={
                "target_text": args.text,
                "temperature": args.temperature,
                "cfg_value": args.cfg_value,
                "max_generate_length": args.max_generate_length,
            },
        ) as response:
            response.raise_for_status()
            
            # Each chunk is raw float32 audio samples
            buffer = b""
            chunk_size = 4 * 5120  # 5120 float32 samples = 320ms of audio
            
            for data in response.iter_bytes():
                buffer += data
                
                while len(buffer) >= chunk_size:
                    chunk_bytes = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
                    
                    chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
                    
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                        ttfb = (first_chunk_time - start_time) * 1000
                        print(f"  TTFB: {ttfb:.0f}ms")

                    chunks.append(chunk)
                    chunk_count += 1

                    # Print streaming progress
                    audio_so_far = sum(len(c) for c in chunks) / 16000
                    elapsed = time.perf_counter() - start_time
                    current_rtf = elapsed / audio_so_far if audio_so_far > 0 else 0
                    print(f"  Chunk {chunk_count}: +{len(chunk)/16000:.2f}s audio, "
                          f"total={audio_so_far:.2f}s, RTF={current_rtf:.3f}x", end="\r")
            
            # Handle remaining bytes
            if len(buffer) > 0 and len(buffer) % 4 == 0:
                chunk = np.frombuffer(buffer, dtype=np.float32)
                if len(chunk) > 0:
                    chunks.append(chunk)
                    chunk_count += 1

    end_time = time.perf_counter()
    print()  # newline after progress

    if not chunks:
        print("ERROR: No audio received from server")
        raise SystemExit(1)

    # Combine all chunks
    wav = np.concatenate(chunks, axis=0)
    
    # Stats
    total_time = end_time - start_time
    audio_duration = len(wav) / 16000
    rtf = total_time / audio_duration if audio_duration > 0 else 0
    xrt = audio_duration / total_time if total_time > 0 else 0

    print("-" * 60)
    print(f"Total time:     {total_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Chunks:         {chunk_count}")
    print(f"RTF:            {rtf:.3f}x {'(real-time capable)' if rtf < 1 else ''}")
    print(f"XRT:            {xrt:.2f}x real-time")

    # Save output
    sf.write(args.output, wav, 16000)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming TTS Example - calls FastAPI server")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="FastAPI server URL")
    parser.add_argument("--text", type=str,
                        default="Hello, this is a demonstration of real-time text to speech synthesis. "
                                "The audio is being generated and streamed chunk by chunk as it's produced.",
                        help="Text to synthesize")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--cfg-value", type=float, default=1.5,
                        help="CFG value for classifier-free guidance")
    parser.add_argument("--max-generate-length", type=int, default=400,
                        help="Max audio tokens (~15s at 400)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output WAV file path")

    args = parser.parse_args()
    main(args)
