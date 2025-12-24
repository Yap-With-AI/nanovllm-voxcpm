"""
Streaming TTS Example
Demonstrates real-time audio streaming with VoxCPM
"""

from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
import time
import argparse


async def main(args):
    print("Loading model...")
    server = VoxCPM.from_pretrained(
        model=args.model,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        max_model_len=512,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        devices=[args.device],
    )
    await server.wait_for_ready()
    print("Model ready!\n")

    print(f"Generating audio for: \"{args.text[:80]}{'...' if len(args.text) > 80 else ''}\"")
    print(f"Temperature: {args.temperature}, CFG: {args.cfg_value}")
    print("-" * 60)

    chunks = []
    chunk_count = 0
    start_time = time.perf_counter()
    first_chunk_time = None

    async for chunk in server.generate(
        target_text=args.text,
        temperature=args.temperature,
        cfg_value=args.cfg_value,
        max_generate_length=args.max_generate_length,
    ):
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

    end_time = time.perf_counter()
    print()  # newline after progress

    # Combine all chunks
    wav = np.concatenate(chunks, axis=0)
    
    # Stats
    total_time = end_time - start_time
    audio_duration = len(wav) / 16000
    rtf = total_time / audio_duration
    xrt = audio_duration / total_time

    print("-" * 60)
    print(f"Total time:     {total_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Chunks:         {chunk_count}")
    print(f"RTF:            {rtf:.3f}x {'(real-time capable)' if rtf < 1 else ''}")
    print(f"XRT:            {xrt:.2f}x real-time")

    # Save output
    sf.write(args.output, wav, 16000)
    print(f"\nSaved to: {args.output}")

    await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming TTS Example")
    parser.add_argument("--model", type=str, default="openbmb/VoxCPM1.5",
                        help="Model path or HuggingFace repo ID")
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
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID")

    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args))
