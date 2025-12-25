import aiohttp
import asyncio
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import sys


async def tts_request(session: aiohttp.ClientSession, server: str, text: str):
    url = f"{server}/generate"
    async with session.post(url, json={"target_text": text}) as response:
        response.raise_for_status()
        sample_rate_str = response.headers.get("X-Sample-Rate", "44100")
        sample_rate = int(sample_rate_str) if sample_rate_str else 44100
        dtype = response.headers.get("X-Dtype", "float32")
        raw = await response.content.read()
        audio = np.frombuffer(raw, dtype=np.dtype(dtype))
        return audio, sample_rate


async def run(texts, server: str, concurrency: int, save_first: int, out_dir: Path):
    # Normalize server URL - remove trailing slash to avoid double slashes
    server = server.rstrip("/")
    connector = aiohttp.TCPConnector(limit=concurrency)
    tasks = set()
    cnt = 0
    task_idx = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting {len(texts)} request(s) to {server} with concurrency={concurrency}")

    async with aiohttp.ClientSession(connector=connector) as session:
        while task_idx < len(texts) or tasks:
            while len(tasks) < concurrency and task_idx < len(texts):
                tasks.add(asyncio.create_task(tts_request(session, server, texts[task_idx])))
                task_idx += 1

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    audio, sample_rate = task.result()
                except aiohttp.ClientResponseError as e:
                    print(f"Request failed: {e.status}, message='{e.message}', url='{e.request_info.url}'", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"Request failed: {e}", file=sys.stderr)
                    continue
                if cnt < save_first:
                    sf.write(out_dir / f"tts_{cnt}.wav", audio, sample_rate)
                    print(f"Saved {out_dir / f'tts_{cnt}.wav'} @ {sample_rate} Hz")
                cnt += 1
                print(f"Processed {cnt}/{len(texts)} tasks")

    print("Done.")


def load_texts(args) -> list[str]:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            raise ValueError("Input file is empty.")
        return lines
    if not args.text:
        raise ValueError("Provide --text or --input.")
    return [args.text]


def main():
    parser = argparse.ArgumentParser(description="Simple TTS client for remote FastAPI server")
    parser.add_argument("--server", type=str, required=True, help="Server base URL, e.g. http://host:port")
    parser.add_argument("--text", type=str, help="Single text to synthesize")
    parser.add_argument("--input", type=str, help="Path to a file with one text per line")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests")
    parser.add_argument("--save-first", type=int, default=5, help="How many responses to save to WAV")
    parser.add_argument("--out-dir", type=str, default="outs", help="Directory to save WAVs")

    args = parser.parse_args()
    texts = load_texts(args)
    asyncio.run(run(texts, args.server, args.concurrency, args.save_first, Path(args.out_dir)))


if __name__ == "__main__":
    main()

