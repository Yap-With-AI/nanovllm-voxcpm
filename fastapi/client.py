import aiohttp
import asyncio
import numpy as np
import soundfile as sf

API_URL = "http://localhost:8080"
CONCURRENCY = 10


async def tts_request(session: aiohttp.ClientSession, text: str):
    async with session.post(f"{API_URL}/generate", json={"target_text": text}) as response:
        sample_rate = int(response.headers.get("X-Sample-Rate", "16000"))
        dtype = response.headers.get("X-Dtype", "float32")
        raw = await response.content.read()
        audio = np.frombuffer(raw, dtype=np.dtype(dtype))
        return audio, sample_rate


async def main():
    texts = [
        "Short English test line.",
        "Hello there!",
    ] * 50

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    tasks = set()
    cnt = 0
    task_idx = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        while task_idx < len(texts) or len(tasks) > 0:
            while len(tasks) < CONCURRENCY and task_idx < len(texts):
                tasks.add(asyncio.create_task(tts_request(session, texts[task_idx])))
                task_idx += 1

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                audio, sample_rate = task.result()
                if cnt < 10:
                    sf.write(f"test_{cnt}.wav", audio, sample_rate)
                cnt += 1
                print(f"Processed {cnt} tasks")


if __name__ == "__main__":
    asyncio.run(main())

