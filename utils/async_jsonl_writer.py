import asyncio
import json
from typing import Optional
from loguru import logger

class AsyncJsonlWriter:
    """
    e.g.
        async with AsyncJsonlWriter("files/test_output.jsonl") as writer:
            await writer.write({"msg": "hello"})
            await writer.write({"msg": "world"})
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self._writer_task = asyncio.create_task(self._writer())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.queue.put(None)  # sentinel to stop the writer
        await self._writer_task     # wait for writer to finish

    async def write(self, data: dict):
        await self.queue.put(data)

    async def _writer(self):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                try:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    self.queue.task_done()
                except Exception as e:
                    logger.error("Failed to write item due to:", e)
                    logger.error("Offending item:", repr(item))
                    raise
    
    async def stop(self):
        await self.queue.put(None)
        await self._writer_task



# 运行
if __name__ == "__main__":
    async def main():
        async with AsyncJsonlWriter("files/test_output.jsonl") as writer:
            await writer.write({"msg": "hello"})
            await writer.write({"msg": "world"})
            
    asyncio.run(main())