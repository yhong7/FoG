import asyncio
import inspect
import os
import traceback

from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from loguru import logger
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm


def _get_type_signature(obj):
    """返回对象的嵌套类型结构签名"""
    if isinstance(obj, dict):
        return ('dict', tuple(sorted((_get_type_signature(k), _get_type_signature(v)) for k, v in obj.items())))
    elif isinstance(obj, (list, tuple, set)):
        return (type(obj).__name__, tuple(_get_type_signature(x) for x in obj))
    else:
        return type(obj).__name__



def _get_callable_path(func: Callable) -> str:
    # 获取类、函数、方法的路径
    module = inspect.getmodule(func)
    module_name = module.__name__ if module else "unknown_module"

    try:
        # 如果是类方法，找出所属类
        qualname = func.__qualname__
    except AttributeError:
        qualname = func.__name__

    try:
        source_file = inspect.getsourcefile(func)
        source_lines, lineno = inspect.getsourcelines(func)
    except Exception:
        source_file = "unknown_file"
        lineno = -1

    return f"{module_name}:{qualname}:{lineno}"

class _RateLimiter:
    """
    令牌桶限速：平均速率 <= rate（次/秒），支持突发 burst 次。
    注意，即使rate == burst 也无法控制严格最大速率，只是限制滑动窗口大小 == 滚动步长 == 1s 的情况下的最大速率
    """
    def __init__(self, rate: Optional[float], burst: Optional[int]):
        """
        :param rate: 每秒生成多少令牌
        :param burst: 桶的容量，最多积累多少令牌（支持突发）
        """
        self.rate = rate  # 如果rate == None 则不控制速率
        self.capacity = float(burst) if burst and burst > 0 else rate
        self._tokens = self.capacity  # 初始填满，允许立即突发
        self._last: Optional[float] = None
        self._lock = asyncio.Lock()

    async def acquire(self):
        if not self.rate or self.rate <= 0 or self.capacity <= 0:
            return
        loop = asyncio.get_running_loop()
        while True:
            async with self._lock:
                now = loop.time()
                if self._last is None:
                    self._last = now
                # 补充令牌
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                    self._last = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return  # 拿到一个令牌，立即返回

                # 不足 1 个令牌：计算需要等待多久（锁外睡眠）
                need = 1.0 - self._tokens
                sleep_for = need / self.rate  # rate>0 已保证
            await asyncio.sleep(sleep_for)

class TaskWrapper(BaseModel):
    """
    用于包装异步任务的类。
    """
    # 参数相关
    index: int = Field(default=None, description="结果在列表中的索引")
    args: Tuple = Field(default_factory=tuple, description="传入的参数，是一个参数元组")
    default_kwargs: Dict[str, Any] = Field(default_factory=dict, description="实例AsyncRunner的全局kwargs，是一个dict")
    override_kwargs: Dict[str, Any] = Field(default_factory=dict, description="run的时候新的全局kwargs，会覆盖原参数，是一个dict")
    tqdm_update_val: int = Field(default=1, description="完成这个task时增长几个tqdm")

    # 重试相关
    max_retries: int = Field(default=0, ge=0, description="允许最大重试次数")
    attempts: int = Field(default=0, ge=0, description="已尝试次数")

    # 异常相关
    exception: Optional[str] = Field(default=None, description="抛出的异常信息（如果有）")
    default_value: Optional[Any] = Field(default=None, description="异常时的默认值，默认返回None")
    future: Any = Field(default=None, description="用于 append 局部等待的 Future")  # 新增

    def should_retry(self) -> bool:
        return self.attempts < self.max_retries

    def increment_attempts(self) -> None:
        object.__setattr__(self, 'attempts', self.attempts + 1)


class AsyncRunner:
    """
    异步任务并发执行工具类。

    支持对同步或异步函数进行批量并发调用，带超时控制和进度条显示。

    Args:
        func (Callable): 要并发执行的函数，支持同步或异步函数。
        max_concurrency (int): 并发任务数量。
        if_tqdm (bool, optional): 是否显示进度条，默认 True。
        batch_detection (bool, optional): 在进度条中是否按照实际batch长度更新 (默认视第一个arg为batch输入)
        unpack (bool, optional): 是否对输入参数进行解包。若输入为 List[[*args], ...] 设为 True，默认 False。
        func_desc (str, optional): func 的描述名，默认使用 func 的函数名。用于log的输出和tqdm的描述
        timeout (float, optional): 单个任务超时时间（秒），默认无超时。
        max_retries (int): 最大尝试次数，0为不重试
        **kwargs: 传递给 func 的默认固定参数。

    Usage:
        runner = AsyncRunner(func=my_func, max_concurrency=4)
        input_lst = [input1, input2, input3]
        results_lst = asyncio.run(runner.run(input_lst))  # ret: [output1, output2, output3]

    Returns:
        List[Any]: 按输入顺序返回每个任务的执行结果，异常时对应结果为 None。
    """
    def __init__(
            self,
            func: Callable, 
            max_concurrency: int, 
            if_tqdm:Optional[bool]=True,
            batch_detection: Literal["on", "off", "auto"]="auto",
            unpack: bool=False,
            func_desc: Optional[str] = None,
            timeout: Optional[float] = None,
            max_retries: Optional[int] = 0,
            max_rate: Optional[float] = None,
            burst: Optional[int] = None,
            **kwargs
            ):
        self.func = func
        self.max_concurrency = max_concurrency
        self.if_tqdm = if_tqdm
        self.batch_detection = batch_detection
        self.unpack = unpack  # 输入若为List[param, param]则设置False，若输入为List[[*model_params], [*model_params]]则设置为True
        self.func_desc = func_desc if func_desc else _get_callable_path(func)
        self.timeout = timeout
        self.max_retries = max_retries

        self.default_kwargs = kwargs  # 存储func的固定参数

        self.debug_mode = bool(os.getenv('ASYNC_RUNNER_DEBUG_MODE'))
        if self.debug_mode:
            logger.debug("AsyncRunner debug mode is ON (ASYNC_RUNNER_DEBUG_MODE environment variable is set).")
        self.exception_queue = asyncio.Queue() # 用于存储异常的队列


        # 限速器
        self._rate_limiter = _RateLimiter(max_rate, burst)

        # append的参数
        self._queue: Optional[asyncio.Queue] = None
        self._workers: List[asyncio.Task] = []
        self._results: List[Any] = []
        self._next_index: int = 0
        self._pbar: Optional[tqdm] = None
        self._is_batch_for_tqdm: bool = False
        self._append_lock = asyncio.Lock()
        self._running: bool = False


    async def _worker(self, queue: asyncio.Queue, pbar: tqdm, results_lst: List[Any]):
        while True: # 持续运行，直到被取消或者异常发生
            task = TaskWrapper()
            done_this_attempt = False  # 防止报错重试时进度条多记

            try:  # 如果没拿到任务，不能调用 pbar.update 和 queue.task_done
                # 获取任务并解包
                task: TaskWrapper = await queue.get()
                final_kwargs = {**task.default_kwargs, **task.override_kwargs}

            except asyncio.CancelledError:  # 用于处理queue为空、加载时被取消
                if task.index is not None:
                    logger.warning(f"{self.func_desc} --task_index={task.index}  [Worker Exit] Being cancelled. Cleaning up...")
                raise 

            try:  # 已经从 queue 拿到任务，需要调用 pbar.update 和 queue.task_done。因此需要拆分成两个try。
                # 限速
                await self._rate_limiter.acquire()
                if asyncio.iscoroutinefunction(self.func):
                    result = await asyncio.wait_for(
                        self.func(*task.args, **final_kwargs), timeout=self.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.func, *task.args, **final_kwargs), timeout=self.timeout
                    )
                results_lst[task.index] = result
                if task.future is not None:
                    task.future.set_result(result)   # 就地完成本任务的局部 Future
                done_this_attempt = True

            except asyncio.CancelledError:  # 用于处理运行中被取消
                if task.index is not None:
                    logger.warning(f"{self.func_desc} --task_index={task.index}  [Worker Exit] Being cancelled. Cleaning up...")
                raise

            except Exception as e:  # 用于处理拿到任务后，任务异常
                exc_tb = traceback.format_exc()
                task.exception = exc_tb
                logger.error(f"{self.func_desc} --task_index={task.index}  [Worker Exception] Exception occurred:\n{exc_tb}")

                if self.debug_mode:
                    await self.exception_queue.put(e)
                elif task.should_retry():
                    task.increment_attempts()
                    await queue.put(task)
                else:
                    results_lst[task.index] = task.default_value
                    if task.future is not None:
                        task.future.set_result(task.default_value)
                    done_this_attempt = True
            finally:
                if self.if_tqdm and pbar is not None and done_this_attempt:
                    pbar.update(task.tqdm_update_val)
                queue.task_done()

    async def _ensure_started(self):
        # 懒启动：首次 append 自动启动 worker/tqdm
        if self._running:
            return
        self._running = True
        self._queue = asyncio.Queue()
        self._results = []
        self._next_index = 0
        self._is_batch_for_tqdm = False
        self._pbar = tqdm(total=0, desc=self.func_desc) if self.if_tqdm else None
        self._workers = [
            asyncio.create_task(self._worker(self._queue, self._pbar, self._results))
            for _ in range(self.max_concurrency)
        ]

    def append(self, items: Any, **override_kwargs):
        """
        非阻塞提交：
        - 传单个 item -> 返回一个 awaitable 句柄 (Future)
        - 传列表  -> 返回同等长度的句柄列表
        需要结果时再 await 这些句柄。后台 worker 并发执行。
        """
        loop = asyncio.get_running_loop()
        single_input = not isinstance(items, list)
        if single_input:
            items = [items]

        # 预先为每个 item 准备一个 Future 作为“结果句柄”，立即可返回
        futs = [loop.create_future() for _ in items]

        async def _enqueue_many():
            await self._ensure_started()

            first_append = (self._next_index == 0)
            # 首次 append：只决定是否 batch；不要立刻改 total
            if self.if_tqdm and self._pbar is not None and first_append:
                is_batch, _ = self._get_tqdm_update_val(items)
                self._is_batch_for_tqdm = is_batch

            added_total = 0
            async with self._append_lock:
                for elem, fut in zip(items, futs):
                    args = elem if self.unpack else (elem,)
                    if self.unpack and not isinstance(args, tuple):
                        args = tuple(args)

                    # 索引与结果位
                    self._results.append(None)
                    idx = self._next_index
                    self._next_index += 1

                    # tqdm 该任务完成时的增量
                    tqdm_val = len(args[0]) if (self._is_batch_for_tqdm and isinstance(args[0], list)) else 1
                    added_total += tqdm_val

                    # 入队（worker 完成后会 set_result 到 fut）
                    task = TaskWrapper(
                        index=idx,
                        args=args,
                        default_kwargs=self.default_kwargs,
                        override_kwargs=override_kwargs,
                        max_retries=self.max_retries,
                        tqdm_update_val=tqdm_val,
                        future=fut,
                    )
                    await self._queue.put(task)

                # 更新 total（真正的进度递增在 worker 完成时）
                if self.if_tqdm and self._pbar is not None and added_total:
                    if first_append:
                        self._pbar.total = added_total
                    else:
                        self._pbar.total += added_total
                    self._pbar.refresh()

        # 把“入队工作”丢到后台，不阻塞当前调用
        asyncio.create_task(_enqueue_many())

        return futs[0] if single_input else futs

    async def finish(self) -> List[Any]:
        if not self._running or self._queue is None:
            return []
        try:
            if self.debug_mode:
                join_task = asyncio.create_task(self._queue.join())
                exc_task = asyncio.create_task(self.exception_queue.get())
                done, pending = await asyncio.wait([join_task, exc_task], return_when=asyncio.FIRST_COMPLETED)
                if exc_task in done:
                    for t in pending: t.cancel()
                    raise exc_task.result()
                else:
                    exc_task.cancel()
            else:
                await self._queue.join()
        finally:
            for w in self._workers:
                w.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            if self.if_tqdm and self._pbar is not None:
                self._pbar.close()
            # 清理状态
            res = list(self._results)
            self._queue = None
            self._workers = []
            self._results = []
            self._next_index = 0
            self._pbar = None
            self._is_batch_for_tqdm = False
            self._running = False
        return res

    async def _run_async(self, input_lst: List[Any], **override_kwargs) -> List[Any]:
        """
        :param input_lst: (List[Any]): 输入的参数列表。
        :param **override_kwargs: 运行时附加参数。
        :return: List[Any]: 每个任务的结果（顺序与 input_lst 保持一致）。
        """

        # 预定义结果列表
        results_lst = [None] * len(input_lst)

        # 初始化tqdm
        pbar = None
        is_batch = False
        if self.if_tqdm:
            is_batch, tqdm_len = self._get_tqdm_update_val(input_lst)  # 尝试自动检测是否为batch
            pbar = tqdm(total=tqdm_len, desc=self.func_desc)

        # 填充队列
        queue = asyncio.Queue()
        for index, args in enumerate(input_lst):
            if not self.unpack:  # 是否解包
                args = (args, )
            task = TaskWrapper(
                index=index,
                args=args,
                default_kwargs=self.default_kwargs,
                override_kwargs=override_kwargs,
                max_retries=self.max_retries,
                tqdm_update_val=len(args[0]) if is_batch else 1,
            )
            await queue.put(task)


        # 启动 workers
        workers = [
            asyncio.create_task(self._worker(queue, pbar, results_lst))
            for _ in range(self.max_concurrency)
        ]

        try:
            if self.debug_mode:
                join_task = asyncio.create_task(queue.join())
                exc_task = asyncio.create_task(self.exception_queue.get())

                done, pending = await asyncio.wait(
                    [join_task, exc_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                if exc_task in done:
                    # 有异常：主动 raise 并取消 join
                    for t in pending:  # 取消 pending 中的所有 tasks，这里其实只有一个 join_task
                        t.cancel()
                    raise exc_task.result()  # 直接抛异常，在 finally 中取消所有 workers
                else:
                    exc_task.cancel()  # 否则 join 成功，取消 exception_task
                    
            else:
                await queue.join()
            
        except Exception as e:
            raise
        
        finally:
            # 无论是否发生异常, 都要取消所有 worker 并清理
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if self.if_tqdm:
                pbar.close()
                
        return results_lst


    def _get_tqdm_update_val(self, input_lst: List[Any]):
        """
        获取实际元素长度
        demo: [1,2,3,4,5,6] -> 6
        [[1,2,3], [4,5,6], [7,8,9] -> 9
        [[1, "a"], [2, "b"], [3, "c"]] -> 3
        :return: is_batch, tqdm_len
        """
        # # 判断列表中所有元素是否拥有相同的嵌套结构和类型
        # if self.batch_detection == "off":
        #     return False, len(input_lst)
        #
        # items = [args[0] if self.unpack else args for args in input_lst]
        # if not all(isinstance(item, list) for item in items):
        #     return False, len(input_lst)
        #
        # flattened = [sub_item for item in items for sub_item in item]
        # if not flattened:
        #     return True, 0  # 没有实际元素，total=0
        #
        # # 检查所有元素是否相同
        # if self.batch_detection == "auto":
        #     first_sig = _get_type_signature(flattened[0])
        #     if not all(_get_type_signature(x) == first_sig for x in flattened):
        #         return False, len(input_lst)

        # return True, len(flattened)
        return False, len(input_lst)


    async def run(self, input_lst: List[Any], **override_kwargs) -> list:
        return await self._run_async(input_lst,  **override_kwargs)


def demo_case(use_async: bool = True):
    # 功能测试
    import asyncio
    import random
    import time

    print("=" * 50)
    print(f"{'异步' if use_async else '同步'} 测试开始")

    if use_async:
        async def test_func(x: int, delay: float = 1.0, label: str = "") -> int:
            print(f"[Async] Begin x={x}, label={label}")
            await asyncio.sleep(random.uniform(0.5, delay))
            if x in [2]:
                raise ValueError(f"Simulated error for x={x}")
            print(f"[Async] End x={x}, label={label}")
            return x * x
    else:
        def test_func(x: int, delay: float = 1.0, label: str = "") -> int:
            print(f"[Sync] Begin x={x}, label={label}")
            time.sleep(random.uniform(0.5, delay))
            if x in [2]:
                raise ValueError(f"Simulated error for x={x}")
            print(f"[Sync] End x={x}, label={label}")
            return x * x

    # 构造 runner
    runner = AsyncRunner(
        func=test_func,
        max_concurrency=3,
        unpack=True,  # 参数解包模式
        timeout=2.0,  # 单个任务最多运行2秒
        label="default_label"  # 默认参数
    )

    input_args = [(i, 1.2) for i in range(1, 6)]  # 任务输入为 (x, delay)

    # 加入 max_retries 和 default_value（通过 patch TaskWrapper 或者内部改造）
    # 这里只是演示默认值覆盖
    async def run_all():
        # 1. 默认运行（部分任务会失败）
        print("\n--- Run: 默认执行（无重试） ---")
        results = await runner.run(input_args, label="no_retry")
        print("Results:", results)

        # 2. 模拟重试功能（通过修改 runner 内部或 TaskWrapper patch 实现）
        print("\n--- Run: 模拟失败重试（max_retries=2） ---")
        runner_with_retry = AsyncRunner(
            func=test_func,
            max_concurrency=3,
            unpack=True,
            timeout=2.0,
            label="default_label",
            max_retries=2
        )
        retry_results = await runner_with_retry.run(input_args, label="retry")
        print("Results with retry:", retry_results)

        # 3. 测试 unpack=False 情况（输入为 List[Any] 而非 List[Tuple]）
        print("\n--- Run: unpack=False 测试 ---")
        def simple_func(x: int) -> int:
            time.sleep(0.2)
            return x + 10

        runner_unpack_false = AsyncRunner(
            func=simple_func,
            max_concurrency=2,
            unpack=False
        )

        result_unpack_false = await runner_unpack_false.run(list(range(5)))
        print("unpack=False Results:", result_unpack_false)

        # 4. 测试 debug 模式（将环境变量打开）
        print("\n--- Run: debug 模式测试 ---")
        os.environ["ASYNC_RUNNER_DEBUG_MODE"] = "true"

        async def error_func(x: int) -> int:
            await asyncio.sleep(0.1)
            if x == 1:
                raise RuntimeError("Debug mode exception!")
            return x * 2

        runner_debug = AsyncRunner(func=error_func, max_concurrency=2)
        try:
            await runner_debug.run([0, 1, 2])
        except Exception as e:
            print("Caught debug exception:", repr(e))
        finally:
            os.environ.pop("ASYNC_RUNNER_DEBUG_MODE", None)

    asyncio.run(run_all())

    print(f"{'异步' if use_async else '同步'} 测试结束")
    print("=" * 50)


async def demo_case2():
    # batch测试
    async def a(inputs):
        await asyncio.sleep(0.1)
        return [i**2 for i in inputs]
    runner = AsyncRunner(a, max_concurrency=10, max_rate=15)
    inputs = [[i for i in range(j, j+5)] for j in range(100)]
    res = await runner.run(inputs)
    print(res)
    return res

async def demo_case3():
    # append测试
    import random
    async def test_func(int_input, sleep_time):
        if random.random() < 0.01:
            raise Exception
        await asyncio.sleep(sleep_time)
        return int_input * 3

    runner = AsyncRunner(func=test_func, max_concurrency=3, unpack=True, max_retries=0)
    # 懒启动 + 并发提交多次 append（此处不立刻 await）
    tasks = [runner.append((i, 0.1)) for i in range(50)]
    res = [await t for t in tasks[:]]
    print("res:", res)
    # 全部处理完再收尾（如果你需要等队列清空并关闭 worker）
    await runner.finish()

if __name__ == "__main__":
    # os.environ['ASYNC_RUNNER_DEBUG_MODE'] = 'true'
    # demo_case(use_async=True)
    asyncio.run(demo_case3())