from openai import AsyncOpenAI
import asyncio
from loguru import logger
from typing import Union, Literal, List, Dict, Any, Optional
import json
import re
from config.settings import CONFIG
from pydantic import BaseModel, Field
import time
from tenacity import AsyncRetrying, retry, stop_after_attempt, wait_random_exponential, RetryError


class ConnectionConfig(BaseModel):
    base_url: str
    api_key: str
    model_name: str

    model_config = {
        "protected_namespaces": ()
    }

class ModelParams(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 8192
    stream: bool = False
    enable_thinking: bool = False
    timeout: int = 30

class ChatResponse(BaseModel):
    """
    单次 LLM 调用的响应
    """
    answer: str
    usage: Optional[Dict[str, int]] = None
    latency: float

class ChatModel:
    def __init__(
            self,
            chat_name: str,
            model_name: str,
            model_params: ModelParams,
            client: AsyncOpenAI,
    ):
        self.chat_name = chat_name
        self.model_name = model_name
        self.default_model_params = model_params
        self.client = client

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        提供一个同步的便捷调用方法，仅返回答案字符串。
        """
        try:
            # 确保调用 chat 并只获取字符串结果
            if "return_full_response" in kwargs:
                return_full_response = kwargs.pop("return_full_response")
            else:
                return_full_response = False
            return asyncio.run(self.chat(prompt=prompt, return_full_response=return_full_response, **kwargs))
        except RetryError as e:
            logger.error(f"LLM call failed after multiple retries: {e}")
            raise ConnectionError("LLM chat error. Out of retried times.") from e

    async def _execute_single_call(self, prompt: str, **kwargs) -> ChatResponse:
        if prompt is None:
            raise ValueError("Prompt must be provided")

        begin_time = time.time()
        
        # 合并和覆盖模型参数
        runtime_params = self.default_model_params.model_copy(update=kwargs)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=runtime_params.temperature,
                max_tokens=runtime_params.max_tokens,
                stream=runtime_params.stream,
                extra_body={"chat_template_kwargs": {"enable_thinking": runtime_params.enable_thinking}},
                timeout=runtime_params.timeout,
            )
            
            answer = response.choices[0].message.content
            usage_info = None
            if response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            latency = time.time() - begin_time

            # logger.info(f"LLM call successful for chat_name='{self.chat_name}'. Latency: {latency:.4f}s")
            # if usage_info:
            #      logger.info(f"Usage - Prompt: {usage_info.get('prompt_tokens', 0)}, Completion: {usage_info.get('completion_tokens', 0)}, Total: {usage_info.get('total_tokens', 0)}")

            return ChatResponse(answer=answer, usage=usage_info, latency=latency)

        except Exception as e:
            logger.warning(f"LLM chat error on attempt. Exception: {e}. Retrying...")
            raise  # Reraise exception to trigger tenacity's retry mechanism

    async def chat(self, prompt: str, retried_times: int = 0, return_full_response: bool = False, **kwargs) -> Union[str, ChatResponse]:
        """
        与大语言模型进行交互，并提供可配置的重试机制。

        :param prompt: 输入的提示词。
        :param retried_times: 发生异常时的重试次数。设为 0 则不重试。
        :param return_full_response: 若为 True，返回包含详细信息的 ChatResponse 对象。默认为 False，只返回答案字符串。
        :param kwargs: 传递给模型接口的其他参数 (如 temperature, max_tokens)。
        :return: 默认返回答案字符串，或在指定时返回 ChatResponse 对象。
        """
        # 配置 tenacity 重试器
        retryer = AsyncRetrying(
            stop=stop_after_attempt(retried_times + 1),  # +1 因为第一次尝试不算重试
            wait=wait_random_exponential(min=1, max=30),
            reraise=True  # 确保在重试耗尽后，最终的异常被抛出
        )

        try:
            response_obj = None
            async for attempt in retryer:
                with attempt:
                    response_obj = await self._execute_single_call(prompt, **kwargs)
        except RetryError as e:
            logger.error(f"LLM call failed after {retried_times} retries. Final exception: {e}")
            raise ConnectionError(f"LLM chat error after {retried_times} retries.") from e

        if return_full_response:
            return response_obj
        else:
            return response_obj.answer

    async def chat_with_json_response(
        self,
        prompt: str,
        retried_times: int = 0,
        lowercase: bool = False,
        return_full_response:bool=False,
        **kwargs
    ) -> Union[list, dict]:
        """
        获取并解析用 ```json ``` 包裹的返回值。
        在所有重试失败时，记录“最后一次响应”的完整输出。
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(retried_times + 1),
            wait=wait_random_exponential(min=1, max=10),
            reraise=True,  # 注意：最后会抛最后一次的原始异常，而非 RetryError
        )

        last_response_str: Optional[str] = None

        async def attempt_parse():
            nonlocal last_response_str

            response_str = await self.chat(prompt=prompt, retried_times=retried_times, return_full_response=return_full_response, **kwargs)
            last_response_str = response_str  # 保存完整文本

            try:
                matches = re.findall(r'```(?:json|JSON)?\s*(.*?)```', response_str, re.DOTALL)
                if not matches:
                    logger.warning("No ```json``` block found, attempting to parse the whole response.")
                    raw_json_str = response_str
                else:
                    raw_json_str = matches[0].strip()

                cleaned_json_str = self._clean_json_string(raw_json_str)
                result = json.loads(cleaned_json_str)

                if lowercase:
                    result = self._lowercase_json(result)

                return result

            except (json.JSONDecodeError, ValueError) as e:
                # 中间尝试：只打印截断片段，避免日志爆量
                cleaned_snippet = (response_str[:100]).replace("\n", " ")
                # 用 loguru 的 {} 占位
                logger.warning(
                    "Failed to parse JSON response. Response snippet: '{}'. Exception: {}. Retrying parse...",
                    cleaned_snippet,
                    e,
                )
                raise ValueError("Failed to parse JSON response.") from e

        try:
            async for attempt in retryer:
                with attempt:
                    return await attempt_parse()

        # 关键：reraise=True 时，这里接到的是“最后一次的原始异常”（如 ValueError）
        except Exception as e:
            if last_response_str is not None:
                # 最后一轮失败：打印完整输出
                logger.error(
                    "Failed to parse JSON after multiple retries. Last full response below:\n{}",
                    last_response_str,
                )
            else:
                logger.error("Failed to parse JSON after multiple retries. No response captured.")
            # 保持对外一致的异常
            raise ValueError("Failed to parse JSON response after multiple retries.") from e


    def _clean_json_string(self, s: str) -> str:
        s = re.sub(r',\s*(\]|\})', r'\1', s)
        s = ''.join(c for c in s if c >= ' ' or c in '\r\n\t')
        return s

    def _lowercase_json(self, obj):
        if isinstance(obj, dict):
            return {str(k).lower(): self._lowercase_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._lowercase_json(i) for i in obj]
        elif isinstance(obj, str):
            return obj.lower()
        else:
            return obj

class MetricsTracker:
    """一个简单的、有状态的指标追踪器，用于单个任务范围。"""
    def __init__(self, tracker_name: str = "DefaultTracker"):
        self.tracker_name = tracker_name
        self.total_tokens: int = 0
        self.total_latency: float = 0.0
        self.request_count: int = 0

    def record(self, response: ChatResponse):
        """记录一次调用的指标。"""
        if response.usage and response.usage.get('total_tokens'):
            self.total_tokens += response.usage['total_tokens']
        self.total_latency += response.latency
        self.request_count += 1

    @property
    def average_latency(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count

    def report(self):
        """打印一份格式化的报告。"""
        print(f"--- Metrics Report for: {self.tracker_name} ---")
        print(f"Total Requests: {self.request_count}")
        print(f"Total Tokens Consumed: {self.total_tokens}")
        print(f"Average Latency: {self.average_latency:.4f}s")
        print("--------------------------------------")


class ChatModelFactory:
    @staticmethod
    def create(llm_engine: str, **overrides: Any) -> ChatModel:
        llm_cfg = CONFIG.get("llm", {}).get(llm_engine)
        if not llm_cfg:
            raise ValueError(
                f"Model '{llm_engine}' not found in CONFIG['llm']. "
                "Expected CONFIG['llm'][llm_engine] to contain base_url/api_key."
            )

        conn_config_data: Dict[str, Any] = {
            "base_url": llm_cfg.get("base_url"),
            "api_key": llm_cfg.get("api_key"),
            "model_name": llm_cfg.get("model_name"),
        }
        model_params_data: Dict[str, Any] = {}

        # 覆盖默认参数
        for k, v in overrides.items():
            if k in ConnectionConfig.model_fields:
                conn_config_data[k] = v
            elif k in ModelParams.model_fields:
                model_params_data[k] = v
            else:
                logger.debug(f"Ignored unknown override key: {k}")

        connection_config = ConnectionConfig(**conn_config_data)
        model_params = ModelParams(**model_params_data)

        client = AsyncOpenAI(
            base_url=connection_config.base_url,
            api_key=connection_config.api_key,
            timeout=model_params.timeout,
            max_retries=0  # 由 tenacity 控制重试
        )

        logger.info(f"[ChatModel] Init → customized_name={llm_engine} | llm_engine={conn_config_data.get('model_name')}")
        return ChatModel(
            chat_name=llm_engine,
            model_name=connection_config.model_name,
            model_params=model_params,
            client=client,
        )



if __name__ == "__main__":
    DEFAULT_CHAT_MODEL = ChatModelFactory.create(
        llm_engine="ali",
        temperature=0.0,
        max_tokens=8192,
        stream=False,
        enable_thinking=False,
        timeout=30,
    )

    llm_res = asyncio.run(DEFAULT_CHAT_MODEL.chat("你是谁？"))
    print(llm_res)