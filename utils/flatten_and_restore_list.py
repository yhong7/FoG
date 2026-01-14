from functools import wraps
from typing import Callable, Any, List, Tuple, Union
import inspect
import asyncio


def flatten_with_template(nested: Any) -> Tuple[List[Any], Any]:
    flat = []

    def _flatten(x):
        if isinstance(x, list):
            return [_flatten(i) for i in x]
        else:
            flat.append(x)
            return None

    template = _flatten(nested)
    return flat, template


def restore_from_template(flat: List[Any], template: Any) -> Any:
    flat_iter = iter(flat)

    def _restore(tmpl):
        if tmpl is None:
            return next(flat_iter)
        elif isinstance(tmpl, list):
            return [_restore(i) for i in tmpl]
        else:
            raise ValueError("Unexpected template element")

    return _restore(template)


def flatten_and_restore_list(flatten_param_name: str):
    """
    装饰器工厂：指定某个参数名(List[Any])，对其进行 flatten + 执行 + restore。
    兼容同步/异步函数。
    """
    def decorator(func: Callable):
        is_async = inspect.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()

            if flatten_param_name not in bound.arguments:
                raise ValueError(f"Parameter '{flatten_param_name}' not found in function arguments.")

            nested_value = bound.arguments[flatten_param_name]
            flat_value, template = flatten_with_template(nested_value)
            bound.arguments[flatten_param_name] = flat_value

            result = await func(*bound.args, **bound.kwargs)
            return restore_from_template(result, template)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()

            if flatten_param_name not in bound.arguments:
                raise ValueError(f"Parameter '{flatten_param_name}' not found in function arguments.")

            nested_value = bound.arguments[flatten_param_name]
            flat_value, template = flatten_with_template(nested_value)
            bound.arguments[flatten_param_name] = flat_value

            result = func(*bound.args, **bound.kwargs)
            return restore_from_template(result, template)

        return async_wrapper if is_async else sync_wrapper

    return decorator

if __name__ == "__main__":
    # 同步
    @flatten_and_restore_list("x")
    def multiply(x:list):
        res = [{k: v * 2} for d in x for k, v in d.items()]
        return res
    print(multiply([[{'1':1}, {'2':2}], [{'3':3}, [{'4':4}]]]))
    # 输出: [[2, 4], [6, [8]]]


    # 异步
    @flatten_and_restore_list("x")
    async def async_multiply(x):
        return [i * 3 for i in x]

    async def main():
        result = await async_multiply([[1, 2], [3, [4]]])
        print(result)

    asyncio.run(main())
    # 输出: [[3, 6], [9, [12]]]
