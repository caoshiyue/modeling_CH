import openai
import asyncio
import functools
import time
from api_key import url, key

client = openai.OpenAI(
    base_url=url,
    # sk-xxx替换为自己的key
    api_key=key
)

aclient = openai.AsyncOpenAI(
    base_url=url,
    # sk-xxx替换为自己的key
    api_key=key
)

# 异步适配器
def async_adapter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return func(*args, **kwargs)  # 异步调用
        except RuntimeError:
            pass
        return asyncio.run(func(*args, **kwargs))  # 同步调用
    return wrapper

# 重试装饰器
def retry(max_retries=3, delay=1, exceptions=(Exception,)):
    """
    装饰器，用于捕获异常并重试函数。
    
    参数:
    - max_retries: 最大重试次数
    - delay: 每次重试之间的等待时间（秒）
    - exceptions: 捕获的异常类型（默认为 Exception）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)  # 尝试执行被装饰函数
                except exceptions as e:
                    retries += 1
                    print(f"函数 {func.__name__} 请求失败，第 {retries} 次重试: {e}")
                    if retries < max_retries:
                        time.sleep(delay)  # 等待指定时间后重试
                    else:
                        print(f"函数 {func.__name__} 达到最大重试次数，退出。")
                        raise  # 超过最大重试次数后抛出异常
        return wrapper
    return decorator


#异步函数难用装饰器
async def openai_response_async(**kwargs):
    retries = 0
    while retries < 20:
        try:
            #completion = await aclient.chat.completions.create(timeout=30,**kwargs)
            completion = await asyncio.wait_for(aclient.chat.completions.create(**kwargs), timeout=30)
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            print(f"API retry {retries}")
            retries += 1
            time.sleep(1.5)

@retry(max_retries=5, delay=2, exceptions=(Exception,))
def openai_response_sync(**kwargs):
    completion = client.chat.completions.create(timeout=30,**kwargs)
    return completion.choices[0].message.content

async def openai_response(**kwargs):
    """
    根据当前上下文选择同步或异步调用 openai API。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 异步上下文
        return await openai_response_async(**kwargs)
    else:
        # 同步上下文
        return openai_response_sync(**kwargs)

    