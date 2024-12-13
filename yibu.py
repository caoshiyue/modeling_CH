import asyncio
import functools
import openai
from response import *


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

# Player 类
class Player:
    def __init__(self, name):
        self.name = name
        self.result = None

    @async_adapter
    async def act(self, **kwargs):
        print(f"Player {self.name} 正在执行动作...")
        response = await openai_response(**kwargs)
        response2 = await openai_response(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": response}
            ])
        self.result = f"Player {self.name} 的动作完成"
        print(f"Player {self.name} {response2} 动作完成.")


players = [Player(f"Player {i}") for i in range(1, 4)]

# 同步调用的方法
def run_players_sync(players, **kwargs):
    for player in players:
        player.act(**kwargs)

# 异步调用的方法
async def run_players_async(players, **kwargs):
    tasks = [player.act(**kwargs) for player in players]
    await asyncio.gather(*tasks)

# 主函数
def main():
    tongbu=True 
    if tongbu:
        # 如果当前处于异步上下文，使用异步调用
        print("正在执行异步调用...")
        asyncio.run(run_players_async(players, 
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]))
        print("检查所有异步执行完成...")
    else:
        # 如果当前处于同步上下文，使用同步调用
        print("正在执行同步调用...")
        run_players_sync(players, 
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ])

if __name__ == "__main__":
    main()