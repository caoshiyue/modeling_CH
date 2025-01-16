import json
import os
from typing import List, Optional
from response import *
import asyncio
import re

class MemoryRecord:
    def __init__(self, state: str, action: str):
        self.state = state
        self.action = action

    def to_dict(self):
        return {
            "state": self.state,
            "action": self.action
        }

    @staticmethod
    def from_dict(data):
        return MemoryRecord(
            state=data["state"],
            action=data["action"]
        )

class Memory:
    def __init__(self, memory_file: str = "memory.json", model=None, background=None,):
        self.memory_file = memory_file
        self.model=model
        self.background=background
        self.records: List[MemoryRecord] = []
        self.load_memory()
    
    def set_background(self,input_text):
        self.background=input_text

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.records = [MemoryRecord.from_dict(record) for record in data]
        else:
            self.records = []

    def save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump([record.to_dict() for record in self.records], f, ensure_ascii=False, indent=4)

    @async_adapter
    async def query(self, current_state: str) -> str:
        similar_index = await self.state_similar(current_state)
        if similar_index>=0:
            action = self.records[similar_index].action
            print(f"找到相似状态，使用{self.records[similar_index].state} 。")
            return similar_index,action
        else: # 由主agent进行推理
            print(f"未找到相似状态，添加新记录，索引为 {len(self.records) }。")
            return similar_index, None

    @async_adapter
    async def reflect(self, idx: int, new_last_action: str): #last_action="player choose X"
        similar_index = idx
        stored_action = self.records[similar_index].action[0]
        sim=await self.action_similar(stored_action, new_last_action)
        if sim:
            print(f"动作相似，无需修改。 记录：{self.records[similar_index].action[0][-100:]},  实际：{new_last_action}。")
            return False
        else:
            # 调用LLM进行重新推理            
            print(f"动作不相似，更新。{similar_index} 。")
            return True

    @async_adapter
    async def state_similar(self, current_state: str) -> Optional[int]:
        """
        判断current_state与memory中所有state的相似性。
        通过将所有state和索引合并成一个字符串，然后调用simple_similarity方法。
        返回相似state的索引，如果不存在则返回None。
        """
        all_states = ""
        for idx, record in enumerate(self.records):
            all_states += f"{idx}: {record.state};\n"

        flat_input_text = all_states
        sys_prompt1 = f""" You are a game expert and analyzing player's state. This is background rule of game: {self.background}"""
        sys_prompt2 = f""" You need to understand the rule of game and evaluat the recorded game states.
        If you think a record state that is similar to or can represent (within 20% of value deviation) the current player's state, answer "STATE X" at the end of reply, X is id of record_state. 
        If you think that none of the recorded player's state can represent the current state, please answer "STATE -1" at the end of reply;
        Notice that each word of current player's state must be considered. 
        """#! 这有个trick，必须删掉 
        sys_prompt3 = f"""The current player's state is "{current_state}".  Recorded player states and id are as follows (id: record_state) : """      

        messages = [
            {'role': 'system', 'content': sys_prompt1},
            {'role': 'system', 'content': sys_prompt2},
            {'role': 'system', 'content': sys_prompt3},
            {'role': 'system', 'content': flat_input_text}, 
        ]
        n_retry=0
        pattern = r"STATE (-?\d+)"
        while n_retry<10:
            try:
                response = await self.call_llm(messages, model=self.model)
                match = re.search(pattern, response[-100:])
                idx = int(match.group(1))
                n_retry=10
            except: #再试一次
                n_retry+=1

        return idx
 

    @async_adapter
    async def action_similar(self, stored_action, last_action) -> bool:
        """
        判断两个action的相似性。
        返回True如果相似，否则False。
        """
        sys_prompt1 = f""" You are a game expert and analyzing player's state. This is background rule of game: {self.background}"""
        sys_prompt2 = f""" You need to understand the rule of game and evaluate the predition of player's action.
        If you think player's action reasonablly correspond to the predition, answer "PRED 1" at the end of reply. 
        If you think player's action doesn't correspond to the predition, answer "PRED 0" at the end of reply. 
        """
        sys_prompt3 = f"""The player's action is "{last_action}". The predition of player's action is "{stored_action} """      

        messages = [
            {'role': 'system', 'content': sys_prompt1},
            {'role': 'system', 'content': sys_prompt2},
            {'role': 'system', 'content': sys_prompt3},
        ]
        n_retry=0
        pattern = r"PRED (0|1)"
        while n_retry<10:
            try:
                response = await self.call_llm(messages, model=self.model)
                match = re.search(pattern, response)
                if match.group(1) == "1":
                    return True
                else:
                    return False
                n_retry=10
            except: #再试一次
                n_retry+=1
        return False

    def add_record(self, state: str, action: str):
        """
        添加新的记录到memory中。
        """
        new_record = MemoryRecord(state=state, action=action)
        self.records.append(new_record)
        self.save_memory()
        return len(self.records)-1

    def update_action(self, index: int, new_action: str):
        """
        更新指定索引的action，并保存memory。
        """
        if 0 <= index < len(self.records):
            self.records[index].action = new_action
            self.save_memory()
        else:
            print(f"索引 {index} 超出范围，无法更新。")

    def get_action(self, index: int):
        if 0 <= index < len(self.records):
            return self.records[index].action
        else:
            return None


    @async_adapter
    async def call_llm(self, prompt, model):
        """
        调用OpenAI API的通用函数
        """
        response = await openai_response(
            model=model,
            messages=prompt,
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response
    
# 示例用法
if __name__ == "__main__":
    memory = Memory("game_memory.json")
