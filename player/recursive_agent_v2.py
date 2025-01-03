from response import *
import time
import re
import json
import asyncio
from .big_agent_v2 import Bigagent

#适配当前环境
player_personas =[
            "You are Alex and involved in a survive challenge. ",
            "You are Bob and involved in a survive challenge. ",
            "You are Cindy and involved in a survive challenge. " ,
            "You are David and involved in a survive challenge. ",
            "You are Eric and involved in a survive challenge. "
        ]
player_names = ["Alex", "Bob", "Cindy", "David", "Eric"]
sim_result= "{name} might choose {action}," 
INQUIRY_HISTORY= "According to the history of game, finally make a decision." 
INQUIRY_EXPERT="According to the game expert's prediction, think carefully about your next step of strategy to be most likely to win, and finally make a decision."

class Re_agent(Bigagent):
    def __init__(self, name,persona, decision_model="gpt-4o-mini", summary_model="gpt-4o-mini", external_knowledge_api=None,background_rules=None,history=None):
        super().__init__( name, persona, decision_model,summary_model)
        self.player_personas = player_personas 
        self.player_names = player_names
        self.player_id= player_names.index(self.name)
        if history!=None:
            self.history = history
        if background_rules!=None:
            self.background_rules=background_rules


    @async_adapter
    async def MDP_act(self, input_text):
        """
        主流程：解析输入、构造提示、调用决策LLM、返回动作
        """
        # 如果是首次初始化背景规则，则提取背景规则
        if self.background_rules is None:
            extract_rules = await self.extract_background_rules(input_text)
            self.background_rules = extract_rules["Q1"]
            self.persona = extract_rules["Q2"]
            self.history_prompt = self.construct_rule(background_rules=self.background_rules,persona = self.persona)
            print(extract_rules)

        # 1. 使用总结LLM提取关键信息
        parsed_input = await self.parse_input(input_text)
        print(parsed_input)
        # 提取输入信息的各部分
        #current_state = parsed_input["current_state"]

        game_state = parsed_input["Q1"]
        agent_state = parsed_input["Q2"]
        player_state = parsed_input["Q3"]
        player_task = parsed_input["Q4"]

        step_and_task= player_state+player_task
        # 2. 历史记录更新
        if game_state!="None" and game_state!=None:
            game_state+=agent_state
        else:
            game_state=agent_state
        self.history.append(game_state)
        # 3. 外部知识调用（如果需要）
        flat_history_text = "\n ".join([item for item in self.history])
        external_knowledge = await self.simulate(game_state, player_task,[0,1,2,3,4])
        
        # 4. 构造LLM请求上下文
        prompt = self.construct_prompt(
            last_step_result=game_state,
            step_and_task=step_and_task,
            external_knowledge=external_knowledge,
            inquiry=INQUIRY_EXPERT,
        )
        
        # 5. 调用决策LLM生成输出，
        self.history_prompt=self.history_prompt+ prompt

        final_prompt= self.history_prompt
        self.llm_response = await self.call_llm(final_prompt, model=self.engine)
        self.history_prompt.append({'role': 'assistant',"content": self.llm_response})

        with open("final_prompt.json", "w", encoding="utf-8") as file:
            json.dump(final_prompt, file, ensure_ascii=False, indent=4)  # 确保中文字符可读，格式化输出
        # 6. 转换LLM输出为可执行动作/决策文本
        action = await self.parse_llm_output(self.llm_response)
        
        return action


    @async_adapter
    async def simulate(self, game_state,step_and_task, sub_players, depth=0, level_k=1):
        """
        模拟虚拟对手的动作

        :param input_text: 输入的文本
        :param sub_players: 虚拟玩家列表[], 除自己之外
        :param depth: 当前递归深度
        :param level_k: 允许的最大递归深度, k=0 不执行递归,主agent根据情况决策, k=1 等于cot, k=2 超过cot一阶,
        :return: 主代理动作
        """
        # sub_players = [index for index, name in enumerate(player_names) if name != self.name] # 主agent中初始化

        if depth > level_k:
            return None
        if self.player_id in sub_players: #排除自己
            sub_players.remove(self.player_id)

        # 初始化虚拟对手实例
        sub_agents = [
            Re_agent(
                name=self.player_names[pid],
                persona=self.player_personas[pid],
                decision_model=self.engine,
                summary_model=self.summary_llm,
                external_knowledge_api=self.external_knowledge_api,
                background_rules=self.background_rules,
                history=self.history,
            ) for pid in sub_players
        ]

        # 如果递归深度未达到最大值，允许虚拟对手继续模拟它们的虚拟对手
        simulation_promt=None
        if depth < level_k:
            simulate_tasks = [
                sub_agent.simulate(game_state, step_and_task, sub_players=sub_players+[self.player_id], depth=depth + 1, level_k=level_k)
                for sub_agent in sub_agents
            ]
            await asyncio.gather(*simulate_tasks) 
            # 根据虚拟对手的动作构造prompt
            simulation_promt="".join(sim_result.format(name=sub_agent.name, action=sub_agent.last_bidding) for sub_agent in sub_agents)
            if depth==0:
                return simulation_promt # 获得虚拟对手后，不用再决策，实际上在外面会再次请求决策
            inquiry=INQUIRY_EXPERT
        else: # 最低一层，根据历史决策
            game_state="\n ".join([item for item in self.history])
            inquiry=INQUIRY_HISTORY # 它的任务是模拟普通玩家
        #! 潜在的问题是，如果每个玩家的step_and_task不同，或者step_and_task带有玩家信息，则不对
        sys_prompt1 = self.persona +" "+self.background_rules
        sys_prompt2 = self.construct_prompt(
        last_step_result=game_state,
        step_and_task=step_and_task,
        external_knowledge=simulation_promt,
        inquiry=inquiry,
        )
        messages = [
            {'role': 'system', 'content': sys_prompt1},
        ]+sys_prompt2
        self.llm_response = await self.call_llm(messages, model=self.engine)
        #print(self.llm_response)
        action = await self.parse_llm_output(self.llm_response)
        
        return # 属性赋值，无需返回

    
