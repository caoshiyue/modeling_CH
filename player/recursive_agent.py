from response import *
import time
import re
import json
import asyncio
from .big_agent_v2 import Bigagent

#适配当前环境
player_persona =[
            "You are Alex and involved in a survive challenge. ",
            "You are Bob and involved in a survive challenge. ",
            "You are Cindy and involved in a survive challenge. " ,
            "You are David and involved in a survive challenge. ",
            "You are Eric and involved in a survive challenge. "
        ]
player_names = ["Alex", "Bob", "Cindy", "David", "Eric"]
sim_result= "{name} might choose {action}," 


class Re_agent(Bigagent):
    def __init__(self, name,persona, decision_model="gpt-4o-mini", summary_model="gpt-4o-mini", external_knowledge_api=None):
        super().__init__( name, persona, decision_model,summary_model)
        self.player_persona = player_persona 
        self.player_names = player_names
        self.player_id= player_names.index(self.name)

    @async_adapter
    async def MDP_act(self, input_text):
        """
        主流程：解析输入、构造提示、调用决策LLM、返回动作
        """
        # 如果是首次初始化背景规则，则提取背景规则
        if self.background_rules is None:
            extract_rules = await self.extract_background_rules(input_text)
            self.background_rules = extract_rules["rule"]
            self.persona = extract_rules["persona"]
            self.rule = self.construct_rule(background_rules=self.background_rules,persona = self.persona)
            self.history_prompt = self.rule

        # 1. 使用总结LLM提取关键信息
        parsed_input = await self.parse_input(input_text)

        # 提取输入信息的各部分
        #current_state = parsed_input["current_state"]
        last_step_result = parsed_input["last_step_result"] #! 历史信息过于冗余
        step_and_task = parsed_input["step_and_task"]

        # 2. 历史记录更新
        if last_step_result!="None" and last_step_result!=None:
            self.history.append(last_step_result)

        # 3. 外部知识调用（如果需要）
        external_knowledge = ""
        if self.external_knowledge_api:
            external_knowledge = await self.external_knowledge_api.query(
                #current_state=current_state,
                step_and_task=step_and_task
            )
        flat_history_text = "\n ".join([item for item in self.history])
        # 4. 构造LLM请求上下文
        external_knowledge = await self.simulate(flat_history_text,[0,1,2,3,4])
        prompt = self.construct_prompt(
            last_step_result=last_step_result,
            #current_state=current_state,
            step_and_task=step_and_task,
            external_knowledge=external_knowledge
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
    async def simulate(self, history, sub_players, depth=0, level_k=1):
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
                persona=self.player_persona[pid],
                decision_model=self.engine,
                summary_model=self.summary_llm,
                external_knowledge_api=self.external_knowledge_api,
            ) for pid in sub_players
        ]

        # 如果递归深度未达到最大值，允许虚拟对手继续模拟它们的虚拟对手
        if depth < level_k:
            simulate_tasks = [
                sub_agent.simulate(history, sub_players=sub_players+[self.player_id], depth=depth + 1, level_k=level_k)
                for sub_agent in sub_agents
            ]
            await asyncio.gather(*simulate_tasks) 
            # 根据虚拟对手的动作构造prompt
            simulation_promt="".join(sim_result.format(name=sub_agent.name, action=sub_agent.last_bidding) for sub_agent in sub_agents)
            if depth==0:
                return simulation_promt #! 获得虚拟对手后，不用再决策，实际上在外面会再次请求决策
            sys_prompt = f"You are {self.name} and involved in a survive challenge. SURVIVE CHALLENGE!!! You are one of five players in the game. Everyone is required to choose an integer between 1 and 100 in each round. The player whose chosen number is closest to (0.8 * the average of all chosen numbers) wins the round without any HP deduction. All other players will have 1 HP deducted. But if all players choose the same number, their health points are deducted together."
            sys_prompt += "Now, you need to choose an integer between 1 and 100."
            sys_prompt += f"""An game expert predict that: {simulation_promt}. 
                            According to the game expert's prediction, think carefully about your next step of strategy to be most likely to win, and finally make a decision.\n"
                            """
            messages = [
                {'role': 'system', 'content': sys_prompt},
            ]
            self.llm_response = await self.call_llm(messages, model=self.engine)

        else: # 递归最后一层k=1，根据当前state 和 history 进行决策 
            sys_prompt = f"You are {self.name} and involved in a survive challenge. SURVIVE CHALLENGE!!! You are one of five players in the game. Everyone is required to choose an integer between 1 and 100 in each round. The player whose chosen number is closest to (0.8 * the average of all chosen numbers) wins the round without any HP deduction. All other players will have 1 HP deducted. But if all players choose the same number, their health points are deducted together."
            sys_prompt += "Now, you need to choose an integer between 1 and 100."
            sys_prompt += f"""
                        Below is previous game record related to you:
                        {history}
                        """
            sys_prompt += "Think carefully about your next step of strategy to be most likely to win. Let's think step by step, and finally make a decision." 
            messages = [
                {'role': 'system', 'content': sys_prompt},
            ]
            self.llm_response = await self.call_llm(messages, model=self.engine)
        action = await self.parse_llm_output(self.llm_response)
        
        return 

    def construct_prompt(self, last_step_result,  step_and_task, external_knowledge):
        """
        构造给决策LLM的完整提示上下文
        """
        sys_prompt = f"""
        The following is last step results of game: {last_step_result}
        """
        if external_knowledge:
            step_and_task+= f""" 
            An game expert predict that: {external_knowledge}. 
            According to the game expert's prediction, think carefully about your next step of strategy to be most likely to win, and finally make a decision.
            """

        user_prompt = step_and_task
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'system', 'content': user_prompt},
        ]
        return messages
    
    @async_adapter
    async def parse_input(self, input_text):
        """
        使用总结LLM提取输入文本中的关键信息。
        包含当前状态、上一轮结果、当前步骤要求和Agent任务的合并信息。
        人称的问题，我们需要保证system以第二人称描述
        重复role 的问题，
        goal 和 动作描述的问题
        history 中 人称对不齐的问题
        """

        flat_input_text = "; ".join([f"role: {item['role']}, content: {item['content']}." for item in input_text])
        flat_input_text= self.persona + self.background_rules + flat_input_text
        sys_prompt = self.persona + f"""
        Below is game record related to you. Please extract these information related to game decisions.
        Output the result in JSON format, and ensure the JSON keys match the given schema exactly:
        Schema: 
        - last_step_result: Description of what happened in last round (if not, answer None), including the round number, actions of all players, and outcomes in natural text.
        - step_and_task: Description your acheivement of game and required action in current step in natural text. Retain the second-person perspective.
        """
        #        - current_state: Description of the current game state, you and other players' state in natural text. 
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': flat_input_text},
        ]
        n_retry=0
        while n_retry<10:
            try:
                response = await self.call_llm(messages, model=self.summary_llm)
                js_dict=self.extract_json_from_text(response)
                n_retry=10
            except: #再试一次
                n_retry+=1
        return js_dict
