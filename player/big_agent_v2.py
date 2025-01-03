from response import *
import time
import re
import json
from .reasoning_player import AgentPlayer
import asyncio

INQUIRY_COT= "Think carefully about your next step of strategy to be most likely to win. Let's think step by step, and finally make a decision." 
INQUIRY_PERSONA2 = "Don't forget your expert status, use your expertise to win this round!"
INQUIRY_PERSONA1 = "You are a game expert, good at predicting other people's behavior and deducing calculations, and using the most favorable strategy to win the game."
REFLECT_INQUIRY = "Review the previous round games, summarize the experience."
INQUIRY_PCOT= "First of all, predict the next round of choices based on the choices of other players in the previous round."  #需要提供全历史


class Bigagent(AgentPlayer):
    def __init__(self, name,persona, decision_model="gpt-4o-mini", summary_model="gpt-4o-mini", external_knowledge_api=None):
        super().__init__( name, persona, decision_model)
        """
        父类会执行：
            self.hp = 10
            self.biddings=[]
            self.cur_round = -1
            self.logs = None
            self.engine = decision_model  # 决策LLM模型版本
        """

        self.summary_llm = summary_model  # 总结LLM模型版本
        self.external_knowledge_api = external_knowledge_api
        self.history = []  # 仅记录last_step_result历史
        self.history_prompt = []
        self.background_rules = None  # 背景规则

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
        self.history.append(game_state) #! 这里调整了一些
        # 3. 外部知识调用（如果需要）
        external_knowledge = ""
        if self.external_knowledge_api:
            external_knowledge = await self.external_knowledge_api.query(
                #current_state=current_state,
                step_and_task=step_and_task
            )
        flat_history_text = "\n ".join([item for item in self.history])
        # 4. 构造LLM请求上下文
        
        prompt = self.construct_prompt(
            last_step_result=game_state,
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
    async def act(self):   #适配接口
        print(f"Player {self.name} conduct bidding")
        if not hasattr(self, "last_message_length"): # 由于这个game 用append message形式，因此在这里转换成MDP，无需更早轮次信息
            self.last_message_length = 0  
        # 获取新增消息
        new_messages = self.message[self.last_message_length:]
        
        await self.MDP_act(new_messages)
        # self.message需要记录llm response
        self.message.append({'role': 'assistant', 'content': self.llm_response})
        # 更新记录的消息长度
        self.last_message_length = len(self.message)


    @async_adapter
    async def parse_input(self, input_text): #! 已改为问答式，效果显著； 是否有可能混淆上一轮状态和本轮状态
        """
        使用总结LLM提取输入文本中的关键信息。
        """
        flat_input_text = "\n".join([f"{item['role']}: {item['content']}." for item in input_text])
        flat_input_text = "game record: \n"+ self.background_rules + flat_input_text
        sys_prompt1 = """ You are a game expert.The following is a game record. Please answer 4 questions, avoid unnecessary explanations, copying the original wording as much as possible."""
        sys_prompt2 = """ Questions:
                            1.What are all players' action and its result in last round? if not, answer None. Retain the narrator's perspective, specify the time using "after ROUND X" of description in game first. 
                            2.What are all player's specific status. Retain the narrator's perspective, specify the time using "before ROUND X" of description in game first.
                            3.What is specific status information related to this player. Retain the second-person perspective, specify the time using "before ROUND X" of description in game first.
                            4.In this round, what are the goals that this player needs to achieve, and the actions that can be chosen? Retain the second-person perspective, specify the time using "in ROUND X" of description in game first.
                            Output the result in given JSON format:
                            {
                            "Q1":"answer1",
                            "Q2":"answer2",
                            "Q3":"answer3",
                            "Q4":"answer4",
                            }
                        """ 

        messages = [
            {'role': 'system', 'content': sys_prompt1},
            {'role': 'system', 'content': flat_input_text}, 
            {'role': 'system', 'content': sys_prompt2},
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

    @async_adapter
    async def extract_background_rules(self, input_text):
        """
        提取背景规则, 适用于身份基本不变的game
        游戏规则: 与具体玩家无关的游戏基本信息，包括游戏的主题、设定和环境描述,游戏运行的核心机制和约束,明确玩家需要达成的目标和判定胜利的标准
        角色： 当前玩家在游戏中的角色设定和背景信息，当前玩家特定的任务或目标（如果有）
        """
        flat_input_text = "; ".join([f"role: {item['role']}, content: {item['content']}." for item in input_text])
        sys_prompt1 = """ You are a game expert.The following is a game record. Please answer some questions, avoid unnecessary explanations, copying the original wording as much as possible."""
        sys_prompt2 = """ Questions:
                        1.what is Background and basic rule about the game that is not related to specific players, include the theme and environment description of the game, the core mechanics and constraints of the game, and the objectives players need to achieve along with the criteria for determining victory. Ensure this section contains only general game information and excludes any player-specific details. Retain the narrator's perspective.
                        2.what is the character of this player: The role and background information of the {self.name} in the game. Ensure excludes any general game information. Retain the second-person perspective.
                        Output the result in given JSON format:
                        {
                        "Q1":"answer1",
                        "Q2":"answer2",
                        }
                        """
        messages = [
            {'role': 'system', 'content': sys_prompt1},
            {'role': 'user', 'content': flat_input_text},
            {'role': 'system', 'content': sys_prompt2},
        ]
        response = await self.call_llm(messages, model=self.summary_llm)
        return self.extract_json_from_text(response)

    def construct_prompt(self, last_step_result,  step_and_task, external_knowledge,inquiry=INQUIRY_COT):
        """
        构造给决策LLM的完整提示上下文
        """
        sys_prompt = f"""The following is last step results and current state of game: 
        {last_step_result}
        """
        ex_prompt=" "
        if external_knowledge:
            ex_prompt+= f"An game expert suggusts {external_knowledge}. \n"

        user_prompt = step_and_task+ex_prompt+inquiry
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'system', 'content': user_prompt},
        ]
        return messages
    def construct_rule(self, background_rules, persona):
        """
        构造给决策LLM的完整提示上下文
        """
        messages = [
            {'role': 'system', 'content': background_rules  },
            {'role': 'system', 'content': persona},
        ]
        return messages

    @async_adapter
    async def call_llm(self, prompt, model):
        """
        调用OpenAI API的通用函数
        """
        response = await openai_response(
            model=model,
            messages=prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return response

    @async_adapter
    async def parse_llm_output(self, llm_response):
        """
        将LLM的输出解析为可执行动作
        """
        action= await self.parse_result(llm_response)
        self.biddings.append(action)
        return action

    def extract_json_from_text(self, text):
        """
        从LLM返回文本中提取JSON格式的数据
        """
        try:
            json_start = text.index("{")
            json_end = text.rindex("}") + 1
            json_data = text[json_start:json_end]
            return json.loads(json_data)
        except (ValueError, json.JSONDecodeError):
            raise ValueError("未能从LLM响应中提取有效的JSON数据")
