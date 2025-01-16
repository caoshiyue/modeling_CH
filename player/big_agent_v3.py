from response import *
import time
import re
import json
from .reasoning_player import AgentPlayer
import asyncio
from .memory import Memory

INQUIRY_COT= "Think carefully about your next step of strategy to be most likely to win. Let's think step by step, and finally make a decision." 
REFLECT_INQUIRY = "Review the previous round games, summarize the experience."

player_names = ["Alex", "Bob", "Cindy", "David", "Eric"]

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
        self.memory=Memory("game_memory.json",summary_model)
        self.opponent_dict = {}

    @async_adapter
    async def MDP_act(self, input_text):
        """
        主流程：解析输入、构造提示、调用决策LLM、返回动作
        """
        # 如果是首次初始化背景规则，则提取背景规则
        if self.background_rules is None:
            extract_rules = await self.extract_background_rules(input_text)
            self.background_rules = extract_rules["Q1"] +' The names of five players are Alex, Bob, Cindy, David and Eric.'
            self.persona = extract_rules["Q2"]
            self.history_prompt = self.construct_rule(background_rules=self.background_rules,persona = self.persona)
            self.memory.set_background(self.background_rules)
            print(extract_rules)

        # 1. 使用总结LLM提取关键信息
        parsed_input = await self.parse_input(input_text)

        # 提取输入信息的各部分
        #current_state = parsed_input["current_state"]

        game_state = parsed_input["Q1"]
        agent_state = parsed_input["Q2"]
        player_state = parsed_input["Q3"]
        player_task = parsed_input["Q4"]

        step_and_task= player_state+player_task
        # 2. 历史记录更新
        memory_query=None
        if game_state!="None" and game_state!=None:
            game_state+=agent_state
            self.history.append(game_state) #! 这里调整了一些
        else:
            # game_state="This is first round, no previous round results. "
            # game_state+=agent_state
            game_state=" "

        
        # 3. 外部知识调用（如果需要）
        external_knowledge = ""
        memory_query = await self.extract_specific_state(input_text)
        if memory_query!=None:
            await self.process_memory_query(memory_query, player_task,False)
            new_dict = {key: str(value[1])+'\n' for key, value in self.opponent_dict.items()}
            external_knowledge=str(new_dict)

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
                            1.Summarize the action of players and the winner in last round? if not, answer None. Retain the narrator's perspective, specify the time using "After ROUND X" of description in game first. 
                            2.What are all player's specific status. Retain the narrator's perspective, specify the time using "Before ROUND X" of description in game first.
                            3.What is specific status information related to this player. Retain the second-person perspective, specify the time using "Before ROUND X" of description in game first.
                            4.In this round, what the players need to do? Retain the second-person perspective using "Please + instructions", specify the time using "In ROUND X" of description in game first.
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
        sys_prompt1 = """ You are a game expert. The following is a game record. Please answer some questions, avoid unnecessary explanations, copying the original wording as much as possible."""
        sys_prompt2 = """ Questions:
                        1.what is background and basic rule about the game that is not related to specific players. Retain the narrator's perspective.
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


    @async_adapter
    async def extract_specific_state(self, input_text): #!此处记录的是当前状态，在此游戏中 = 上一轮number+上一轮对手动作
        """
        使用总结LLM提取输入文本中的关键信息。
        """
        flat_input_text = "\n".join([f"{item['role']}: {item['content']}." for item in input_text])
        flat_input_text = "game record: \n"+ self.background_rules  + flat_input_text
        sys_prompt1 = """ You are a game expert.The following is a game record. Please answer following questions, avoid unnecessary explanations, if no answer, answer None."""
        sys_prompt2 = """ Questions:
                            1. What is the value of 0.8 times the average last round? If it is first round, answer "the target value is unknow", else answer like "the target value is X".
                            2. What are the action of other players (Not this player) in last round? If it is first round, answer "", else answer like "the player choose X". 
                            Output the result in given JSON format:
                            {
                            "Q1":"answer1",
                            "player_name1":"action1",
                            "player_name2":"action2",
                            ...
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
    async def process_memory_query(self,js_dict, step_and_task,fast_infer=False):
        #* 首先，根据当前状态反思
        if fast_infer==False:
            for player, value in self.opponent_dict.items():
                idx=value[0]
                re_reason =  await self.memory.reflect(idx,js_dict[player])  # 动作是动作
                if re_reason:
                    res_action = await self.re_simulate( self.memory.get_action(idx)[1],self.memory.get_action(idx)[0],js_dict[player])
                    self.memory.update_action(idx, res_action)

        #* 然后，对新state query
        q1_value = js_dict['Q1']
        opponent_dict={}
        async def process_player(player, js_dict, q1_value, opponent_dict, step_and_task,fast_infer):
            if player in js_dict:
                q = q1_value + ', ' + js_dict[player] # 动作是状态
                idx, action=await self.memory.query(q)  
                if action!=None:
                    opponent_dict[player]=(idx, action[0])
                else:
                    if fast_infer==False:
                        res_action = await self.simulate( q1_value,js_dict[player],step_and_task,player)
                        idx=self.memory.add_record(q, res_action)
                        opponent_dict[player]=(idx,res_action[0])
        if fast_infer:
            await asyncio.gather(*( process_player(player, js_dict, q1_value, opponent_dict, step_and_task,fast_infer)for player in player_names))
        else:
            for player in player_names:#! 学习过程中不能聚合
                process_player(player, js_dict, q1_value, opponent_dict, step_and_task,fast_infer)
        self.opponent_dict=opponent_dict


    def construct_prompt(self, last_step_result,  step_and_task, external_knowledge,inquiry=INQUIRY_COT):
        """
        构造给决策LLM的完整提示上下文
        """
        #sys_prompt = f"""The following is last step results and current state of game: 
        if isinstance(last_step_result,list ):
            messages = [
                {'role': 'system', 'content': "The following is game record: "},
            ]
            for prompt in last_step_result:
                messages.append({'role': 'user', 'content': prompt})
        else:
            messages =[
                {'role': 'system', 'content': f"The following is game record: {last_step_result}"},
            ]
        ex_prompt=" "
        if external_knowledge:
            ex_prompt+= f"An game expert predict other players' strategies are: {external_knowledge}. \n"

        user_prompt = f"OK, {self.name}! "+ step_and_task+ex_prompt+inquiry
        messages += [
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
    async def simulate(self, game_state, last_action,step_and_task,player_name):
        sys_prompt1 = "You are a game expert involved in a survive challenge." +self.background_rules
        inquiry =f"Let's consider briefly about this PLAYER's belief and strategy and make a prediction of possible actions this PLAYER may choose at the end of reasoning. Answer briefly. DO NOT mention player's name."
        messages = [
            {'role': 'system', 'content': sys_prompt1},
            {'role': 'system', 'content': f"In last round, one PLAYER's action is: {last_action}, and {game_state}. "},
            {'role': 'system', 'content': f"OK, now all players are required to make a decision: {step_and_task}."},
            {'role': 'system', 'content': inquiry},
        ]

        n_retry=0
        pattern = r"PREDICT (-?\d+)"
        while n_retry<10:
            try:
                llm_response = await self.call_llm(messages, model=self.engine)
                #match = re.search(pattern, llm_response)
                #llm_response  = re.sub(pattern, '', llm_response)
                #action = int(match.group(1))
                n_retry=10
            except: #再试一次
                n_retry+=1
    
        return (llm_response,messages)
    
    @async_adapter
    async def re_simulate(self, messages, last_reasoning, new_last_action):

        inquiry =f"""Now, analyze this action to see if they reveal any new strategies or beliefs, 
        think about why your reasoning is flawed, rather than simply remembering the action.
        Based on the above analysis, update your chain of reasoning to make it more general and flexible for similar situations in the future
        """
        new_messages = [
            {'role': 'assistant', 'content': last_reasoning},
            {'role': 'system', 'content': f"But actually, this PLAYER's action is: {new_last_action}"},
            {'role': 'system', 'content': inquiry},
        ]
        messages+=new_messages
        n_retry=0
        while n_retry<10:
            try:
                llm_response = await self.call_llm(messages, model=self.engine)
                n_retry=10
            except: #再试一次
                n_retry+=1
        new_messages=[
                    {'role': 'assistant', 'content': llm_response},
                    {'role': 'system', 'content': "Now you encountering similar situation again: "},
                    messages[1],
                    {'role': 'system', 'content': "Let's consider about this PLAYER's belief and strategy and make a prediction of possible actions this PLAYER may choose at the end of reasoning. Answer briefly. DO NOT mention player's name."},
                ]
        messages+=new_messages
        n_retry=0
        while n_retry<10:
            try:
                llm_response = await self.call_llm(messages, model=self.engine)
                n_retry=10
            except: #再试一次
                n_retry+=1

        return (llm_response,messages)


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
    async def parse_llm_output(self, llm_response,record=True):
        """
        将LLM的输出解析为可执行动作
        """
        action= await self.parse_result(llm_response)
        if record:
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
