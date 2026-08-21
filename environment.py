import ast
import asyncio
import re

import cn2an
import art
from sentence_transformers import SentenceTransformer, util
import openai
import utils
from transformers import AutoTokenizer

MAX_TURNS = 10
MAX_TOKENS_PER_TURN = 512
MAX_TOKENS_RAG_FOLLOWUP = 250
SIM_THRESHOLD = 0.70                      
OPPONENT_MODEL = "deepseek/deepseek-chat"



RL_SYSTEM_PROMPT = ( 
    "你是本案的原告律师，请根据以下事实进行控方辩论：\n{fact}\n\n【工具使用说明】\n"
    "你在辩论过程中可以使用检索工具查询相关法律法规或案事实细节。\n"
    "当你需要进行检索时，请在回答中使用 `<search>案情特征或法律争议点描述</search>` 格式。\n"
    "注意：检索时请提交与案情事实、行为性质或争议焦点相关的上下文描述（而非直接搜索具体法条名称），"
    "以便系统为你匹配最相关的法律依据。\n"
    "例如：`<search>未经同意秘密转移他人财物 盗窃罪认定与量刑标准</search>` 或 "
    "`<search>合同到期拒绝履行还款义务 违约金与利息计算</search>`。\n"
    "系统会自动拦截你的检索请求并为你提供相关法条/证据，之后你可以继续进行辩论。\n\n"
    "目前的辩论进展如下："
)

class TrajectoryHelper():

    def __init__(self, id, fact, milestones, law, size, max_turns, model, semantic_model, device):
        self.device = device
        self.size = size
        self.fact = fact
        self.id = id
        self.milestones = ast.literal_eval(milestones)
        self.law = law
        self.threshold = 0.90
        self.max_turns = max_turns
        self.model = model
        self.semantic_model = semantic_model
        self.process_embeddings = self.semantic_model.encode(
                self.milestones, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
        )
        self.model_name = self.model.get_inference_name() 
        self.rag_db = utils.LawRetriever("law.json", "cpu")

        

    def sentence_split(self, text: str):
        split = re.split(r"[。！？\n；]", text)
        return [s.strip() for s in split if len(s.strip())> 4]
    
    async def milestone_checker(self, text, achieved):
        sentences = self.sentence_split(text)

        sentence_embeddings = await asyncio.to_thread(
            self.semantic_model.encode,
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarity_matrix = util.cos_sim(sentence_embeddings, self.process_embeddings)


        
        return {
            j for j in range(len(self.milestones))
            if j not in achieved and similarity_matrix[:, j].max().item() >= self.threshold
        }

    async def trajectory_rollout(self):
        api = self.model.openai_client()
        milestone_reward = 1.0 / (len(self.milestones) + 1)

        trajectory = art.Trajectory(
            messages_and_choices=[
                {"role": "system", "content": RL_SYSTEM_PROMPT.format(fact=self.fact)}
            ],
            metadata={"case_id": self.id},
        reward=0.0,
        )

        achieved = set()
        law_check = False

        for turn in range(self.max_turns):
            try:
                reply = await api.chat.completions.create(
                            model=self.model_name,
                            messages=trajectory.messages(),
                            max_completion_tokens=MAX_TOKENS_PER_TURN,
                            temperature=0.85,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
            except openai.BadRequestError as e:
                if "context length" in str(e) or "input tokens" in str(e):
                    trajectory.metrics["truncated_by_context"] = 1.0
                    break                     # episode over; keep what we have
                raise                         # anything else is a real bug
            

            trainable = reply.choices[0]

            trajectory.messages_and_choices.append(trainable)
            agent_text = trainable.message.content or ""

            query = self.rag_db.extract_search_query(agent_text)
            turn_reward = 0.0
            if query:
                retrieved = await asyncio.to_thread(self.rag_db.group_search, [query])
                trajectory.messages_and_choices.append(          
                    {"role": "user", "content": f"[Database Result]:\n {retrieved[0]}"}
                )
                try:
                    query_reply = await api.chat.completions.create(
                                        model=self.model_name,
                                        messages=trajectory.messages(),
                                        max_completion_tokens=MAX_TOKENS_PER_TURN,
                                        temperature=0.85,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                    )
                except openai.BadRequestError as e:
                    if "context length" in str(e) or "input tokens" in str(e):
                        trajectory.metrics["truncated_by_context"] = 1.0
                        break                     # episode over; keep what we have
                    raise  
                
                query_trainable = query_reply.choices[0]

                trajectory.messages_and_choices.append(query_trainable)

                agent_text += "\n" + (query_trainable.message.content or "")

                translated_text = cn2an.transform(agent_text)
                print("CHECKING LAW:")
                law_numbers = re.findall(r'\d+', self.law)
                if law_numbers and law_numbers[0] in translated_text:
                    print("LAW FOUND")
                    turn_reward += milestone_reward
                    law_check = True
            

            milestone_check = await self.milestone_checker(agent_text, achieved)


            for i in milestone_check:
                achieved.add(i)
                turn_reward += milestone_reward

            trajectory.metrics[f"turn_reward_{turn}"] = turn_reward

            if len(achieved) == len(self.milestones) and law_check:
                break

            env_lawyer = await asyncio.to_thread(
                utils.chat, system="你是本案的被告律师",
                model = OPPONENT_MODEL, user=trajectory.messages(), temperature=0.0
            )

            trajectory.messages_and_choices.append({"role": "user", "content": env_lawyer})

        trajectory.metrics["milestones_achieved"] = float(len(achieved))
        trajectory.metrics["law_found"]= float(law_check)

        return trajectory




                        