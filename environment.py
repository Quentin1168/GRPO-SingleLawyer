import ast
import asyncio
import re

import cn2an
import art
from sentence_transformers import SentenceTransformer, util
import openai
import utils
from transformers import AutoTokenizer
import json

# Parameters for various token limits and models
MAX_TURNS = 8
MAX_TOKENS_PER_TURN = 300
MAX_DB_TOKENS = 250
MAX_TOKENS_RAG_FOLLOWUP = 250
SIM_THRESHOLD = 0.70                      
OPPONENT_MODEL = "deepseek/deepseek-chat"
JUDGE_MODEL = "z-ai/glm-4.7-flash"
CONTEXT_LEN = 12288
MAX_NEW_TOKENS = 300
BUFFER = 64
MAX_INPUT = CONTEXT_LEN - MAX_NEW_TOKENS - BUFFER
MAX_RAG_INPUT = CONTEXT_LEN - MAX_TOKENS_RAG_FOLLOWUP - BUFFER

# RL Agent Prompt, which encourages using <search></search> containers to search the RAG database.
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

"""
Trajectory Helper class, represents a singular trajectory. 
Used for advancing and finishing given trajectory given the various iteractions as it progresses.

"""
class TrajectoryHelper():

    _tokeniser = None
    _rag_db = None
    # tokeniser and rag database searcher shared across trajectories for efficiency.
    @classmethod
    def _shared_tokeniser(cls):
        if cls._tokeniser is None:
            cls._tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")
        return cls._tokeniser

    @classmethod
    def _shared_rag_db(cls):
        if cls._rag_db is None:
            cls._rag_db = utils.LawRetriever("law.json", "cpu")
        return cls._rag_db

    """
    params:
    id - Trajectory ID
    fact - Case fact
    milestones - Milestones of the case (excluding the law)
    law - specific statue of the case 
    max_turns - maximum dialogue turns before case ends
    model - the agent model being used
    semantic_model - the semantic model used to encode text to detect milestones
    device - hardware used to run the code 
    """
    def __init__(self, id, fact, milestones, law, max_turns, model, semantic_model, device):
        self.device = device
        self.fact = fact
        self.id = id
        self.milestones = ast.literal_eval(milestones)
        self.law = law
        self.threshold = 0.7 # threshold for cosine similarity matching, should be low as it will also pass a llm judge.
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
        self.rag_db = self._shared_rag_db() # shared rag retriever
        self.tokeniser = self._shared_tokeniser() # shared tokeniser
        self.search_count = 0
        
    """

    """
    def sentence_split(self, text: str):
        split = re.split(r"[。！？\n；]", text)
        return [s.strip() for s in split if len(s.strip())> 4]
    
    """
    milestone_checker used to detect for case specific milestones in at the current turn's RL agent dialogue. 
    LLM judge milestone determination handled by trajectory_rollout.

    params: 
    text - current text to be processed
    achieved - milestones that already have been achieved so far
    """
    async def milestone_checker(self, text, achieved):
        sentences = [s for s in self.sentence_split(text) if s.strip()]
        if not sentences:
            return None  

        sentence_embeddings = await asyncio.to_thread(
            self.semantic_model.encode,
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarity_matrix = util.cos_sim(sentence_embeddings, self.process_embeddings)


        # check if detected milestones have already been achieved, otherwise return detected ones.
        for j in range(len(self.milestones)):
            if j in achieved:
                continue
            col = similarity_matrix[:, j]
            # must exceed threshold
            if col.max().item() >= self.threshold:
                return j, sentences[int(col.argmax())]
        return None

    """
    trajectory_rollout used to advance and finish the trajectory. 
    Uses Agent Reinforcement Trainer's trajectory object
    Detects milestones and law citations.
    Logs various metrics for wandb viewing
    
    """
    async def trajectory_rollout(self):
        api = self.model.openai_client()
        # reward recieved per milestone should be 1/(total milestones)
        milestone_reward = 1.0 / (len(self.milestones) + 1)
        # initialise trajectory
        trajectory = art.Trajectory(
            messages_and_choices=[
                {"role": "system", "content": RL_SYSTEM_PROMPT.format(fact=self.fact)}
            ],
            metadata={"case_id": self.id},
        reward=0.0,
        )
        milestone_found_turn = []
        law_found_turn = -1
        achieved = set()
        law_check = False
        # judge rejections are when the judge rejects the cosine detected milestone
        judged_rejections = 0
        # judge failures are when the judge failed to return a decision
        judge_failures = 0

        # iterate til max turns
        for turn in range(self.max_turns):
            n = len(self.tokeniser.apply_chat_template(trajectory.messages(), \
                    add_generation_prompt=True, tokenize=True))
            # detect if the current token count for the entire dialogue exceeds the maximum context length
            if n > MAX_INPUT:
                # stops dialogue and logs overflow
                trajectory.metrics["truncated_by_context"] = 1.0
                break
            try:
                # get reply from agent
                reply = await api.chat.completions.create(
                            model=self.model_name,
                            messages=trajectory.messages(),
                            max_completion_tokens=min(MAX_INPUT, MAX_TOKENS_PER_TURN),
                            temperature=0.0,
                            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        )
            except openai.BadRequestError as e:
                # detect if current token count for entire dialogue exceeds context length based on error
                if "context length" in str(e) or "input tokens" in str(e):
                    # keep dialogue so far if so
                    trajectory.metrics["truncated_by_context"] = 1.0
                    break                     
                # anything else is a real bug
                raise                         
            

            trainable = reply.choices[0]
            # add agent's response to the ongoing dialogue
            trajectory.messages_and_choices.append(trainable)
            agent_text = trainable.message.content or ""
            # detect if there agent wants to invoke a search 
            query = self.rag_db.extract_search_query(agent_text)
            turn_reward = 0.0
            if query:
                # if so, search based on agent's request within <search>
                # since this is done during other trajectory rollouts, group them together
                retrieved = await asyncio.to_thread(self.rag_db.group_search, [query])
                self.search_count += 1
                # format the retrieved result
                db_text = self.rag_db.format_results(retrieved)
                tok_ids = self.tokeniser.encode(db_text, add_special_tokens = False)
                # if the retrieved text is greater than the specified max db tokens, needs to be truncated
                if len(tok_ids) > MAX_DB_TOKENS:
                    db_text = self.tokeniser.decode(tok_ids[:MAX_DB_TOKENS])
                # append the database result to the dialogue
                trajectory.messages_and_choices.append(          
                    {"role": "user", "content": f"[Database Result]:\n {db_text}"}
                )
                n2 = len(self.tokeniser.apply_chat_template(trajectory.messages(), \
                    add_generation_prompt=True, tokenize=True))
                # also need to check if the dialogue + database result exceeds context length
                if n2 > MAX_RAG_INPUT:
                    trajectory.metrics["truncated_by_context"] = 1.0
                    break
                try:
                    # get subsequent agent input based on the RAG result. 
                    query_reply = await api.chat.completions.create(
                                        model=self.model_name,
                                        messages=trajectory.messages(),
                                        max_completion_tokens=min(MAX_INPUT, MAX_TOKENS_RAG_FOLLOWUP),
                                        temperature=0.0,
                                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                    )
                except openai.BadRequestError as e:
                    # detect if current token count for entire dialogue exceeds context length based on error
                    if "context length" in str(e) or "input tokens" in str(e):
                        # keep dialogue so far if so
                        trajectory.metrics["truncated_by_context"] = 1.0
                        break                    
                    # anything else is a real bug
                    raise  
                # add agent input to the dialogue
                query_trainable = query_reply.choices[0]
                
                trajectory.messages_and_choices.append(query_trainable)

                agent_text += "\n" + (query_trainable.message.content or "")
                # determine if law has been detected if have not already
                if law_check != True:
                    # translate chinese numbers to arabic for detection
                    translated_text = cn2an.transform(agent_text)
                    law_numbers = re.findall(r'\d+', self.law)
                    # assign milestone reward and note law has been checked 
                    if law_numbers and law_numbers[0] in translated_text:
                        turn_reward += milestone_reward
                        law_check = True
                        law_found_turn = turn

            
            # milestone checks are limited to one to prevent turns from shrinking
            # get the 1 checked milestone and feed it to the judge
            candidate = await self.milestone_checker(agent_text, achieved)
            if candidate is not None:
                idx, sentence = candidate
                # since this is done during other trajectory rollouts, group them together
                verified, judge_ok = await utils.judge_milestone_async(
                    self.fact,  JUDGE_MODEL, self.milestones[idx], agent_text, sentence)
                # detect judge failing reply
                if not judge_ok:
                    judge_failures += 1
                # if milestone is OK by the judge, log the milestone and turn it was detected
                if verified:
                    milestone_found_turn.append((turn, idx))
                    # add to milestone already found
                    achieved.add(idx)
                    # recieve milestone reward
                    turn_reward += milestone_reward
                else:
                    judged_rejections += 1

            trajectory.metrics[f"turn_reward_{turn}"] = turn_reward

            # check if the debate is considered complete (all milkestones done)
            if len(achieved) == len(self.milestones) and law_check:
                break

            # get static lawyer presonse
            env_lawyer = await utils.async_chat(
                system="你是本案的被告律师",
                model = OPPONENT_MODEL, user=trajectory.messages(), temperature=0.0
            )

            trajectory.messages_and_choices.append({"role": "user", "content": env_lawyer})

        # log metrics
        trajectory.metadata["milestone_found_turn"] = json.dumps(milestone_found_turn)
        trajectory.metrics["law_found_turn"] = float(law_found_turn)
        trajectory.metrics["milestones_achieved"] = float(len(achieved))
        trajectory.metrics["milestone_frac"] = len(achieved) / max(len(self.milestones), 1)
        trajectory.metrics["law_found"]= float(law_check)
        trajectory.metrics["milestone_rejections"] = float(judged_rejections)
        trajectory.metrics["milestone_judge_failures"] = float(judge_failures)
        trajectory.metrics["searches"] = float(self.search_count)

        return trajectory




                        
