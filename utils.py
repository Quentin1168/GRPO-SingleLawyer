import json
from openai import OpenAI, RateLimitError, AsyncOpenAI
import jieba
from sentence_transformers import SentenceTransformer, util
import re
import torch
import time, random
from dotenv import load_dotenv
import os
import asyncio
# set up some hyperparamters
TEMPERATURE = 0
API_BASE_URL = "https://openrouter.ai/api/v1"
RAG_TOP_K = 1
_client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
async_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
MAX_RETRIES = 5

# milestone judge prompt
MILESTONE_JUDGE_SYSTEM = """你是一名严格的法律辩论评审。你的任务是判断律师是否【真正论证了】某个关键点，还是仅仅【机械复述/罗列】了它。

判定为 acknowledged=true 的标准（须同时满足）：
1. 律师用自己的论证语言表达了该关键点的实质内容（而非原样照抄关键点文字或法条检索结果）；
2. 该关键点被结合到具体案情事实或法律推理中（说明了它为什么成立、或它如何支持我方主张）。

判定为 acknowledged=false 的情形（任一即可）：
- 仅仅原文复读、同义改写关键点，未连接任何案情事实或推理；
- 仅在罗列/堆砌中顺带提及，无论证作用；
- 内容与案情事实明显不符（凭空断言）。

只输出严格 JSON：{"acknowledged": true/false, "reason": "一句话理由"}"""

MILESTONE_JUDGE_USER = """【案情事实】
{fact}

【待验证的关键点】
{milestone}

【律师本轮发言全文】
{agent_text}

【触发检测的句子】
{sentence}

请判断该律师是否真正论证了上述关键点。只输出 JSON。"""
"""
render_labeled_turns merges the main reply and the rag follow up reply together and skips the RAG query results. 
It also relabels the agent/opponent with the same example headers used by the LLM judge.
This is used to make the prompt so far for the agent in question.

parameters:
- messages: messages dict to be merged
returns:
- the merged dict as a string
"""
def render_labeled_turns(messages):
    turns, current_agent = [], []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = (m["content"] if isinstance(m, dict) else m.content) or ""
        if role == "system" or not content:
            continue
        if role == "assistant":
            # for the agent merge main + RAG follow-up
            current_agent.append(content)
        # skip the RAG output
        elif content.startswith("[Database Result]"):
            continue                                
        else:                                     
            if current_agent:
                turns.append(("agent", "\n".join(current_agent))); current_agent = []
            turns.append(("opponent", content))
    
    if current_agent:
        turns.append(("agent", "\n".join(current_agent)))

    out, a, o = [], 0, 0
    # append the label depending on the role
    for kind, text in turns:
        if kind == "agent":
            a += 1; out.append(f"[Agent Turn {a}]: {text}")
        else:
            o += 1; out.append(f"[Opponent Turn {o}]: {text}")
    return "\n\n".join(out)

"""
chat - general use chat function, mainly for the RL agent model and the static model

parameters:
system - the system prompt
model - the model being prompted
user - the prompt so far
temperature - temperature of the chat
client - the API client (should be OpenAI)

"""
def chat(system, model, user, temperature=TEMPERATURE, client=_client,
         max_tokens=300):                                   
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                # use rendered_labeled_turns to reformat the dialogue.
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": render_labeled_turns(user)}],
            )

            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            err = getattr(resp, "error", None)
            last_err = RuntimeError(f"empty choices from API: {err}")
        except Exception as e:
            last_err = e
        # increase the timeout by 2x every time
        wait = 2 ** attempt
        print(f"Attempt bricked because of {last_err}, retrying in {wait}s, {attempt+1}/{MAX_RETRIES}")
        time.sleep(wait)
    raise RuntimeError(f"chat function failed after {MAX_RETRIES} attempts: {last_err}")

"""
async_chat - asynchronous chat function, used for concurrent trajectory rollouts.

parameters:
system - the system prompt
model - the model being prompted
user - the prompt so far
temperature - temperature of the chat
client - the API client (should be OpenAI)

"""
async def async_chat(system, model, user, temperature=TEMPERATURE, max_tokens =300):
    for attempt in range(MAX_RETRIES):
        try:
            async with asyncio.Semaphore(16):   
                resp = await async_client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                            {"role": "user", "content": render_labeled_turns(user)}],
                    )
                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
                err = getattr(resp, "error", None)
                last_err = RuntimeError(f"empty choices from API: {err}")
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"Attempt bricked because of {last_err}, retrying in {wait}s, {attempt+1}/{MAX_RETRIES}")
            
            await asyncio.sleep(2** attempt)

    raise RuntimeError(f"chat function failed after {MAX_RETRIES} attempts: {last_err}")

"""
chat_json - json function, used for the judge LLM calls.

parameters:
model - the model being prompted
user - the prompt so far
max_tokens - max tokens for the call
client - the API client (should be OpenAI)


"""
def chat_json(model, user, max_tokens = 12288, client=_client):
    for _ in range(MAX_RETRIES):
         # force a json format
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  
            messages=[
                {"role": "system", "content": "\n你必须只输出一个合法的JSON对象，不要输出其他任何内容。"},
                {"role": "user", "content": user},
            ],
            extra_body={"reasoning": {"enabled": False}}
        )
        raw = resp.choices[0].message.content or ""
        # clean the json output
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[chat_json] JSON parse failed, raw tail: ...{raw[-200:]!r}")  # ← make it loud
            wait = 2 ** attempt
            time.sleep(2**attempt)
    return {}

"""
judge_milestone_async - asynchronous LLM judge milestone checker.
Acts as the second check for milestones after cosine similarity.
Determines if the agent's use of the milestone is genuine use or mere parroting.

parameters:
fact - the case facts
model - model being used currently
agent_text - the particular RL agent output being tested
sentence - the particular milestone being tested

returns:
(verified, judge_ok)
verified - whether the judge acknowledged the milestone being genuine
judge_ok - whether the judge failed replying
"""
async def judge_milestone_async(fact, model, milestone, agent_text, sentence):
    """Returns (verified: bool, judge_ok: bool). judge_ok=False => API failed, fail-open."""
    for attempt in range(MAX_RETRIES):
        try:
            async with asyncio.Semaphore(16):
                # call the judge with the formatted prompt
                resp = await async_client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": MILESTONE_JUDGE_SYSTEM},
                        {"role": "user", "content": MILESTONE_JUDGE_USER.format(
                            fact=fact, milestone=milestone,
                            agent_text=agent_text, sentence=sentence)},
                    ],
                    extra_body={"reasoning": {"enabled": False}}, 
                )
            if resp.choices and resp.choices[0].message.content:
                data = json.loads(resp.choices[0].message.content)
                # if the judge failed to even return an "acknowledged", return false, otherwise check the acknowledged json
                return bool(data.get("acknowledged", False)), True
            last_err = RuntimeError(f"empty choices: {getattr(resp, 'error', None)}")
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"Attempt bricked because of {last_err}, retrying in {wait}s, {attempt+1}/{MAX_RETRIES}")
        
            await asyncio.sleep(wait)
    # output an answer even though the judge failed
    print(f"[milestone-judge] failed after {MAX_RETRIES} attempts: {last_err}")
    return True, False        

"""
LawRetriever class, used to search for laws based based on the query using cosine similarity

"""
class LawRetriever:
    """
    parameters:
    json_path - path to the law json file
    device - device used
    """

    def __init__(self, json_path, device):

        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)

        with open(json_path, encoding="utf-8") as f:
            laws = json.load(f)
        self.laws = {value: key for key, value in laws.items()}

        self.keys = list(self.laws.keys())
        # index on article name as well as article text so searches by number also work
        corpus = [f"{k} {v}" for k, v in self.laws.items()]
        
        
        self.corpus_embeddings = self.model.encode(
            corpus,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    """
    group_search - searches top k laws based on the query text

    parameters:
    query - the query text 
    top_k - top k laws needded

    returns:
    res - the top k laws in (law name, law description) format

    """
    def group_search(self, query, top_k = RAG_TOP_K):

        if not query:
            return []

        query_embeddings = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarity_matrix = util.cos_sim(query_embeddings, self.corpus_embeddings)

        res = []

        for i in range(len(query)):

            tk = torch.topk(similarity_matrix[i], k=top_k)

            for idx in tk.indices.tolist():
                name = self.keys[idx]
                res.append((name, self.laws[name]))    
        return res

    def format_results(self, results: list[tuple[str, str]]):
        if not results:
            return "（未检索到相关法条）"
        return "\n".join(f"{v} {k}" for k, v in results)


    """
    extract_search_query Extracts query from <search>query</search> tags.
    """
    def extract_search_query(self, text: str):
        match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
        if match:
            print("search was found")
            return match.group(1).strip()
        return None


