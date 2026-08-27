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

TEMPERATURE = 0
API_BASE_URL = "https://openrouter.ai/api/v1"
RAG_TOP_K = 1
_client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
async_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
MAX_RETRIES = 5

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

def render_labeled_turns(messages):
    """Merge (main reply + RAG follow-up) into one Agent Turn; skip DB results."""
    turns, current_agent = [], []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = (m["content"] if isinstance(m, dict) else m.content) or ""
        if role == "system" or not content:
            continue
        if role == "assistant":
            current_agent.append(content)          # accumulate main + RAG follow-up
        elif content.startswith("[Database Result]"):
            continue                                # tool output ≠ anyone's speech
        else:                                       # genuine opponent reply
            if current_agent:
                turns.append(("agent", "\n".join(current_agent))); current_agent = []
            turns.append(("opponent", content))
    if current_agent:
        turns.append(("agent", "\n".join(current_agent)))

    out, a, o = [], 0, 0
    for kind, text in turns:
        if kind == "agent":
            a += 1; out.append(f"[Agent Turn {a}]: {text}")
        else:
            o += 1; out.append(f"[Opponent Turn {o}]: {text}")
    return "\n\n".join(out)

def chat(system, model, user, temperature=TEMPERATURE, client=_client,
         max_tokens=300):                                   
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
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
        time.sleep(wait)
    raise RuntimeError(f"chat function failed after {MAX_RETRIES} attempts: {last_err}")

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

def chat_json(model, user: str, max_tokens: int = 12288, client=_client,) -> dict:
    for _ in range(MAX_RETRIES):
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},   # in chat_json's call path only
            messages=[
                {"role": "system", "content": "\n你必须只输出一个合法的JSON对象，不要输出其他任何内容。"},
                {"role": "user", "content": user},
            ],
            extra_body={"reasoning": {"enabled": False}}
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[chat_json] JSON parse failed, raw tail: ...{raw[-200:]!r}")  # ← make it loud
            wait = 2 ** attempt
            time.sleep(2**attempt)
    return {}

async def judge_milestone_async(fact, model, milestone, agent_text, sentence):
    """Returns (verified: bool, judge_ok: bool). judge_ok=False => API failed, fail-open."""
    for attempt in range(MAX_RETRIES):
        try:
            async with asyncio.Semaphore(16):
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
                return bool(data.get("acknowledged", False)), True
            last_err = RuntimeError(f"empty choices: {getattr(resp, 'error', None)}")
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"Attempt bricked because of {last_err}, retrying in {wait}s, {attempt+1}/{MAX_RETRIES}")
        
            await asyncio.sleep(wait)
        
    print(f"[milestone-judge] failed after {MAX_RETRIES} attempts: {last_err}")
    return True, False        
class LawRetriever:
    """ retrieval over the law-article JSON file."""

    def __init__(self, json_path, device):

        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)

        with open(json_path, encoding="utf-8") as f:
            laws = json.load(f)
        self.laws = {value: key for key, value in laws.items()}

        self.keys = list(self.laws.keys())
        # index on "article name + article text" so searches by number also work
        corpus = [f"{k} {v}" for k, v in self.laws.items()]
        
        
        self.corpus_embeddings = self.model.encode(
            corpus,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    def group_search(self, query, top_k = RAG_TOP_K) -> list[tuple[str, str]]:

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

            best_res = tk.indices[0].item()

            res.append(self.corpus_embeddings[best_res])

        return res

    def format_results(self, results: list[tuple[str, str]]) -> str:
        if not results:
            return "（未检索到相关法条）"
        return "\n".join(f"【{k}】{v}" for k, v in results)

    def extract_search_query(self, text: str) -> str | None:
        """Extracts query from <search>query</search> tags."""
        match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
        if match:
            print("search was found")
            return match.group(1).strip()
        return None
