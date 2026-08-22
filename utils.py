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
MEMORY_TOP_K = 10
RAG_TOP_K = 3
_client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
async_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
MAX_RETRIES = 5

def render_transcript(messages):
    parts = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content
        if role == "system" or not content:
            continue
        speaker = "原告律师" if role == "assistant" else "被告律师"
        parts.append(f"{speaker}：{content}")
    return "\n\n".join(parts)

def chat(system, model, user, temperature=TEMPERATURE, client=_client,
         max_tokens=300):                                   
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": render_transcript(user)}],
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
                            {"role": "user", "content": render_transcript(user)}],
                    )
                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
                err = getattr(resp, "error", None)
                last_err = RuntimeError(f"empty choices from API: {err}")
        except Exception as e:
            last_err = e
        print(f"Attempt bricked because of {last_err}, retrying in {wait}s, {attempt+1}/{MAX_RETRIES}")
        await asyncio.sleep(2** attempt)

    raise RuntimeError(f"chat function failed after {MAX_RETRIES} attempts: {last_err}")

def chat_json(model, user: str, max_tokens: int = 4096, client=_client,) -> dict:
    for _ in range(3):
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},   # in chat_json's call path only
            messages=[
                {"role": "system", "content": "\n你必须只输出一个合法的JSON对象，不要输出其他任何内容。"},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content or ""
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"[chat_json] JSON parse failed, raw tail: ...{raw[-200:]!r}")  # ← make it loud
            continue
    return {}
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
