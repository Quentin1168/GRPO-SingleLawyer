#!/usr/bin/env python

import json
from openai import OpenAI
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEMPERATURE = 0
API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = "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"
MEMORY_TOP_K = 10
RAG_TOP_K = 5
_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def chat(system: str, model: str, user: str, temperature: float = TEMPERATURE) -> str:
    resp = _client.chat.completions.create(
        model = model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def chat_json(system: str, model, user: str) -> dict:
    """Chat call that must return a JSON object; retries once on parse failure."""
    for _ in range(3):
        raw = chat(system + "\n你必须只输出一个合法的JSON对象，不要输出其他任何内容。",
                   model, user, temperature=0.2)
        try:
            # strip markdown fences if present
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return {}

class LawRetriever:
    """TF-IDF retrieval over the law-article JSON file."""

    def __init__(self, json_path: str):
        with open(json_path, encoding="utf-8") as f:
            self.laws: dict[str, str] = json.load(f)

        self.keys = list(self.laws.keys())
        # index on "article name + article text" so searches by number also work
        corpus = [f"{k} {v}" for k, v in self.laws.items()]
        self.vectorizer = TfidfVectorizer(tokenizer=lambda t: jieba.lcut(t),
                                          token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[tuple[str, str]]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = scores.argsort()[::-1][:top_k]
        return [(self.keys[i], self.laws[self.keys[i]]) for i in idx if scores[i] > 0]

    @staticmethod
    def format_results(results: list[tuple[str, str]]) -> str:
        if not results:
            return "（未检索到相关法条）"
        return "\n".join(f"【{k}】{v}" for k, v in results)

class MemoryStore:
    """Stores per-agent reflections; injects the most recent K into prompts.

    This is the 'contextual learning' mechanism: no weights are updated,
    the agent simply conditions on distilled past experience.
    """

    def __init__(self):
        self._memories: list[str] = []

    def add(self, reflection: str):
        self._memories.append(reflection)

    def recall(self, k: int = MEMORY_TOP_K) -> str:
        if not self._memories:
            return "（暂无以往经验）"
        return "\n---\n".join(self._memories[-k:])


