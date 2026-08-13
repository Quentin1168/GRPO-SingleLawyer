import json
from openai import OpenAI
import jieba
from sentence_transformers import SentenceTransformer, util
import re
import torch

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
    """ retrieval over the law-article JSON file."""

    def __init__(self, json_path):

        self.model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="gpu")

        with open(json_path, encoding="utf-8") as f:
            self.laws = json.load(f)

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

        similarity_matrix = util.cosine_similarity(query_embeddings, self.corpus_embeddings)

        res = []

        for i in range(len(query)):

            topk = torch.top_k(similarity_matrix[i], k=top_k)

            best_res = topk.indices[0].item()

            res.append(self.corpus[best_res])

        return res

    def format_results(results: list[tuple[str, str]]) -> str:
        if not results:
            return "（未检索到相关法条）"
        return "\n".join(f"【{k}】{v}" for k, v in results)

    def extract_search_query(text: str) -> str | None:
        """Extracts query from <search>query</search> tags."""
        match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None