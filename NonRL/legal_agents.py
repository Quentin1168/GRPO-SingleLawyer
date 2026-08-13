#!/usr/bin/env python

from llm import chat, chat_json, MemoryStore, LawRetriever

MODEL_1 = "qwen/qwen3.5-flash-02-23"
MODEL_2 = "google/gemini-3.1-flash-lite"

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.memory = MemoryStore()


# --------------------------------------------------------------------------- #
#  Lawyers                                                                     #
# --------------------------------------------------------------------------- #
class LawyerAgent(BaseAgent):
    """Shared logic for plaintiff (公诉人) and defendant (辩护人)."""

    STANCE = {
        "plaintiff": "你是公诉人（检察官），你的职责是依据事实和法律指控被告人，论证其罪行成立并主张相应刑罚。",
        "defendant": "你是被告人的辩护律师，你的职责是依据事实和法律为被告人辩护，争取无罪、罪轻或从宽处理。",
    }

    def __init__(self, role: str, retriever: LawRetriever):
        assert role in ("plaintiff", "defendant")
        super().__init__(name={"plaintiff": "公诉人", "defendant": "辩护人"}[role],
                         role=role)
        self.retriever = retriever
        self.strategy = ""          # per-case, set in prepare()
        self.cited_laws = ""        # per-case retrieved laws

    # ---------- Preparation phase: one-time RAG + strategy ---------- #
    def prepare(self, case_facts: str):
        system = (f"{self.STANCE[self.role]}\n\n"
                  f"以下是你以往办案的经验总结，请吸取教训：\n{self.memory.recall()}")

        # 1) formulate a single search query for the law database
        q = chat_json(
            system,
            MODEL_1,
            f"案件事实如下：\n{case_facts}\n\n"
            "庭审前你有一次检索法律条文数据库的机会。"
            '请输出JSON：{"search_query": "用于检索相关法条的查询语句"}',
        )
        query = q.get("search_query", case_facts[:300])
        results = self.retriever.search(query)
        self.cited_laws = LawRetriever.format_results(results)

        # 2) build a strategy from facts + retrieved laws
        self.strategy = chat(
            system,
            MODEL_1,
            f"案件事实：\n{case_facts}\n\n检索到的法条：\n{self.cited_laws}\n\n"
            "请制定你的庭审策略：选出你认为适用的法条，并列出你计划提出的核心论点（简明扼要）。",
        )

    # ---------- Debate round: optional RAG + argument ---------- #
    def argue(self, case_facts: str, dialogue: str, round_no: int) -> str:
        system = (f"{self.STANCE[self.role]}\n\n"
                  f"你的庭审策略：\n{self.strategy}\n\n"
                  f"你已掌握的法条：\n{self.cited_laws}\n\n"
                  f"以往经验：\n{self.memory.recall()}")

        # optional mid-trial RAG
        """q = chat_json(
            system,
            MODEL_1,
            f"案件事实：\n{case_facts}\n\n目前庭审记录：\n{dialogue or '（庭审刚开始）'}\n\n"
            "本轮发言前，你可以再次检索法条数据库。"
            '如需检索请输出 {"search_query": "..."}, 否则输出 {"search_query": null}',
        )
        if q.get("search_query"):
            extra = self.retriever.search(q["search_query"])
            self.cited_laws += "\n" + LawRetriever.format_results(extra)
        """
        return chat(
            system,
            MODEL_1,
            f"案件事实：\n{case_facts}\n\n目前庭审记录：\n{dialogue or '（庭审刚开始）'}\n\n"
            f"现在是第{round_no}轮辩论，请发表你的意见。要求：回应对方上一轮的观点（如有），"
            "结合具体案件事实，明确引用法条。发言控制在200字以内。",
        )


# --------------------------------------------------------------------------- #
#  Judge                                                                       #
# --------------------------------------------------------------------------- #
class JudgeAgent(BaseAgent):
    SYSTEM = "你是一名中华人民共和国刑事法庭的法官，你必须依据双方的辩论意见、案件事实和法律作出公正裁判。"

    def __init__(self):
        super().__init__(name="法官", role="judge")

    def _system(self) -> str:
        return f"{self.SYSTEM}\n\n以下是你以往审判的经验总结：\n{self.memory.recall()}"

    def comment(self, case_facts: str, dialogue: str, round_no: int,
                max_rounds: int) -> dict:
        """Remark on the round and decide whether to end the debate."""
        out = chat_json(
            self._system(),
            MODEL_1,
            f"案件事实：\n{case_facts}\n\n庭审记录：\n{dialogue}\n\n"
            f"现在是第{round_no}轮（最多{max_rounds}轮）。请对本轮双方发言作简短点评，"
            "并决定是否结束辩论进入判决。输出JSON："
            '{"comment": "点评内容", "end_debate": true或false}',
        )
        return {"comment": out.get("comment", ""),
                "end_debate": bool(out.get("end_debate", False))}

    def judge(self, case_facts: str, dialogue: str) -> dict:
        """Final structured verdict."""
        return chat_json(
            self._system(),
            MODEL_2,
            f"案件事实：\n{case_facts}\n\n完整庭审记录：\n{dialogue}\n\n"
            "辩论已结束，请作出最终判决。输出JSON：\n"
            "{\n"
            '  "accusation": "认定的罪名",\n'
            '  "relevant_articles": ["引用的法条编号"],\n'
            '  "imprisonment_months": 有期徒刑月数（无则为0）,\n'
            '  "life_imprisonment": true或false,\n'
            '  "death_penalty": true或false,\n'
            '  "fine": 罚金金额（无则为0）,\n'
            '  "reasoning": "判决理由，须回应控辩双方的主要观点"\n'
            "}",
        )