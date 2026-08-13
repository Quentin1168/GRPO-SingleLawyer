from llm import chat

MODEL_1 = "qwen/qwen3-4b"
MODEL_2 = "google/gemini-3.1-flash-lite"

LAWYER_REFLECT_SYS = (
    "你是一名律师庭审表现评估专家。请根据律师的实际发言与该案预设的论证要点（milestones），"
    "评估该律师的表现，并生成一段简短的经验总结供其今后参考。"
)

JUDGE_REFLECT_SYS = (
    "你是一名司法裁判质量评估专家。请对比模拟法官的判决与真实判决，"
    "评估其准确性与说理质量，并生成一段简短的经验总结供其今后参考。"
)


def reflect_lawyer(lawyer, dialogue: str, milestones: list[str],
                   opponent_name: str, case_facts: str) -> str:
    """Evaluate milestone coverage + relative performance; returns reflection text."""
    reflection = chat(
        LAWYER_REFLECT_SYS,
        MODEL_2,
        f"角色：{lawyer.name}\n\n案件事实（摘要）：\n{case_facts[:500]}\n\n"
        f"该律师本案应当使用的论证要点（含应引用的法条）：\n"
        + "\n".join(f"{i+1}. {m}" for i, m in enumerate(milestones))
        + f"\n\n完整庭审记录：\n{dialogue}\n\n"
        f"请输出一段不超过150字的经验总结，须包含：\n"
        f"(1) 案件一句话概括；(2) 该律师命中了哪些要点（做对了什么）；"
        f"(3) 遗漏或错误了哪些要点；(4) 与{opponent_name}相比谁的论证更有力。",
        temperature=0.3,
    )
    lawyer.memory.add(reflection)
    return reflection


def reflect_judge(judge, verdict: dict, ground_truth: dict,
                  dialogue: str, fact_reason: str, case_facts: str) -> str:
    reflection = chat(
        JUDGE_REFLECT_SYS,
        MODEL_2,
        f"案件事实（摘要）：\n{case_facts[:500]}\n\n"
        f"模拟法官的判决：\n{verdict}\n\n"
        f"真实判决结果：\n{ground_truth}\n\n"
        f"真实判决说理：\n{fact_reason}\n\n"
        f"庭审记录：\n{dialogue}\n\n"
        "请输出一段不超过150字的经验总结，须包含：\n"
        "(1) 案件一句话概括；(2) 模拟判决 vs 真实判决的差异（罪名、刑期、罚金）；"
        "(3) 法官是否回应了控辩双方观点、说理是否充分；(4) 与真实判决理由相比的差距。",
        temperature=0.3,
    )
    judge.memory.add(reflection)
    return reflection