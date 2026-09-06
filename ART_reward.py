import asyncio
import re
import art
import utils
import numpy as np

MAX_TURNS = 5

# LLM Judge for determining LLM quality
JUDGE_PROMPT = """你是一名资深刑事/民事审判法官。本轮评估中，你需要对智能体（RL Agent，如公诉人/诉讼代理人）在【一场完整多轮法庭辩论】中的【每一轮发言（Agent Turn）】进行细粒度的质量评估与定性评分。

【重要原则】：
1.  你【不需要】考核是否提到了法条，只需【严格按照阶梯标准】评估辩论逻辑、反驳针对性、涵摄质量以及跨轮次的逻辑连贯性。
2. 【客观独立评分】：请结合对话上下文，对智能体的【每一轮发言】分别进行独立打分。不同轮次的发言质量可能存在起伏，请准确识别出质量较高或较低的轮次，【切勿对所有轮次打出完全相同的分数】。

=========================================
【案件背景与完整辩论历史】
=========================================
[案情事实]:
{case_facts}

[完整法庭辩论历史记录]:
{dialogue_history}
<!-- 对话历史格式示例:
[Agent Turn 1]: ...
[Opponent Turn 1]: ...
[Agent Turn 2]: ...
[Opponent Turn 2]: ...
-->

=========================================
【评分维度与阶梯式标准 (0.0 - 1.0 分)】
=========================================
针对智能体（Agent）的【每一轮发言】，请严格对照以下阶梯标准独立打分：

1. 反驳针对性 ($Q_{{rebuttal}}$) - 范围 0.0 至 1.0
- [0.8 - 1.0分]: 精准切中对手上一轮（或开庭阶段）发言的核心漏洞，利用案情或证据进行了有针对性的事实或法理反驳。
- [0.4 - 0.7分]: 提到了对手的观点，但反驳泛泛而谈，仅重复我方立场，未能击中对方逻辑要害。
- [0.0 - 0.3分]: 完全无视对手上一轮的发言，自说自话，或仅重复第一轮的陈辞；若为第一轮发言且未能针对被告的初始答辩/诉求进行有效回应，亦归为此档。

2. 案情与法条的涵摄深度 ($Q_{{subsumption}}$) - 范围 0.0 至 1.0
- [0.8 - 1.0分]: 涵摄极佳。详细论述了本轮涉及的“具体案件事实/证据”是如何精准满足“所引法条/罪名/构成要件”的（完成了事实到法条的逻辑桥梁）。
- [0.4 - 0.7分]: 罗列了事实和法条，但未深入解释“该事实为何触发该法条”，两者结合较为机械。
- [0.0 - 0.3分]: 逻辑脱节。列举的事实与所引用的法律要件完全无关（不相关推论）。

3. 逻辑严密性与跨轮连贯性 ($Q_{{logic}}$) - 范围 0.0 至 1.0
- [0.8 - 1.0分]: 本轮推理严密，符合逻辑三段论；且与智能体【此前历史轮次的发言观点保持完全一致】，无任何自相矛盾或立场动摇。
- [0.4 - 0.7分]: 本轮存在小幅逻辑跳跃，但结论大致能由前提推导得出；或与历史轮次相比观点稍有推诿但未构成严重矛盾。
- [0.0 - 0.3分]: 本轮内部存在严重自相矛盾，或者【与智能体之前的发言存在直接立场冲突】（例如：上一轮称被告不知情，本轮又称被告主观故意）。

4. 法庭表达规范 ($Q_{{rhetoric}}$) - 范围 0.0 至 1.0
- [0.8 - 1.0分]: 语言庄重规范，使用标准司法措辞，举证责任分配表达准确。
- [0.0 - 0.5分]: 表达过于口语化、情绪化、人身攻击或不符合法庭礼仪。



=========================================
【评估步骤与 JSON 输出格式】
=========================================
步骤 1: 浏览完整对话，梳理智能体（Agent）的辩论脉络与各轮次表现。
步骤 2: 遍历智能体的【每一轮发言】（如 Agent Turn 1, Agent Turn 2...），依据阶梯标准给出 4 个维度的打分 (0.0 - 1.0)、违规标记及阶梯打分理由。

严格按以下 JSON 格式输出评估结果:
{{
"dialogue_overall_summary": "...",
"turn_evaluations": {{
    "Agent Turn 1": {{
    "scores": {{"Q_rebuttal": 0.8, "Q_subsumption": 0.9, "Q_logic": 0.9, "Q_rhetoric": 0.9}},
    "tier_reasoning": "..."
    }}
}}
}}"""



"""
flush function to clear the buffer, which is used to convert the dialogue in blocks

parameters:
agent_turn - current turn of dialogue
prompt_str - the current dialogue block converted to string
buffer - the current stored buffer

returns: 
buffer - a cleared out buffer
prompt_str - the updated dialogue block converted into a string with the buffer contents
"""

def flush(agent_turn, prompt_str, buffer):
    if buffer:
        prompt_str.append(f"[Agent Turn {agent_turn}]: " + "\n".join(buffer))
        buffer = []

        return buffer, prompt_str

    return buffer, prompt_str

"""
build_transcript converts trajectory messages_and_choices into string dialogue for judge to read.

parameters: 
trajectory - ART trajectory object to be converted

returns - trajectory dialogue in string format
"""
def build_transcript(trajectory):
    trajectory_messages = trajectory.messages_and_choices
    prompt_str, buffer, agent_turn = [], [], 0

    # buffer system allows for dialogue to be converted in blocks
    for m in trajectory_messages:
        # detect if dialogue to be formatted is actually part of the chat or just a database result
        # ART dialogues can be picked up as dicts
        # if it is, store in a buffer til static agent's chat
        if not isinstance(m, dict):
            buffer.append(m.message.content or "")

        # if it is the static agent, flush (add the buffer in the dialog in)
        elif m["role"] == "user" and not m["content"].startswith("[Database Result]"):
            agent_turn += 1
            buffer, prompt_str = flush(agent_turn, prompt_str, buffer)
            # add the static dialogue in 
            prompt_str.append(f"[Opponent Turn {agent_turn}]: {m['content']}")

    agent_turn += 1
    buffer, prompt_str = flush(agent_turn, prompt_str, buffer)

    return "\n".join(prompt_str)

"""
llm_judge to calculate debate quality score. The score is calculated as a rolling mean calculated per dialogue.

paramters:
trajectory - specified trajectory in question
fact - the initial case facts

returns:
running_mean - the final rolling mean

"""
async def llm_judge(trajectory, fact):
    # format judge prompt with specific case facts
    prompt = JUDGE_PROMPT.format(case_facts=fact,
                                 dialogue_history=build_transcript(trajectory))
    
    output = {}
    # retry 3 times due to API response unpredictability
    for attempt in range(3):
        try:
            output = await asyncio.to_thread(utils.chat_json,
                                             "z-ai/glm-4.7-flash", prompt)
            if output:
                break
        except Exception as e:
            print(f"Judge attempt {attempt} failed: {e}")
    # log failure to parse if any
    trajectory.metrics["judge_parse_failure"] = 0.0 if output else 1.0
    if not output:
        return np.array([0.0])
    
    traj_results = []
    turn_evals = output.get("turn_evaluations", {})
    for key in turn_evals:
        # get per-turn data
        turn_data = turn_evals[key]
        scores = turn_data.get("scores", {})

        # weighted calculation
        total_score = 0.5 * float(scores.get("Q_rebuttal", 0.0)) + \
        0.25 * float(scores.get("Q_subsumption", 0.0)) + 0.15 * float(scores.get("Q_logic", 0.0)) + \
            0.1 * float(scores.get("Q_rhetoric", 0.0))
        traj_results.append(total_score)

    traj_np = np.array(traj_results)
    # calculate running mean
    running_mean = np.cumsum(traj_np) / np.arange(1, len(traj_np) + 1)

    return running_mean

"""
batch_score to allow for group-wide scoring of trajectories to increase efficiency.
This is calculated after the milestone rewards at it scores the entire debate.

parameters:
group - the group of trajectories
case - the case the trajectories are under

returns:
the group of trajectories with updated rewards

"""
async def batch_score(group, case):
    trajectories = list(group.trajectories)
    judge_results = await asyncio.gather(
        *(llm_judge(t, case["fact_clean"]) for t in trajectories)
    )

    for i, t in enumerate(trajectories):
        milestone_reward = sum(t.metrics.get(f"turn_reward_{k}", 0.0)
                               for k in range(MAX_TURNS))
        # get final running mean
        judge_score = float(judge_results[i][-1])  
        # update reward for current trajectory
        t.reward = judge_score + milestone_reward
        t.metrics["milestone_reward"] = milestone_reward
        t.metrics["judge_reward"] = judge_score
    return group





    
    
                


