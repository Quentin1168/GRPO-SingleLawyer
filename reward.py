import torch
import re
import numpy as np
import utils
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import trajectory


class RewardCalculator():

    def __init__(self, threshold: float = 0.70):
        self.trajectories = []
        self.semantic_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
        self.threshold = threshold
        self.process_flag = False
        self.judge_prompt = """你是一名资深刑事/民事审判法官。本轮评估中，你需要对智能体（RL Agent，如公诉人/诉讼代理人）在【一场完整多轮法庭辩论】中的【每一轮发言（Agent Turn）】进行细粒度的质量评估与定性评分。

        【重要原则】：
        1. 案情事实与法律条文的覆盖率（$P_{frac}$）已由底层系统独立计算，你【不需要】考核是否提到了法条，只需【严格按照阶梯标准】评估辩论逻辑、反驳针对性、涵摄质量以及跨轮次的逻辑连贯性。
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
        【违规标记 (Penalty Flags)】
        =========================================
        对智能体的【每一轮发言】分别检查是否存在以下违规，按实际情况标记 true 或 false：
        - stuffing_penalty: 是否仅将法条和事实做无逻辑的列表式堆砌，缺乏辩论结构。
        - hallucination_penalty: 是否歪曲对手原意、捏造案情中不存在的证据/案件事实。

        =========================================
        【评估步骤与 JSON 输出格式】
        =========================================
        步骤 1: 浏览完整对话，梳理智能体（Agent）的辩论脉络与各轮次表现。
        步骤 2: 遍历智能体的【每一轮发言】（如 Agent Turn 1, Agent Turn 2...），依据阶梯标准给出 4 个维度的打分 (0.0 - 1.0)、违规标记及阶梯打分理由。

        严格按以下 JSON 格式输出评估结果:
        {
        "dialogue_overall_summary": "智能体整体辩论逻辑连贯，但在 Turn 2 中面对对手关于主观故意的辩解时反驳力道不足。",
        "turn_evaluations": {
            "Agent Turn 1": {
            "scores": {
                "Q_rebuttal": 0.8,
                "Q_subsumption": 0.9,
                "Q_logic": 0.9,
                "Q_rhetoric": 0.9
            },
            "penalties": {
                "stuffing_penalty": false,
                "hallucination_penalty": false
            },
            "tier_reasoning": "开庭阐述明确，精准将发票证据涵摄入诈骗罪要件，逻辑严密。"
            },
            "Agent Turn 2": {
            "scores": {
                "Q_rebuttal": 0.4,
                "Q_subsumption": 0.6,
                "Q_logic": 0.8,
                "Q_rhetoric": 0.8
            },
            "penalties": {
                "stuffing_penalty": false,
                "hallucination_penalty": false
            },
            "tier_reasoning": "反驳属0.4-0.7阶梯：仅重复了第一轮的立场，未能有效击中 Opponent Turn 1 中提到的‘不知情’辩解。"
            }
        }
        }"""

    def sentence_split(self, text: str):
        split = re.split(r"[。！？\n；]", text)
        return [s.strip() for s in split if len(s.strip())> 4]

    def semantic_flag(self):
        self.process_flag = True

    """
    Evaluates Process milestones using cosine simulation of encoded text vectors. Rewards for the partic
    
    """
    def group_semantic_eval(self, process, trajectories):


        with torch.no_grad():
            process_embeddings = self.model.encode(
                process, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
        
        results = []
        # for each existing trajectory of the episode
        count = 0
        if t.law_check == True:
            count = 1
        for t in trajectories:
            traj_milestones = []
            for a in t.achieved_milestones:
                traj_milestones.append((a[0], 1))
            results.append(traj_milestones)

        return results

    """
    Use this prompt:
    你是一名资深刑事/民事审判法官。本轮评估中，你需要对智能体（RL Agent，如公诉人/诉讼代理人）在【一场完整多轮法庭辩论】中的【每一轮发言（Agent Turn）】进行细粒度的质量评估与定性评分。

    【重要原则】：
    1. 案情事实与法律条文的覆盖率（$P_{frac}$）已由底层系统独立计算，你【不需要】考核是否提到了法条，只需【严格按照阶梯标准】评估辩论逻辑、反驳针对性、涵摄质量以及跨轮次的逻辑连贯性。
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

    1. 反驳针对性 ($Q_{rebuttal}$) - 范围 0.0 至 1.0
    - [0.8 - 1.0分]: 精准切中对手上一轮（或开庭阶段）发言的核心漏洞，利用案情或证据进行了有针对性的事实或法理反驳。
    - [0.4 - 0.7分]: 提到了对手的观点，但反驳泛泛而谈，仅重复我方立场，未能击中对方逻辑要害。
    - [0.0 - 0.3分]: 完全无视对手上一轮的发言，自说自话，或仅重复第一轮的陈辞；若为第一轮发言且未能针对被告的初始答辩/诉求进行有效回应，亦归为此档。

    2. 案情与法条的涵摄深度 ($Q_{subsumption}$) - 范围 0.0 至 1.0
    - [0.8 - 1.0分]: 涵摄极佳。详细论述了本轮涉及的“具体案件事实/证据”是如何精准满足“所引法条/罪名/构成要件”的（完成了事实到法条的逻辑桥梁）。
    - [0.4 - 0.7分]: 罗列了事实和法条，但未深入解释“该事实为何触发该法条”，两者结合较为机械。
    - [0.0 - 0.3分]: 逻辑脱节。列举的事实与所引用的法律要件完全无关（不相关推论）。

    3. 逻辑严密性与跨轮连贯性 ($Q_{logic}$) - 范围 0.0 至 1.0
    - [0.8 - 1.0分]: 本轮推理严密，符合逻辑三段论；且与智能体【此前历史轮次的发言观点保持完全一致】，无任何自相矛盾或立场动摇。
    - [0.4 - 0.7分]: 本轮存在小幅逻辑跳跃，但结论大致能由前提推导得出；或与历史轮次相比观点稍有推诿但未构成严重矛盾。
    - [0.0 - 0.3分]: 本轮内部存在严重自相矛盾，或者【与智能体之前的发言存在直接立场冲突】（例如：上一轮称被告不知情，本轮又称被告主观故意）。

    4. 法庭表达规范 ($Q_{rhetoric}$) - 范围 0.0 至 1.0
    - [0.8 - 1.0分]: 语言庄重规范，使用标准司法措辞，举证责任分配表达准确。
    - [0.0 - 0.5分]: 表达过于口语化、情绪化、人身攻击或不符合法庭礼仪。

    =========================================
    【违规标记 (Penalty Flags)】
    =========================================
    对智能体的【每一轮发言】分别检查是否存在以下违规，按实际情况标记 true 或 false：
    - stuffing_penalty: 是否仅将法条和事实做无逻辑的列表式堆砌，缺乏辩论结构。
    - hallucination_penalty: 是否歪曲对手原意、捏造案情中不存在的证据/案件事实。

    =========================================
    【评估步骤与 JSON 输出格式】
    =========================================
    步骤 1: 浏览完整对话，梳理智能体（Agent）的辩论脉络与各轮次表现。
    步骤 2: 遍历智能体的【每一轮发言】（如 Agent Turn 1, Agent Turn 2...），依据阶梯标准给出 4 个维度的打分 (0.0 - 1.0)、违规标记及阶梯打分理由。

    严格按以下 JSON 格式输出评估结果:
    {
    "dialogue_overall_summary": "智能体整体辩论逻辑连贯，但在 Turn 2 中面对对手关于主观故意的辩解时反驳力道不足。",
    "turn_evaluations": {
        "Agent Turn 1": {
        "scores": {
            "Q_rebuttal": 0.8,
            "Q_subsumption": 0.9,
            "Q_logic": 0.9,
            "Q_rhetoric": 0.9
        },
        "penalties": {
            "stuffing_penalty": false,
            "hallucination_penalty": false
        },
        "tier_reasoning": "开庭阐述明确，精准将发票证据涵摄入诈骗罪要件，逻辑严密。"
        },
        "Agent Turn 2": {
        "scores": {
            "Q_rebuttal": 0.4,
            "Q_subsumption": 0.6,
            "Q_logic": 0.8,
            "Q_rhetoric": 0.8
        },
        "penalties": {
            "stuffing_penalty": false,
            "hallucination_penalty": false
        },
        "tier_reasoning": "反驳属0.4-0.7阶梯：仅重复了第一轮的立场，未能有效击中 Opponent Turn 1 中提到的‘不知情’辩解。"
        }
    }
    }
    Important:
    Back to forth and forth to back group LLM eval. Get average of both set of scores.


    
    """

    def group_llm_eval(self, trajectories, fact):

        # evaluate each trajectory
        results = []
        index = []
        count = 0
        for t in trajectories:
            traj_results = []
            full_prompt = t.get_prompt()

            # concatenate the judge_prompt with the full_prompt

            current_prompt = self.judge_prompt.format(case_facts = fact, dialogue_history = full_prompt)
            output = utils.chat_json(self.current_prompt, "Deepseek V3")

            # process answers from json output

            for o in output:
                count += 1
                # Weighted Calculation
                total_score = 0.5 * o["Q_rebuttal"] + 0.25 * o["Q_subsumption"] + 0.15 * o["Q_logic"] + 0.1 * ["Q_rhetoric"]

                traj_results.append(total_score)


            results.append(sum(traj_results)/len(traj_results))
    
        return results

                




