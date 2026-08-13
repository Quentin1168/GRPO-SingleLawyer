#!/usr/bin/env python

import ast
from legal_agents import LawyerAgent, JudgeAgent
from reflection import reflect_lawyer, reflect_judge
MAX_ROUNDS = 5


def _parse_list(cell) -> list:
    """Dataset stores lists as strings, e.g. \"['王某']\"."""
    if isinstance(cell, list):
        return cell
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return [str(cell)]


class CourtSimulation:
    def __init__(self, retriever):
        self.judge = JudgeAgent()
        self.plaintiff = LawyerAgent("plaintiff", retriever)
        self.defendant = LawyerAgent("defendant", retriever)

    def run_case(self, row, verbose: bool = True) -> dict:
        facts = row["fact_clean"]
        log: list[str] = []

        def say(speaker: str, text: str):
            entry = f"{speaker}：{text}"
            log.append(entry)
            if verbose:
                print(f"\n{entry}")

        # ---------------- Preparation phase ---------------- #
        if verbose:
            print("=" * 60, "\n[准备阶段] 双方律师检索法条并制定策略...")
        self.plaintiff.prepare(facts)
        self.defendant.prepare(facts)

        # ---------------- Debate rounds ---------------- #
        for rnd in range(1, MAX_ROUNDS + 1):
            if verbose:
                print(f"\n{'='*60}\n[第 {rnd} 轮辩论]")
            dialogue = "\n".join(log)
            say(self.plaintiff.name, self.plaintiff.argue(facts, dialogue, rnd))
            say(self.defendant.name, self.defendant.argue(facts, "\n".join(log), rnd))

            ruling = self.judge.comment(facts, "\n".join(log), rnd, MAX_ROUNDS)
            say(self.judge.name, ruling["comment"])
            if ruling["end_debate"] or rnd == MAX_ROUNDS:
                break

        # ---------------- Judgement ---------------- #
        dialogue = "\n".join(log)
        verdict = self.judge.judge(facts, dialogue)
        say(self.judge.name, f"【判决】{verdict}")

        # ---------------- Reflection (contextual learning) ---------------- #
        ground_truth = {
            "criminals": _parse_list(row["criminals"]),
            "accusation": _parse_list(row["accusation"]),
            "relevant_articles": _parse_list(row["relevant_articles"]),
            "imprisonment_months": row["term_of_imprisonment.imprisonment"],
            "life_imprisonment": row["term_of_imprisonment.life_imprisonment"],
            "death_penalty": row["term_of_imprisonment.death_penalty"],
            "fine": row["punish_of_money"],
        }

        p_ref = reflect_lawyer(self.plaintiff, dialogue,
                               _parse_list(row["process_milestones.plantiff"]),
                               self.defendant.name, facts)
        d_ref = reflect_lawyer(self.defendant, dialogue,
                               _parse_list(row["process_milestones.defendant"]),
                               self.plaintiff.name, facts)
        j_ref = reflect_judge(self.judge, verdict, ground_truth,
                              dialogue, row["fact_reason"], facts)

        if verbose:
            print(f"\n{'='*60}\n[反思阶段]")
            print(f"\n公诉人反思：{p_ref}")
            print(f"\n辩护人反思：{d_ref}")
            print(f"\n法官反思：{j_ref}")

        return {"dialogue": dialogue, "verdict": verdict,
                "ground_truth": ground_truth,
                "reflections": {"plaintiff": p_ref, "defendant": d_ref, "judge": j_ref}}