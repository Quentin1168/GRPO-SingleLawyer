#!/usr/bin/env python

import pandas as pd
import llm
from simulation import CourtSimulation

def main():
    df = pd.read_csv("/workspace/CAIL_ds/CAIL_2018_TRAIN_SAMPLE.csv")
    retriever = llm.LawRetriever("/workspace/CAIL_ds/law.json")

    # Agents persist across cases -> memories accumulate (contextual learning)
    sim = CourtSimulation(retriever)

    results = []
    for i, row in df.iterrows():
        print(f"\n{'#'*70}\n# 案件 {i + 1}/{len(df)}：{row['accusation']}\n{'#'*70}")
        results.append(sim.run_case(row))

    # crude accuracy check across the run
    correct = sum(
        set(map(str, r["verdict"].get("relevant_articles", [])))
        & set(map(str, r["ground_truth"]["relevant_articles"]))
        != set()
        for r in results
    )
    print(f"\n法条引用命中率: {correct}/{len(results)}")

if __name__ == "__main__":
    main()