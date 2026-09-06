import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"   # ALWAYS precede any art/unsloth import
from dotenv import load_dotenv
load_dotenv()
import asyncio
import pandas as pd
import unsloth
import art
from art.local import LocalBackend
from art.utils import iterate_dataset
import environment 
import ART_reward
from sentence_transformers import SentenceTransformer
import glob
import shutil
import json
import wandb
import numpy as np
import psutil


LEARNING_RATE = 5e-6
GROUP_SIZE = 8
RUN_NAME = os.environ.get("RUN_NAME", "grpo-lawyer-56")   
PROJECT  = "GRPO-SingleLawyer"
CONTEXT_LEN = 12288 

RUN_DIR   = f"./runs/{RUN_NAME}"
LOG_FILE  = f"{RUN_DIR}/trajectories.jsonl"
EVAL_FILE = f"{RUN_DIR}/eval.jsonl"
SNAP_DIR  = f"{RUN_DIR}/snapshots"
DEMO_FILE = "demo.jsonl"
os.makedirs(SNAP_DIR, exist_ok=True)

model = art.TrainableModel(
    name =RUN_NAME,
    project=PROJECT,
    base_model="Qwen/Qwen3-4B",
    _internal_config=art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
        max_seq_length=CONTEXT_LEN,        
        load_in_4bit=True,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.8,
            enforce_eager=False,
            max_num_seqs=24,
            max_model_len=CONTEXT_LEN,        
            enable_prefix_caching=True
        ),
        peft_args=art.dev.PeftArgs(r=16, lora_alpha=16, lora_dropout=0),
        
    ),

)

encode_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
# get a completely seperate test set without any train_df entries
train_df = pd.read_csv("CAIL_2018_TRAIN_SAMPLE_2000.csv")
original_df = pd.read_csv("CAIL_2018_TRAIN_SAMPLE.csv")

train_df_ids = train_df.columns[0]
original_df_ids = original_df.columns[0]

# filter by used ids
used_ids = set(train_df[train_df_ids])


available_samples = original_df[~original_df[original_df_ids].isin(used_ids)]

# get 200 cases randomly
test_sample_200 = available_samples.sample(n=100, random_state=21)

test_sample_50 = test_sample_200.sample(n=50, random_state=21)

demo_case = [{"idx": i, **r.to_dict()} for i, r in test_sample_50.iterrows()]


eval_case = [{"idx": i, **r.to_dict()} for i, r in test_sample_200.iterrows()]


"""
test function to evaluate model on a larger test set.
Similar to the evaluate function during training.

"""
async def test(model, eval_cases, encode_model):
    backend = LocalBackend(path="./.art")
    await model.register(backend)
    
    records = []
    for count, case in enumerate(eval_cases):
        print("Current Case: ", count)
        helper = environment.TrajectoryHelper(
            case["idx"], case["fact_clean"], case["process_milestones.plantiff"],
            case["relevant_articles"], 8, model, encode_model, "cuda",
        )
        
        trajs = await art.gather_trajectories(
            [helper.trajectory_rollout() for _ in range(1)]
        )
        for t in trajs:
            n_turns = sum(f"turn_reward_{i}" in t.metrics for i in range(helper.max_turns))
            frac = t.metrics.get("milestone_frac", 0.0)
            records.append({
                "milestone_frac": frac,
                "law_found": t.metrics.get("law_found", 0.0),
                "success": float(frac >= 1.0 and t.metrics.get("law_found", 0.0) > 0),
                "turns_used": n_turns,
            })



    if not records:
        return {}
    agg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    with open(EVAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({**agg}) + "\n")

    return agg

"""
record_demo function to evaluate model on a larger test set.
Similar to the evaluate and test function, but to precompute demo transcripts for display

"""
async def record_demo(model, eval_cases, encode_model):
    backend = LocalBackend(path="./.art")
    await model.register(backend)
    
    records = []
    for count, case in enumerate(eval_cases):
        print("Current Case: ", count)
        helper = environment.TrajectoryHelper(
            case["idx"], case["fact_clean"], case["process_milestones.plantiff"],
            case["relevant_articles"], 8, model, encode_model, "cuda",
        )
        
        trajs = await art.gather_trajectories(
            [helper.trajectory_rollout() for _ in range(1)]
        )
        for t in trajs:
            n_turns = sum(f"turn_reward_{i}" in t.metrics for i in range(helper.max_turns))
            frac = t.metrics.get("milestone_frac", 0.0)
            records.append({
                "milestone_frac": frac,
                "law_found": t.metrics.get("law_found", 0.0),
                "success": float(frac >= 1.0 and t.metrics.get("law_found", 0.0) > 0),
                "turns_used": n_turns,
            })

            milestone_turns = json.loads(t.metadata["milestone_found_turn"])
            law_found = t.metrics.get("law_found_turn")

            messages = serialize_messages(t)
            # append all information needed for case reenactment. 
            row = {
                "case_id": int(case["idx"]),
                "fact": case["fact_clean"],
                "milestones": str(case["process_milestones.plantiff"]),
                "milestone_history": milestone_turns,
                "messages": serialize_messages(t),
                "law_found_turn": law_found,
                "metrics": dict(t.metrics)
            }

            with open(DEMO_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")




def serialize_messages(traj):
    out = []
    for m in traj.messages_and_choices:
        if isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:  
            out.append({"role": "assistant", "content": m.message.content or ""})
    return out



    
if __name__ == "__main__":
    asyncio.run(test(model, eval_case, encode_model))
    #asyncio.run(record_demo(model, demo_case, encode_model))
