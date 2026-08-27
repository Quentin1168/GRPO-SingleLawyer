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
RUN_NAME = os.environ.get("RUN_NAME", "grpo-lawyer-54")   
PROJECT  = "GRPO-SingleLawyer"
CONTEXT_LEN = 12288 

RUN_DIR   = f"./runs/{RUN_NAME}"
LOG_FILE  = f"{RUN_DIR}/trajectories.jsonl"
EVAL_FILE = f"{RUN_DIR}/eval.jsonl"
SNAP_DIR  = f"{RUN_DIR}/snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

model = art.TrainableModel(
    name =RUN_NAME,
    project=PROJECT,
    base_model="Qwen/Qwen3-4B",
    _internal_config=art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
        max_seq_length=CONTEXT_LEN,        # training must consume what inference produced
        load_in_4bit=True,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.45,
            enforce_eager=False,
            max_num_seqs=24,
            max_model_len=CONTEXT_LEN,         # keep in lockstep with max_seq_length
            enable_prefix_caching=True
        ),
        peft_args=art.dev.PeftArgs(r=16, lora_alpha=16, lora_dropout=0),
        
    ),

)

def snapshot_checkpoint(step):
    ckpts = sorted(glob.glob(f"./.art/{PROJECT}/models/{RUN_NAME}/checkpoints/*"))
    if ckpts:
        dst = f"{SNAP_DIR}/step_{step:04d}"
        if not os.path.exists(dst):
            shutil.copytree(ckpts[-1], dst)

def log_trajectories(step, epoch, groups):
    all_rewards, all_milestones, all_judge, all_law, all_trunc = [], [], [], [], []
    all_abs_adv, all_searches = [], []                    
    zero_var_groups, n_groups = 0, 0 
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for g in groups:
            subs = list(g.trajectories)
            rewards = [t.reward for t in subs]
            mean = sum(rewards) / max(len(rewards), 1)
            var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
            std = var ** 0.5

            n_groups += 1                                   
            if std < 1e-6:                                
                zero_var_groups += 1 

            for t in subs:
                all_rewards.append(t.reward)
                adv = (t.reward - mean) / (std + 1e-8)
                all_abs_adv.append(abs(adv)) 
                all_searches.append(t.metrics.get("searches", 0.0))
                all_milestones.append(t.metrics.get("milestones_frac", 0.0))
                all_judge.append(t.metrics.get("judge_reward", 0.0))
                all_law.append(t.metrics.get("law_found", 0.0))
                all_trunc.append(t.metrics.get("truncated_by_context", 0.0))
                f.write(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "case_id": t.metadata.get("case_id"),
                    "turn": int(t.metadata.get("turn", -1)),
                    "reward": t.reward,
                    "advantage": (t.reward - mean) / (std + 1e-8),
                    **{k: v for k, v in t.metrics.items()},
                }, ensure_ascii=False) + "\n")
        f.flush()

    n = max(len(all_rewards), 1)
    mean_all = sum(all_rewards) / n
    std_all = (sum((r - mean_all) ** 2 for r in all_rewards) / n) ** 0.5
    if wandb.run is not None:
        wandb.log({
            "custom/reward_mean": mean_all,
            "custom/reward_std": std_all,
            "custom/milestones_frac_mean": sum(all_milestones) / n,
            "custom/judge_mean": sum(all_judge) / n,
            "custom/law_found_rate": sum(all_law) / n,
            "custom/truncation_rate": sum(all_trunc) / n,
            "custom/reward_hist": wandb.Histogram(all_rewards) if all_rewards else 0.0,
            "custom/epoch": epoch,
            "custom/step": step,          # own axis — never dropped by ART's counter
            "custom/mean_abs_advantage": sum(all_abs_adv) / n,                   
            "custom/zero_variance_group_frac": zero_var_groups / max(n_groups, 1), 
            "custom/searches_per_game": sum(all_searches) / n,                   
        })


async def evaluate(model, eval_cases, step, encode_model):
    """Deterministic metrics only — no judge calls. Builds a fresh helper per
    eval case (the trainer's helper carries the WRONG case's milestones)."""
    records = []
    for case in eval_cases:
        helper = environment.TrajectoryHelper(
            case["idx"], case["fact_clean"], case["process_milestones.plantiff"],
            case["relevant_articles"], GROUP_SIZE, 8, model, encode_model, "cuda",
        )
        
        trajs = await art.gather_trajectories(
            [helper.trajectory_rollout() for _ in range(10)]
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
        f.write(json.dumps({"step": step, **agg}) + "\n")
    if wandb.run is not None:
        wandb.log({**{f"eval/{k}": v for k, v in agg.items()}, "custom/step": step})
    print(f"[EVAL @ step {step}] " + " | ".join(f"{k}={v:.3f}" for k, v in agg.items()))
    return agg


async def train():
    backend = LocalBackend(path="./.art")
    await model.register(backend)

    if wandb.run is not None:
        wandb.define_metric("custom/*", step_metric="custom/step")
        wandb.define_metric("eval/*", step_metric="custom/step")

    df = pd.read_csv("CAIL_2018_TRAIN_SAMPLE_2000.csv")
    eval_df, train_df = df.iloc[:10], df.iloc[10:]
    eval_cases = [{"idx": i, **r.to_dict()} for i, r in eval_df.iterrows()]
    cases = [{"idx": i, **r.to_dict()} for i, r in train_df.iterrows()]

    initial_step = await model.get_step()
    encode_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

    # groups_per_step=1: batch.items[0] is now the WHOLE batch — no cases skipped
    for batch in iterate_dataset(cases, groups_per_step=3, num_epochs=5,
                                 initial_step=initial_step):
        print(f"\n--- Step {batch.step} | Epoch {batch.epoch} | "
              f"Cases {[c['idx'] for c in batch.items]} ---")

        case_by_id = {}
        groups = []
        for case in batch.items:
            helper = environment.TrajectoryHelper(
                case["idx"], case["fact_clean"], case["process_milestones.plantiff"],
                case["relevant_articles"], GROUP_SIZE, 8, model, encode_model, "cuda",
            )
            case_by_id[helper.id] = case
            groups.append(art.TrajectoryGroup(
                (helper.trajectory_rollout() for _ in range(helper.size)),
                metadata={"case_id": helper.id},
            ))

        trajectory_groups = await art.gather_trajectory_groups(
            groups,
            # look up the RIGHT case per group — a plain `lambda g: batch_score(g, case)`
            # would close over the loop variable and score all 5 groups against case #5
            after_each=lambda g: ART_reward.batch_score(
                g, case_by_id[g.metadata["case_id"]]
            ),
            max_exceptions=2,
        )
        print(f"[ram after gather] free={psutil.virtual_memory().available/1e9:.0f}G "
      f"shmem={psutil.virtual_memory().shared/1e9:.0f}G", flush=True)
        await model.train(trajectory_groups,
                          config=art.TrainConfig(learning_rate=LEARNING_RATE, kl_penalty_coef=0.01))
        print(f"[ram after train    ] free={psutil.virtual_memory().available/1e9:.0f}G "
      f"shmem={psutil.virtual_memory().shared/1e9:.0f}G", flush=True)
        log_trajectories(batch.step, batch.epoch, trajectory_groups)
        
        if batch.step % 10 == 0 and batch.step > 0:
            snapshot_checkpoint(batch.step)
            await evaluate(model, eval_cases, batch.step, encode_model)
        

        
    await backend.close()


if __name__ == "__main__":
    asyncio.run(train())
