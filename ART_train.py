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

LEARNING_RATE = 5e-6
GROUP_SIZE = 8
LOG_FILE = "metrics.jsonl"

model = art.TrainableModel(
    name ="Qwen/Qwen3-4B",
    project="GRPO-SingleLawyer",
    base_model="Qwen/Qwen3-4B",
    _internal_config=art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
        max_seq_length=16384,        # training must consume what inference produced
        load_in_4bit=True,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=0.70,
            enforce_eager=True,
            max_num_seqs=8,
            max_model_len=16384,         # keep in lockstep with max_seq_length
        ),
        peft_args=art.dev.PeftArgs(r=16, lora_alpha=16, lora_dropout=0),
    ),

)

def snapshot_checkpoint(step):
    ckpts = sorted(glob.glob(f"./.art/{'GRPO-SingleLawyer'}/models/{'Qwen/Qwen3-4B'}/checkpoints/*"))
    if ckpts:  # path layout can vary slightly by ART version — verify once on disk
        dst = f"./snapshots/step_{step:04d}"
        if not os.path.exists(dst):
            shutil.copytree(ckpts[-1], dst)


def log_trajectories(step, epoch, groups):
    all_rewards, all_milestones, all_judge, all_law, all_trunc = [], [], [], [], []
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for g in groups:
            subs = list(g.trajectories)
            rewards = [t.reward for t in subs]
            mean = sum(rewards) / max(len(rewards), 1)
            var = sum((r - mean) ** 2 for r in rewards) / max(len(rewards), 1)
            std = var ** 0.5

            for t in subs:
                f.write(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "case_id": t.metadata.get("case_id"),
                    "turn": int(t.metadata.get("turn", -1)),
                    "reward": t.reward,                          # reward-to-go
                    "advantage": (t.reward - mean) / (std + 1e-8),  # reconstruction
                    **{k: v for k, v in t.metrics.items()},      # everything else
                }, ensure_ascii=False) + "\n")
        f.flush()

    n = max(len(all_rewards), 1)
    if wandb.run is not None:                      # attaches to ART's run
        wandb.log({
            "custom/reward_mean": sum(all_rewards) / n,
            "custom/reward_std": std,              # GRPO vital sign
            "custom/milestones_mean": sum(all_milestones) / n,
            "custom/judge_mean": sum(all_judge) / n,
            "custom/law_found_rate": sum(all_law) / n,
            "custom/truncation_rate": sum(all_trunc) / n,
            "custom/reward_hist": wandb.Histogram(all_rewards),
            "epoch": epoch,
        }, step=step)

async def evaluate(model, eval_cases, step, helper):
    """Rollouts on held-out cases. Deliberately NO judge call — eval uses only
    deterministic, free metrics (see below). Opponent API is still needed
    (it IS the environment), so eval costs DeepSeek calls but zero judge calls."""
    records = []
    for case in eval_cases:
        milestones = helper.process_milestones
        embs = helper.process_embeddings
        trajs = await art.gather_trajectories(
            [helper.trajectory_rollout(model, case, milestones, embs) for _ in range(10)]
        )
        for t in trajs:
            n_turns = sum(f"turn_reward_{i}" in t.metrics for i in range(10))
            frac = t.metrics["milestones_achieved"] / max(len(milestones), 1)
            records.append({
                "milestone_frac": frac,
                "law_found": t.metrics["law_found"],
                "success": float(frac >= 1.0 and t.metrics["law_found"] > 0),
                "turns_used": n_turns,
            })

    agg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    with open("eval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, **agg}) + "\n")
    print(f"[EVAL @ step {step}] " + " | ".join(f"{k}={v:.3f}" for k, v in agg.items()))
    return agg


async def train():
    backend = LocalBackend(path="./.art")
    await model.register(backend)

    df = pd.read_csv("CAIL_2018_TRAIN_SAMPLE_2000.csv")
    eval_df, train_df = df.iloc[:10], df.iloc[10:]   # held out!
    eval_cases = [{"idx": i, **r.to_dict()} for i, r in eval_df.iterrows()]
    cases = [{"idx": i, **r.to_dict()} for i, r in train_df.iterrows()]

    initial_step = await model.get_step()
    encode_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

    for batch in iterate_dataset(cases, groups_per_step=5, num_epochs=5,
                             initial_step=initial_step):
        case = batch.items[0]
        print(f"\n--- Step {batch.step} | Epoch {batch.epoch} | Case {case['idx']} ---")
        helper = environment.TrajectoryHelper(
            case["idx"],                       # was: id
            case["fact_clean"], case["process_milestones.plantiff"],
            case["relevant_articles"], GROUP_SIZE, 10, model, encode_model,
            "cuda",                            # was: "gpu" — torch device string is "cuda"
        )

        trajectory_groups = await art.gather_trajectory_groups(
            [art.TrajectoryGroup(
                (helper.trajectory_rollout() for _ in range(helper.size)),
                metadata={"case_id": helper.id}
            )],
            after_each=lambda g: ART_reward.batch_score(g, case),
            max_exceptions=0,
        )

        await model.train(trajectory_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))

        trajs = [t for g in trajectory_groups for t in g.trajectories] 

        log_trajectories(batch.step, batch.epoch, trajectory_groups)

        if batch.step % 15 == 0 and batch.step > 0:
            snapshot_checkpoint(batch.step)
            await evaluate(model, eval_cases, batch.step, helper)

    await backend.close()

if __name__ == "__main__":
    asyncio.run(train())