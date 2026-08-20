import torch
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel, PatchFastRL
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import utils, trajectory, reward
import gc


PatchFastRL()
RS = True
device = "cuda" if torch.cuda.is_available() else "cpu"
if RS == False:
    model, tokeniser = FastLanguageModel.from_pretrained(
        model_name="unsloth_lawyer_grpo",
        max_seq_length=8192,
        load_in_4bit=True,
    )
else:
    model, tokeniser = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-4b",
        max_seq_length=8192,
        load_in_4bit=True,
    )

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

model.gradient_checkpointing_enable()


tokeniser.pad_token = tokeniser.eos_token

encode_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

optimiser = torch.optim.AdamW(model.parameters(), lr = 5e-6, weight_decay = 0.01)

K = 3
KL_penalty_B = 0.03
CLIP_eps = 0.2
GRAD_STEPS = 3
NUM_EPOCHS = 1

def build_stepwise_tensors(trajectories, stepwise_advantages, tokeniser, device):

    input_ids_list = []
    completion_masks_list = []
    advantage_masks_list = []

    for i, t in enumerate(trajectories):
        seq_tokens = []
        seq_mask = []
        seq_adv = []

        traj_step_advs = stepwise_advantages[i]


        for entry in t.token_log:
            turn = entry["Turn"]

            source_type = None
            tokens = None
            for k, v in entry.items():
                if k != "Turn":
                    source_type = k
                    tokens = v
                    break

            if tokens is None:
                continue
            
            seq_tokens.extend(tokens)
            if source_type == "RL":
                seq_mask.extend([1.0] * len(tokens))
                seq_adv.extend([traj_step_advs.get(turn, 0.0)] * len(tokens))
            else:
                seq_mask.extend([0.0] * len(tokens))
                seq_adv.extend([0.0] * len(tokens))


        input_ids_list.append(torch.tensor(seq_tokens, dtype=torch.long, device = device))
        completion_masks_list.append(torch.tensor(seq_mask, dtype=torch.float32, device=device))
        advantage_masks_list.append(torch.tensor(seq_adv, dtype=torch.float32, device=device))

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokeniser.pad_token_id
    )
    padded_completion_mask = torch.nn.utils.rnn.pad_sequence(
        completion_masks_list, batch_first=True, padding_value=0.0
    )
    padded_advantage_mask = torch.nn.utils.rnn.pad_sequence(
        advantage_masks_list, batch_first=True, padding_value=0.0
    )

    padded_input_ids = padded_input_ids[:, :4096]
    padded_completion_mask = padded_completion_mask[:, :4096]
    padded_advantage_mask = padded_advantage_mask[:, :4096]

    return padded_input_ids, padded_completion_mask, padded_advantage_mask

    


df = pd.read_csv("CAIL_2018_TRAIN_SAMPLE_500.csv")

class Trainer():

    def __init__(self, data, model, tokeniser, encode_model, optimiser, epochs=NUM_EPOCHS, grad_steps =GRAD_STEPS, log_file = "log.txt"):
        self.epochs = epochs
        self.grad_steps = grad_steps
        self.data = data
        self.reward_model = reward.RewardCalculator()
        self.model = model
        self.tokeniser = tokeniser
        self.encode_model = encode_model
        self.optimiser = optimiser
        self.old_log_probs = None
        self.log_file = log_file
        if RS == True:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("Global_Step\tEpoch\tCase_Step\tLoss\tMean_KL_Div\tMean_Reward\n")

    
    
    def compute_stepwise_advantages(self, semantic_rewards, llm_rewards):
        reward_dict = [{} for _ in range(K)]
        print(f"Raw Semantic Rewards: {semantic_rewards}")
        print(f"Raw LLM Rewards: {llm_rewards}")
        # Add semantic rewards in first
        for i, semantics in enumerate(semantic_rewards):
            for turn, r in semantics:
                reward_dict[i][turn] = reward_dict[i].get(turn, 0.0) + float(r)

        # then llm rewards
        for i, llms in enumerate(llm_rewards):
            for turn, r in enumerate(llms):
                reward_dict[i][turn] = reward_dict[i].get(turn, 0.0) + float(r)

        reward_to_go = [{} for _ in range(K)]
        for i in range(K):
            running = 0.0
            for turn in range(5 - 1, -1, -1):          # iterate BACKWARDS
                running = reward_dict[i].get(turn, 0.0) + 0.95 * running
                reward_to_go[i][turn] = running
        stepwise_advantages = [{} for _ in range(K)]
        trajectory_totals = [0.0] * K

        for turn in range(5):
            turn_rewards = [reward_to_go[i][turn] for i in range(K)]
            turn_tensor = torch.tensor(turn_rewards, dtype=torch.float32, device=device)
            turnwise_mean = turn_tensor.mean()
            turnwise_std = turn_tensor.std() + 1e-8
            turnwise_advantage = (turn_tensor - turnwise_mean) / turnwise_std
               
            

            for i in range(K):
                stepwise_advantages[i][turn] = turnwise_advantage[i].item()

        trajectory_totals = [sum(reward_dict[i].values()) for i in range(K)] 
        mean_episode_reward = sum(trajectory_totals)/max(len(trajectory_totals), 1)

        return stepwise_advantages, mean_episode_reward

    
    def create_log_probs(self, input_ids, shift_labels, log_type = "ref"):

        log_probs = []
        # build ref_log_probs for KL
        if log_type != "current":
            with torch.no_grad():
                for i in range(input_ids.size(0)):
                    traj_input = input_ids[i:i+1]
                    traj_label = shift_labels[i:i+1]

                    if log_type == "ref":
                        with model.disable_adapter():
                            outputs = model(input_ids=traj_input)
                            logits = outputs.logits[:, :-1, :].contiguous()

                    else:
                        outputs = model(input_ids=traj_input)
                        logits = outputs.logits[:, :-1, :].contiguous()


                    log_prob = -torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        traj_label.view(-1),
                        reduction="none"
                    ).view(traj_label.shape)

                    log_probs.append(log_prob)
                    del logits
        else:
            for i in range(input_ids.size(0)):
                traj_input = input_ids[i:i+1]
                traj_label = shift_labels[i:i+1]
                
                outputs = model(input_ids=traj_input)
                logits = outputs.logits[:, :-1, :].contiguous()

                log_prob = -torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    traj_label.view(-1),
                    reduction="none"
                ).view(traj_label.shape)

                log_probs.append(log_prob)
                del logits

        return torch.cat(log_probs, dim=0)


    def compute_grpo_loss(self, log_probabilties, old_log_probs, ref_log_probs, shift_advantages):
        
        # compute importance sampling ratio

        importance_sampling_ratio = torch.exp(log_probabilties - old_log_probs)

        advantage = importance_sampling_ratio * shift_advantages
        clip_value = torch.clamp(importance_sampling_ratio, 1.0-CLIP_eps, 1.0+CLIP_eps) * shift_advantages

        minimum = torch.min(advantage, clip_value)

        # compute KL divergance penalty

        log_ratio = ref_log_probs - log_probabilties
        kl_divergence = torch.exp(log_ratio) - log_ratio - 1.0

        token_loss = minimum - (KL_penalty_B * kl_divergence)
        
        # apply mask to only update RL agent's values
        

        return token_loss, kl_divergence


    def training(self, chkpt=(0, 0)):
        total_steps = chkpt[1] * 500 + chkpt[0]
        
        is_ckhpt = False
        if chkpt[0] != 0:
            is_ckhpt = True
        for e in range(self.epochs-chkpt[1]):
            self.optimiser.zero_grad()
            if is_ckhpt == True:
                index = chkpt[0]
                dataset = self.data.iloc[index:]
                s = chkpt[0]
                is_ckhpt = False
                
            else:
                dataset = self.data
                s = 0
            for step, (idx, case) in enumerate(dataset.iterrows()):
                
                print(f"\n--- Epoch {e+1}/{self.epochs} | Case Step {step+1}/{len(self.data)} ---")
                FastLanguageModel.for_inference(self.model)
                print(case["relevant_articles"])
                ep = trajectory.Episode(
                    case["fact_clean"],
                    case["process_milestones.plantiff"],
                    case["relevant_articles"],
                    K,
                    5,
                    self.tokeniser,
                    self.encode_model,
                    device
                )

                ep.batch_trajectory_rollout(self.model)

                semantic_results = self.reward_model.group_semantic_eval(ep.milestones, ep.trajectories)
                llm_results = self.reward_model.group_llm_eval(ep.trajectories, ep.fact)

                stepwise_advantages, mean_step_reward = self.compute_stepwise_advantages(
                    semantic_results,
                    llm_results
                )


                # loss calculations

                input_ids, completion_lists, advantage_lists = build_stepwise_tensors(ep.trajectories, stepwise_advantages, self.tokeniser, device)

                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = completion_lists[:, 1:].contiguous()
                shift_advantages = advantage_lists[:, 1:].contiguous()

                old_log_probs = self.create_log_probs(
                    input_ids, shift_labels, log_type = "old"
                )

                ref_log_probs = self.create_log_probs(
                    input_ids, shift_labels, log_type = "ref"
                )

                gc.collect()
                torch.cuda.empty_cache()
            
                FastLanguageModel.for_training(self.model)

                for ppo_pass in range(3):  
                    self.optimiser.zero_grad()

                    current_log_probs = self.create_log_probs(input_ids, shift_labels, log_type="current")

                    token_loss, kl_divergence = self.compute_grpo_loss(
                        current_log_probs, old_log_probs, ref_log_probs, shift_advantages
                    )

                    masked_loss = token_loss * shift_mask
                    traj_token_len = shift_mask.sum(dim=1).clamp(min=1.0)
                    grpo_loss = - (masked_loss.sum(dim=1) / traj_token_len).mean()

                    grpo_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
                    self.optimiser.step()
                    

                    print(f"PPO Pass {ppo_pass+1} | Loss: {grpo_loss.item()* self.grad_steps:.4f}")
                    self.log_metrics(
                        global_step=total_steps,
                        epoch=e + 1,
                        case_step=s + 1,
                        loss=grpo_loss.item()* self.grad_steps,
                        kl_div=kl_divergence.mean().item(),
                        mean_reward=mean_step_reward,
                        grad_norm = grad_norm
                    )


                total_steps+=1
                s+=1
                if total_steps%5 == 0:
                    self.model.save_pretrained("unsloth_lawyer_grpo")
                    self.tokeniser.save_pretrained("unsloth_lawyer_grpo")

        self.model.save_pretrained("unsloth_lawyer_grpo")
        self.tokeniser.save_pretrained("unsloth_lawyer_grpo")
                
        

        

    def log_metrics(self, global_step, epoch, case_step, loss, kl_div, mean_reward, grad_norm):
        """Appends metrics to text file and forces buffer flush to disk."""
        log_line = f"{global_step}\t{epoch}\t{case_step}\t{loss:.6f}\t{kl_div:.6f}\t{mean_reward:.4f}\t{grad_norm:.4f}\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
            f.flush()





train = Trainer(df, model, tokeniser, encode_model, optimiser)
train.training((0, 0))

