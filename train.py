import torch
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel, PatchFastRL
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import utils, trajectory, reward


PatchFastRL()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokeniser = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4b",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

tokeniser.pad_token = tokeniser.eos_token

encode_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)

optimiser = torch.optim.AdamW(model.parameters(), lr = 5e-6, weight_decay = 0.01)

K = 5
KL_penalty_B = 0.03
GRAD_STEPS = 2
NUM_EPOCHS = 3

def build_tensors(trajectories, tokeniser, device):

    input_ids_list = []
    completion_masks_list = []

    for t in trajectories:
        seq_tokens = []
        seq_mask = []

        for source, tokens in t.token_log:
            seq_tokens += tokens
            
            if source == "RL":
                seq_mask += [1] * len(tokens)
            else:
                seq_mask += [0] * len(tokens)

        input_ids_list.append(torch.tensor(seq_tokens, dtype=torch.long, device = device))
        completion_masks_list.append(torch.tensor(seq_mask, dtype=torch.float32, device=device))

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokeniser.pad_token_id
    )
    padded_completion_mask = torch.nn.utils.rnn.pad_sequence(
        completion_masks_list, batch_first=True, padding_value=0.0
    )

    return padded_input_ids, padded_completion_mask

    


df = pd.read_csv("/workspace/CAIL_ds/CAIL_2018_TRAIN_SAMPLE_2.csv")
    
class Trainer():

    def __init__(self, data, model, tokeniser, encode_model, optimiser, epochs=NUM_EPOCHS, grad_steps =GRAD_STEPS):
        self.epochs = epochs
        self.grad_steps = grad_steps
        self.data = data
        self.reward_model = reward.RewardCalculator()
        self.model = model
        self.tokeniser = tokeniser
        self.encode_model = encode_model
        self.optimiser = optimiser

    
    


    def training(self):
        for e in range(self.epochs):
            self.optimiser.zero_grad()

            for step, case in enumerate(self.data):

                FastLanguageModel.for_inference(self.model)

                ep = trajectory.Episode(
                    case["fact"],
                    case["prosecutor.milestones"],
                    case["relevant_articles"],
                    K,
                    10,
                    self.tokeniser,
                    self.encode_model
                )

                ep.batch_trajectory_rollout(self.model)

                process_results = self.reward_model.group_semantic_eval(ep.trajectories)
                llm_results = self.reward_model.group_llm_eval(ep.trajectories)

                combined_reward =  [x + y for x, y in zip(process_results, llm_results)]

                reward_tensor = torch.tensor(combined_reward, dtype=torch.float32, device=device)

                group_mean  = reward_tensor.mean()
                group_std = reward_tensor.std() + 1e-8

                group_advantages = (reward_tensor - group_mean) / group_std

                # loss calculations

                input_ids, completion_lists = build_tensors(ep.trajectories, self.tokeniser, device)

                FastLanguageModel.for_training(self.model)

                outputs = self.model(input_ids= input_ids)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].cotinguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = completion_lists[:, 1:].contiguous()

                log_probabilties = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none"
                ).view(shift_labels.shape)

                token_log_probs = -log_probabilties

                adv_broadcast = group_advantages.unsqueeze(1)

                policy_loss = - (token_log_probs * adv_broadcast * shift_mask).sum() / (shift_mask.sum() + 1e-8)

                loss = policy_loss / self.grad_steps
                loss.backward()

                if (step + 1) % self.grad_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                print(f"Epoch {e+1} | Step {step+1} | Loss: {loss.item() * self.grad_steps:.4f} | "
                f"Mean Reward: {group_mean.item():.2f} | Std Reward: {group_std.item():.2f}")
            

        self.model.save_pretrained("unsloth_lawyer_grpo")
        self.tokeniser.save_pretrained("unsloth_lawyer_grpo")





train = Trainer(df, model, tokeniser, encode_model, optimiser)
train.training()

