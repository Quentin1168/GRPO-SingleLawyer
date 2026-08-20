import utils
import concurrent.futures
import numpy as np
import cn2an
import re
import torch
from sentence_transformers import util
import ast


class Trajectory():

    def __init__(self, fact, milestones, id, law):
        self.milestones = milestones
        self.fact = fact
        self.id = id
        self.law = law
        self.rl_prompt = self.build_RL()
        self.static_prompt = self.build_static()
        self.prompt = fact
        self.rl_response = ""
        self.static_response = ""
        self.achieved_milestones = []
        self.achieved_set = set()
        self.turn = 0
        self.token_log = []
        self.law_check = False
        

        
        self.done = False

    def build_RL(self):
        init = f"你是本案的原告律师，请根据以下事实进行控方辩论：\n{self.fact}\n\n 【工具使用说明】\n" \
            f"你在辩论过程中可以使用检索工具查询相关法律法规或案事实细节。\n" \
            f"当你需要进行检索时，请在回答中使用 `<search>案情特征或法律争议点描述</search>` 格式。\n" \
            f"注意：检索时请提交与案情事实、行为性质或争议焦点相关的上下文描述（而非直接搜索具体法条名称），以便系统为你匹配最相关的法律依据。\n" \
            f"例如：`<search>未经同意秘密转移他人财物 盗窃罪认定与量刑标准</search>` 或 `<search>合同到期拒绝履行还款义务 违约金与利息计算</search>`。\n" \
            f"系统会自动拦截你的检索请求并为你提供相关法条/证据，之后你可以继续进行辩论。\n\n 目前的辩论进展如下："
        RL_history = [
            {"role": "system", "content": init}
        ]
        return RL_history

    
    def build_static(self):
        init = f"你是本案的被告律师，请根据以下事实为辩方进行辩论：\n{self.fact}\n\n"
        "请对原告律师的发言进行有针对性的反驳与辩护。"
        "目前的辩论进展如下："
        static_history = [
            {"role": "system", "content": init}
        ]
        return static_history

    def update_token_log(self, token, type):
        if type == 'RL':
            self.token_log.append({"RL" : token, "Turn": self.turn})
        elif type == 'RAG':
            self.token_log.append({"RAG" : token, "Turn": self.turn})
        else:
            self.token_log.append({"Opponent" : token, "Turn": self.turn})


    def update_prompt(self, type):
        if type == "RL":
            print(f"RAW MODEL OUTPUT:\n{repr(self.rl_response)}")
            self.rl_prompt.append({"role": "assistant", "content": self.rl_response})
            self.static_prompt.append({"role": "user", "content": self.rl_response})
        else:
            self.rl_prompt.append({"role": "user", "content": self.static_response})
            self.static_prompt.append({"role": "assistant", "content": self.static_response})

    def set_RL_response(self, res):
        self.rl_response = res     
        

    def set_static_response(self):
        self.static_response = utils.chat(
            system = "你是本案的被告律师", 
            model = "deepseek/deepseek-chat", 
            user=self.rl_prompt, 
            temperature = 0.0)

    def get_static_response(self):
        return self.static_response


    def add_milestone(self, milestone):
        self.achieved_milestones.append((self.turn, milestone))
        self.achieved_set.add(milestone)
        if set(self.milestones).issubset(self.achieved_set) or self.milestones == self.achieved_set:
            self.done = True
    
    # Note, need to move the process milestone shit to here, since episode ending is conditional to it as well.
    # Note, need to make sure that it still transfers to the rewards section.

    def rl_to_string(self):
        string = ""
        for i in self.rl_prompt:
            if i["role"] == "assistant":
                string += i["content"]

        return string

    def transcript_to_string(self):
        string = "Fact: \n"
        string += self.fact + "Transcript: \n"

        for i in self.rl_prompt:
            if i["role"] == "assistant":
                string += "Agent: " + i["content"] + "\n"
            elif i["role"] == "user":
                string += "Opponent: " + i["content"] + "\n"

        return string

    def law_checker(self):
        translated_text = cn2an.transform(self.rl_response)
        print("CHECKING LAW:")
        law_numbers = re.findall(r'\d+', self.law)
        if law_numbers[0] in translated_text:
            print("LAW FOUND")
            self.add_milestone(self.law)
            self.law_check = True
        


    def get_finished(self):
        return self.done

    

# convert to tokens when added to thingo            


    

class Episode():

    def __init__(self, fact, milestones, law, size, max_turns, tokeniser, semantic_model, device):
        self.device = device
        self.size = size
        self.fact = fact
        self.milestones = ast.literal_eval(milestones)
        self.law = law
        self.threshold = 0.70
        self.max_turns = max_turns
        self.trajectories = [Trajectory(self.fact, self.milestones, t ,self.law) for t in range(size)]
        self.active_trajectories = self.trajectories
        self.semantic_model = semantic_model
        self.process_embeddings = self.semantic_model.encode(
                self.milestones, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
        )
        self.law = law

        self.tokeniser = tokeniser

        self.rag_db = utils.LawRetriever("law.json", self.device)

    def sentence_split(self, text: str):
        split = re.split(r"[。！？\n；]", text)
        return [s.strip() for s in split if len(s.strip())> 4]

    """
    Evaluates Process milestones using cosine simulation of encoded text vectors. Rewards for the partic
    
    """
    def batch_milestone_checker(self):
        sentences = []
        index = []

        rag_queries = []
        # for each existing trajectory of the episode
        for n, t in enumerate(self.active_trajectories):
            text = t.rl_response
            split = self.sentence_split(text)
            for s in split:
                sentences.append(s)
                index.append(n)
        

        if not sentences:
            return
            

        sentence_embeddings = self.semantic_model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarity_matrix = util.cos_sim(sentence_embeddings, self.process_embeddings)

        for s, i in enumerate(index):

            trajectory = self.active_trajectories[i]
            saved_milestones = trajectory.achieved_set

            sim = similarity_matrix[s]
            for j, k in enumerate(self.milestones):
                if k in saved_milestones:
                    continue

                similarity = sim[j].item()
                if similarity >= self.threshold:
                    trajectory.add_milestone(k)
                    


    def batch_trajectory_rollout(self, model):
        

        for turn in range(self.max_turns):
            if not self.active_trajectories:
                break
                

            # format trajectory prompts for Qwen model compatability

            formatted_prompts = []
            for t in self.active_trajectories:

                formatted_prompts.append(self.tokeniser.apply_chat_template(t.rl_prompt, tokenize = False, add_generation_prompt=True, enable_thinking=False ))

            inputs = self.tokeniser(formatted_prompts, 
                return_tensors = "pt", 
                padding=True,
                truncation=True,        # <--- Prevents input tokens from exceeding max length
                max_length=8192
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokeniser.pad_token_id
            )
            
            prompt_len = inputs["input_ids"].shape[1]

            # process outputs of trajectories
            traj_query = []
            for i, t in enumerate(self.active_trajectories):
                output_tokens = outputs[i, prompt_len:].tolist()
                tokens = [tok for tok in output_tokens if tok != self.tokeniser.pad_token_id]
                output_text = self.tokeniser.decode(tokens, skip_special_tokens=True)

                # Update the latest RL agent response
                t.set_RL_response(output_text)
                # Add it to the latest response string
                t.update_prompt("RL")
                # Update token log
                t.update_token_log(tokens, "RL")

                q = self.rag_db.extract_search_query(output_text)
                if q:
                    traj_query.append((t, q))

            if traj_query:
                q_traj, query = zip(*traj_query)
                retrieved_laws = self.rag_db.group_search(query)

                for t, l in zip(q_traj, retrieved_laws):
                    self.inject_rag_context(t, l)

                rag_prompts = []
                for t in q_traj:
                    rag_prompts.append(self.tokeniser.apply_chat_template(t.rl_prompt, tokenize=False, add_generation_prompt=True))

                rag_inputs = self.tokeniser(
                    rag_prompts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=8192 - 250
                ).to(self.device)
                

                # Generate another output addressing the retrieved RAG laws
                with torch.no_grad():
                    rag_outputs = model.generate(
                        **rag_inputs,
                        max_new_tokens = 250,
                        temperature = 0.7,
                        pad_token_id=self.tokeniser.pad_token_id
                )

                # No mask for later training

                prompt_len_2 = rag_inputs["input_ids"].shape[1]

                for i, t in enumerate(q_traj):
                    output_tokens_2 = rag_outputs[i, prompt_len_2:].tolist()
                    tokens_2 = [tok for tok in output_tokens if tok != self.tokeniser.pad_token_id]
                    output_text_2 = self.tokeniser.decode(tokens, skip_special_tokens=True)

                    # Update the latest RL agent response
                    t.set_RL_response(output_text_2)
                    # Add it to the latest response string
                    t.update_prompt("RL")
                    # Update token log
                    t.update_token_log(tokens_2, "RL")

                    # immediately check if the response gets the correct law down
                    if t.law_check == False:
                        t.law_checker()
                
            

            self.batch_milestone_checker()

            self.active_trajectories = [t for t in self.active_trajectories if not t.get_finished() or not t.law_check]

            if not self.active_trajectories: 
                break
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.active_trajectories)) as executor:
                list(executor.map(lambda t: (t.set_static_response()), self.active_trajectories))

                res2 = list(executor.map(lambda t: (t, t.get_static_response()), self.active_trajectories))


            for t, r in res2:
                t.update_prompt("static")
                static_tokens = self.tokeniser.encode(r, add_special_tokens=False)
                t.update_token_log(static_tokens, "static")
        
            for t in self.active_trajectories:
                if not t.done:
                    t.done = True
                t.turn += 1
    def inject_rag_context(self, trajectory, retrieved):
        database_result = f"[Database Result]:\n {retrieved}"

        trajectory.rl_prompt.append({"role": "system", "content": database_result})

        trajectory.update_token_log(self.tokeniser.encode(database_result, add_special_tokens=False), type="RAG")