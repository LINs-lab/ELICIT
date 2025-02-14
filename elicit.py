import torch
import torch.nn.functional as F
import numpy as np
from cuml.manifold import TSNE as cumlTSNE
import cupy as cp
import faiss
from rank_bm25 import BM25Okapi
from utils.retrieval_utils import *
from utils.model_utils import *
from utils.data_utils import *
from utils.eval_utils import *
from utils.retrieval_utils import *

import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from collections import defaultdict
from utils.model_utils import seed_everything
from train_retriever import *
import re
import argparse
import pandas as pd
import seaborn as sns
from baukit import TraceDict

MAX_NEW_TOKENS = 256

def test_answer(pred_str, ans, natural_prompt):
    try:
        template = natural_prompt.split(" {input}")[0]
        first_part = pred_str.split(f"\n{template}")[0]
    except:
        first_part = pred_str
    numbers = re.findall(r'\d+(?:\.\d+)?(?:/\d+)?(?:,\d{3})*(?:\.\d+)?', first_part)
    return (numbers[-1], 1) if numbers and numbers[-1].strip() == find_answer(ans).strip() else (None, 0)

class StateDB:
    def __init__(self, state_dir, shots, model, tokenizer, model_config, model_path, weight_fv, group, recollect):
        # read state soup
        print("Loading States")
        saved_states = load_state_soups(state_dir, shots, weight_fv, group, recollect)
        
        self.states = torch.stack([state["state"] for state in saved_states], dim=0)
        self.labels = [state["label"] for state in saved_states]
        self.icl_prompts = [state["icl_prompt"] for state in saved_states]
        self.best_layers = [state["best_layer"] for state in saved_states]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = self.states.to(self.device)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.retriever = PromptRetriever(model_path, "cuda")
        self.group = group
        
        # # Initialize clusters for cluster_query
        # self.n_clusters = 8  # You can adjust this number
        # self._initialize_clusters()

    def query(self, query_text, query_emb, method="cosine", k=10, threshold=1.1, layer=None, dataset_name=None, soft=False, force=False):
        # directly get states
        if dataset_name is not None:
            
            x =  list(filter(lambda x: dataset_name in x[1], enumerate(self.labels)))
            indices, chosen_labels = zip(*x)
            chosen_states = [self.states[idx] for idx in indices]
            chosen_best_layers = [self.best_layers[idx] for idx in indices]
            return False, chosen_states, chosen_labels, chosen_best_layers
        query_emb = query_emb.to(self.device)
        if method == 'cosine':
            similarities = self.cosine_similarity_query(query_emb, layer=layer)
        elif method == 'tsne':
            similarities = self.tsne_query(query_emb, layer=layer)
        elif method == 'distance':
            similarities = self.distance_query(query_emb, layer=layer)
        elif method == "retriever":
            similarities = self.retrieve_query(query_text)
        else:
            raise ValueError("Invalid query method. Choose 'cosine', 'tsne', 'distance', or 'retriever'.")
        # if self.group: breakpoint()
        sorted_indices, sorted_similarities = self.analyze_similarities(similarities)
        print("Threshold: ",threshold)
        indices = torch.where(sorted_similarities >= threshold)[0]
        index = indices[-1].item() if indices.numel() > 0 else -1
        # greedy
        if index > k-1 or force == True:
            index = k-1 
        if soft:
            if index != -1:
                index = k-1
            
        print("Top Similarities: ", sorted_similarities[:k].cpu().tolist())
        top_k_indices = sorted_indices[:index+1]
        if index != -1:
            print("index: ", index)
            chosen_labels = [self.labels[idx] for idx in top_k_indices]
            chosen_best_layers = [self.best_layers[idx] for idx in top_k_indices]   
            chosen_states = [self.states[idx].squeeze() for idx in top_k_indices]
            return False,  chosen_states, chosen_labels, chosen_best_layers
        else:
            top_k_labels = [self.labels[idx] for idx in sorted_indices[:k]]
            top_k_best_layers = [self.best_layers[idx] for idx in sorted_indices[:k]]
            top_k_states = [self.states[idx].squeeze() for idx in sorted_indices[:k]]
            return True, top_k_states, top_k_labels, top_k_best_layers
        
    
    def cosine_similarity_query(self, query_emb, layer=None):
        if layer is None:
            query_emb = query_emb.unsqueeze(0)
            similarities = F.cosine_similarity(query_emb, self.states, dim=-1)
            similarities = similarities.mean(dim=-1)
            
        else:
            
            query_emb = query_emb[layer].unsqueeze(0)
            similarities = F.cosine_similarity(query_emb, self.states[:, layer,:].squeeze(), dim=-1)
       
        return similarities



    def tsne_query(self, query_emb, perplexity=30, n_components=2, layer=None):
        states = self.states.clone()

        if layer is None:
            state_soup_2d = states.view(states.shape[0], -1)
            combined = torch.cat([query_emb.view(1, -1), state_soup_2d], dim=0)
        else:
            state_soup_2d = states[:, layer, :].squeeze()
            combined = torch.cat([query_emb[layer,:].view(1, -1), state_soup_2d], dim=0)

        combined_gpu = cp.asarray(combined.float().cpu().numpy())

        tsne = cumlTSNE(n_components=n_components, perplexity=perplexity, n_iter=1000)
        embedded = tsne.fit_transform(combined_gpu)

        embedded = torch.tensor(cp.asnumpy(embedded), dtype=torch.float32, device=self.device)
        query_embedded = embedded[0].unsqueeze(0)  # [1, 2]
        states_embedded = embedded[1:]  # [200, 2]

        distances = torch.cdist(query_embedded, states_embedded, p=2.0)  # [1, 200]

        similarities = 1 / (1 + distances)

        return similarities.squeeze()

    def distance_query(self, query_emb, layer=None):
        if layer is None:
            query_emb = query_emb.view(1, -1)
            query_emb = query_emb.float()
            distances = torch.cdist(query_emb, self.states.view(self.states.shape[0], -1), p=2)
        else:
            query_emb = query_emb[layer, :].view(1, -1)
            query_emb = query_emb.float()
            distances = torch.cdist(query_emb, self.states[:, layer,:].squeeze(), p=2)
        
        similarities = 1 / (1 + distances)

        return similarities.squeeze()

    def retrieve_query(self, query_text):
        scores = self.retriever.retrieve(query_text, self.icl_prompts)
        prompts, similarities, indices = zip(*scores)
        similarities = torch.tensor(list(similarities))
    
        return similarities

    def _initialize_clusters(self):
        states_np = self.states.cpu().numpy().reshape(len(self.states), -1).astype(np.float32)
        d = states_np.shape[1]
        self.kmeans = faiss.Kmeans(d, self.n_clusters, niter=20, gpu=True)
        self.kmeans.train(states_np)
        self.centroids = torch.tensor(self.kmeans.centroids, device=self.device)

    def cluster_and_visualize(self, n_clusters=5, method='tsne'):
        # Convert states to numpy array
        states_np = self.states.cpu().numpy().reshape(len(self.states), -1)

        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

        reduced_states = reducer.fit_transform(states_np)

        # Prepare labels
        labels = ["_".join(l.split("_")[:-1]) for l in self.labels]
       
        unique_labels = list(set(labels))

        # Create a color map for unique labels
        color_map = plt.cm.get_cmap('viridis')
        colors = [color_map(i) for i in np.linspace(0, 1, len(unique_labels))]
        label_to_color = dict(zip(unique_labels, colors))

        # Visualization
        plt.figure(figsize=(16, 12))
        
        # Plot points
        for label in unique_labels:
            mask = np.array(labels) == label
            plt.scatter(
                reduced_states[mask, 0],
                reduced_states[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.7
            )

        plt.title(f'Clustered States Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Add one label per unique task type
        label_positions = defaultdict(list)
        for i, label in enumerate(labels):
            label_positions[label].append((reduced_states[i, 0], reduced_states[i, 1]))
        
        for label, positions in label_positions.items():
            mean_pos = np.mean(positions, axis=0)
            plt.annotate(label, mean_pos, fontsize=10, alpha=0.8, fontweight='bold')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'clustered_states_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved as clustered_states_{method}.png")


    def analyze_similarities(self, similarities):
        
        similarities = similarities.cuda()
        sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)

        # mean_similarity = torch.mean(similarities)
        # std_similarity = torch.std(similarities)
        # # threshold =  mean_similarity + 1.5*std_similarity
        # # print("Mean + Std: ", threshold)
        # confidence_scores  = (similarities - mean_similarity) / std_similarity
        # # confidence_scores = z_scores[sorted_indices[:k]]

        return sorted_indices, sorted_similarities


class Retrieve_Evaluator:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer, self.model_config = self.load_model()
        self.dataset = self.load_dataset()
        self.state_db = self.load_state_db()
        self.answer_templates = ["Answer:", "The answer is", "A:"]
        
    def load_model(self):
        print("Loading Model and Tokenizer...")
        return load_model_and_tokenizer(self.args.model_name, device=self.args.device)
    
    def load_dataset(self):
        if self.args.specific_data == None:
            print("Loading Dataset...")
            dataset = load_dataset(self.args.dataset_name, root_data_dir=self.args.root_data_dir, test_split=self.args.test_split, seed=self.args.seed)
            # dataset[self.args.dataset_split].raw_data = dataset[self.args.dataset_split].raw_data[:self.args.test_samples]
            test_nums = min(len(dataset[self.args.dataset_split].raw_data), self.args.test_samples)
            dataset[self.args.dataset_split].raw_data = dataset[args.dataset_split].raw_data.sample(n=test_nums, random_state=self.args.seed)
            # dataset['valid'].raw_data = dataset['valid'].raw_data[:self.args.test_samples]
        else:
            with open("results/llama3-8b-aug31/gsm8k/llama3-8b_intervene_zs_layer22_ori1.0_fv2.0_16shots_generation.json", "r") as f:
                data = json.load(f)
            dataset = {"valid":[]}
            for item in data[:10]:
                input = item["input"][0].split("Q: ")[1]
                input = input.split("\nLet's think step by step. A:")[0]
                dataset["valid"].append({
                    "input": input,
                    "output": item["label"]
                    })
            
        return dataset
    
    def load_state_db(self):
        # model_path = 'new_prompt_classifier_model.pth' if self.args.dataset_name != "gsm8k" else 'new_prompt_classifier_model_nov.pth'
        model_path = self.args.retrieval_model
        print(model_path)
        return StateDB(self.args.state_dir, shots=self.args.shots, model=self.model, tokenizer=self.tokenizer, 
                       model_config=self.model_config, model_path=model_path, weight_fv=self.args.weight_fv, group=self.args.group, recollect=self.args.recollect_state)
    
    def evaluate(self):
        self.process_icl_prompt_to_shot()
        if self.args.single_layer_mode:
            return self.evaluate_single_layer()
        else:
            return self.evaluate_all_layers()
    
    def evaluate_single_layer(self):
        eval_results = []
        all_thresholds = self.load_thresholds()
        templates = self.load_templates()
        nshots = [self.args.shots_num]

        for i, template in tqdm(enumerate(templates)):
            if not self.args.local:
                results = self.retrieve_and_intervene_global("test", template, nshots, all_thresholds)
            else:
                results = self.retrieve_and_intervene(self.args.dataset_split, template, nshots, all_thresholds, i)
                
            eval_results.append(self.process_all_layers_results(results, self.args.dataset_split, nshots))
        return eval_results
    
    def evaluate_all_layers(self):
        eval_results = []
        templates = self.load_templates()
        nshots = [1, 2, 4]
        for template in tqdm(templates):
            if self.args.local:
                # val_results = self.retrieve_and_intervene_all_layers("valid", template)
                test_results = self.retrieve_and_intervene_all_layers("test", template, nshots)
            else:
                # val_results = self.retrieve_and_intervene_all_layers_global("valid", template)
                test_results = self.retrieve_and_intervene_all_layers_global("test", template, nshots)
            eval_results.append(self.process_all_layers_results(test_results, "test", nshots))

        return eval_results

    def init_results(self, dataset_split, nshots, all_layers=False):
        results = {
            'token_lengths': [],
            'clean_time_list': [],
            'intervene_time_list': [], #if all_layers else defaultdict(list) ,
            'retrieve_time_list':[],
            'retrieve_acc': [],
            'intervene_results': [],# if all_layers else defaultdict(list),
            'clean_acc_list': [],
            "output": [],
            "chosen_state_num": []
        }
        if dataset_split == "test" and nshots:
            for shot in nshots:
                results[f'{shot}shot_results'] = defaultdict(list)
            results["bm25_results"] = defaultdict(list)
        return results

    def retrieve_and_intervene(self, dataset_split, template=None, nshots=None, all_thresholds=None, index = 0):
        # retrieve 1 state for each sample and intervene single layer
        results = self.init_results(dataset_split, nshots)
        
        for i in tqdm(range(len(self.dataset[dataset_split]))):
            word_pairs_test = self.dataset[dataset_split][i]
            query_text, target = self.prepare_query_text(word_pairs_test, template)
            
            inputs, target_token_id = self.prepare_model_inputs(query_text, target)
            results['token_lengths'].append(len(inputs.input_ids.squeeze()) - 1)
            clean_output, clean_time = self.perform_clean_inference(inputs)
            results["clean_time_list"].append(clean_time)
            if self.args.generate_str == True:
                pred_str, clean_acc = test_answer(clean_output, target, template)
            else:
                clean_acc = int((torch.argsort(clean_output.squeeze(), descending=True)[0] == target_token_id[0]).item())
            results['clean_acc_list'].append(clean_acc)

            # record retrieve time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            query_activation = calculate_natural_text_activations(query_text, self.model, self.tokenizer, self.model_config)

            
            if self.args.retrieve_method == "retriever":
                disable, task_vector, icl_best_layer, task_label = self.retrieve(query_text, query_activation, retrieve_layer=None, all_thresholds=all_thresholds)
                end_time.record()
                torch.cuda.synchronize()
                retrieve_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                results['retrieve_time_list'].append(retrieve_time)
                results["icl_best_layer"] = icl_best_layer
                edit_layer = icl_best_layer
                if self.args.skip == False:
                    self.perform_intervene(disable, icl_best_layer, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results, query_text, clean_output, target, template)
            else:
                # retrieve 
                for edit_layer in self.get_edit_layers():
                    disable, task_vector, icl_best_layer, task_label = self.retrieve(query_text, query_activation, retrieve_layer=edit_layer, all_thresholds=all_thresholds)
                    if self.args.skip == False:
                        self.perform_intervene(disable, edit_layer, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results)
            
            if self.args.calculate_nshots_bm25 == True:
                if self.args.use_template == False and index == 0:
                    print("Calculate nshots and bm25")
                    if disable:
                        self.bm25_retriever(word_pairs_test["input"], results, target_token_id, target, tv=None, task_label=None, edit_layer=None, template=template, all_thresholds=all_thresholds)
                        self.perform_nshot_inference(word_pairs_test, nshots, results, template, all_thresholds, edit_layer=None, task_label=None, tv=None)
                    else:
                        self.bm25_retriever(word_pairs_test["input"], results, target_token_id, target, tv=task_vector, task_label=task_label, edit_layer=edit_layer, template=template, all_thresholds=all_thresholds)
                        self.perform_nshot_inference(word_pairs_test, nshots, results, template, all_thresholds, edit_layer=edit_layer,task_label=task_label,tv=task_vector)
                elif self.args.use_template == True:
                    print("Calculate nshots and bm25")
                    if disable:
                        self.bm25_retriever(word_pairs_test["input"], results, target_token_id, target, tv=None, task_label=None, edit_layer=None, template=template, all_thresholds=all_thresholds)
                        self.perform_nshot_inference(word_pairs_test, nshots, results, template, all_thresholds, edit_layer=None, task_label=None, tv=None)
                    else:
                        self.bm25_retriever(word_pairs_test["input"], results, target_token_id, target, tv=task_vector, task_label=task_label, edit_layer=edit_layer, template=template, all_thresholds=all_thresholds)
                        self.perform_nshot_inference(word_pairs_test, nshots, results, template, all_thresholds, edit_layer=edit_layer,task_label=task_label,tv=task_vector)
        return results

    def retrieve_and_intervene_global(self, dataset_split, template=None, nshots=None, all_thresholds=None):
        results = self.init_results(dataset_split, nshots)

        word_pairs_test = self.dataset[args.dataset_split][np.random.choice(len(self.dataset[args.dataset_split]), 1, replace=False)]
        query_text, target = self.prepare_query_text(word_pairs_test, template)
        print(f"\n{'='*20} Query Text {'='*20}\n{query_text}\n{'='*50}")
        query_activation = calculate_natural_text_activations(query_text, self.model, self.tokenizer, self.model_config)

        tvs = nested_dict()
        if self.args.retrieve_method == "retriever":
            disable, task_vector, icl_best_layer, task_label = self.retrieve(query_text, query_activation, retrieve_layer=None, all_thresholds=all_thresholds)
            results["icl_best_layer"] = icl_best_layer
            tvs[icl_best_layer]["tv"] = task_vector
            tvs[icl_best_layer]["disable"] = disable
            tvs[icl_best_layer]["task_label"] = task_label
    
               
        else:
            for edit_layer in self.get_edit_layers():
                disable, task_vector, icl_best_layer, task_label = self.retrieve(query_text, query_activation, retrieve_layer=edit_layer, all_thresholds=all_thresholds)
                tvs[edit_layer]["tv"] = task_vector
                tvs[edit_layer]["disable"] = disable
                tvs[edit_layer]["task_label"] = task_label
            

        for i in tqdm(range(len(self.dataset[dataset_split]))):
            word_pairs_test = self.dataset[dataset_split][i]
            text, target = self.prepare_query_text(word_pairs_test, template)
            print(f"\n{'='*20} Test Sample {'='*20}\n{text}\n{'='*50}")
            
            inputs, target_token_id = self.prepare_model_inputs(text, target)
            results['token_lengths'].append(len(inputs.input_ids.squeeze()) - 1)
            clean_output, clean_time = self.perform_clean_inference(inputs)
            results["clean_time_list"].append(clean_time)
            clean_acc = int((torch.argsort(clean_output.squeeze(), descending=True)[0] == target_token_id[0]).item())
            results['clean_acc_list'].append(clean_acc)

            if self.args.retrieve_method == "retriever":
                disable = tvs[icl_best_layer]["disable"]
                task_vector = tvs[icl_best_layer]["tv"]
                task_label = tvs[icl_best_layer]["task_label"]
                self.perform_intervene(disable, icl_best_layer, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results, text, clean_output, target)
            else:
                # retrieve 
                for edit_layer in self.get_edit_layers():
                    disable = tvs[icl_best_layer]["disable"]
                    task_vector = tvs[icl_best_layer]["tv"]
                    task_label = tvs[icl_best_layer]["task_label"]
    
                    self.perform_intervene(disable, edit_layer, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results)
            
            if dataset_split == "test" and nshots:
                self.perform_nshot_inference(word_pairs_test, nshots, results, template)

        return results

    def retrieve_and_intervene_all_layers(self, dataset_split, template=None, nshots=None):
        # Implementation for all layers intervention
        results = self.init_results(dataset_split, nshots, True)
        
        for i in tqdm(range(len(self.dataset[dataset_split]))):
            word_pairs_test = self.dataset[dataset_split][i]
            query_text, target = self.prepare_query_text(word_pairs_test, template)
         
            
            inputs, target_token_id = self.prepare_model_inputs(query_text, target)
            results['token_lengths'].append(len(inputs.input_ids.squeeze()) - 1)
            clean_output, clean_time = self.perform_clean_inference(inputs)
            results["clean_time_list"].append(clean_time)
            clean_acc = int((torch.argsort(clean_output.squeeze(), descending=True)[0] == target_token_id[0]).item())
            results['clean_acc_list'].append(clean_acc)

            query_activation = calculate_natural_text_activations(query_text, self.model, self.tokenizer, self.model_config)

            disable, task_vector, _, task_label = self.retrieve(query_text, query_activation, retrieve_layer=None)

            edit_layers = list(range(self.model_config["n_layers"]))
            self.perform_intervene(disable, edit_layers, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results, query_text, clean_output, target)
           
            
            if dataset_split == "test" and nshots:
                self.perform_nshot_inference(word_pairs_test, nshots, results, template)

        return results

    def retrieve_and_intervene_all_layers_global(self, dataset_split, template=None, nshots=None):
        results = self.init_results(dataset_split, nshots, True)
        
        word_pairs_test = self.dataset[args.dataset_split][np.random.choice(len(self.dataset[args.dataset_split]), 1, replace=False)]
        query_text, target = self.prepare_query_text(word_pairs_test, template)
        print(f"\n{'='*20} Query Text {'='*20}\n{query_text}\n{'='*50}")
        query_activation = calculate_natural_text_activations(query_text, self.model, self.tokenizer, self.model_config)

        disable, task_vector, icl_best_layer, task_label = self.retrieve(query_text, query_activation, retrieve_layer=None, all_thresholds=None)
        
    
        edit_layers = list(range(self.model_config["n_layers"]))
        for i in tqdm(range(len(self.dataset[dataset_split]))):
            word_pairs_test = self.dataset[dataset_split][i]
            query_text, target = self.prepare_query_text(word_pairs_test, template)
            
            inputs, target_token_id = self.prepare_model_inputs(query_text, target)
            results['token_lengths'].append(len(inputs.input_ids.squeeze()) - 1)
            clean_output, clean_time = self.perform_clean_inference(inputs)
            results["clean_time_list"].append(clean_time)
            clean_acc = int((torch.argsort(clean_output.squeeze(), descending=True)[0] == target_token_id[0]).item())
            results['clean_acc_list'].append(clean_acc)

            self.perform_intervene(disable, edit_layers, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results, query_text, clean_output, target)
           
            
            if dataset_split == "test" and nshots:
                self.perform_nshot_inference(word_pairs_test, nshots, results, template)

        return results

    def prepare_query_text(self, word_pairs_test, template):
        word_pairs = {'input': [], 'output': []}
        if template:
            prefixes = {"instructions": '', "input": '', "output": ''}
            separators = {"instructions": '', "input": '', "output": ''}
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=False, 
                                                    shuffle_labels=False, template=template, cot=False, 
                                                    prefixes=prefixes, separators=separators, prepend_space=False)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=False, 
                                                    shuffle_labels=False, cot=False, prepend_space=True)
        target = prompt_data["query_target"]["output"]
        return create_prompt(prompt_data, tokenizer=self.tokenizer if "instruct" in self.args.model_name else None), target

    def prepare_model_inputs(self, query_text, target):
        inputs = self.tokenizer([query_text], return_tensors="pt").to(self.model.device)
        target_token_id = get_answer_id(query_text, target, self.tokenizer)
        return inputs, target_token_id

    def perform_clean_inference(self, inputs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        if self.args.generate_str == True:
            start_time.record()
            output = self.model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
            end_time.record()
            torch.cuda.synchronize()
            clean_output = self.tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:])
        else:
            start_time.record()
            clean_output = self.model(**inputs).logits[:, -1, :]
        
        end_time.record()
        torch.cuda.synchronize()
        clean_time = start_time.elapsed_time(end_time) / 1000
        
        return clean_output, clean_time

    def retrieve(self, query_text, query_activation, retrieve_layer=None, all_thresholds=None):
        self.threshold = (all_thresholds[retrieve_layer] if type(all_thresholds) is dict 
                         else all_thresholds if all_thresholds is not None 
                         else self.args.threshold)
        
        disable, tv, task_label, icl_best_layers = self.state_db.query(
            query_text, 
            query_activation,
            method=self.args.retrieve_method,
            k=self.args.k,
            layer=retrieve_layer,
            threshold=self.threshold,
            soft=self.args.soft,
            force=self.args.force
        )
        
        task_vector = torch.mean(torch.stack(tv), dim=0)
        icl_best_layer = max(Counter(icl_best_layers).items(), key=lambda x: x[1])[0]
        return disable, task_vector, icl_best_layer, task_label
    
    def perform_intervene(self, disable, edit_layer, inputs, target_token_id, task_vector, task_label, clean_acc, clean_time, results, query_text, clean_output, target, template):
        if disable:
            return self.handle_disabled_intervention(results, clean_acc, clean_time, edit_layer, query_text, clean_output, target)
        else:
            return self.handle_enabled_intervention(results, inputs, target_token_id, task_vector, task_label, edit_layer, query_text, clean_output, target, template)

    def handle_disabled_intervention(self, results, clean_acc, clean_time, edit_layer, query_text, clean_output, target):
        print("Disable intervention ...")
        results['retrieve_acc'].append(0)
        
        results['intervene_results'].append(clean_acc)
        results['intervene_time_list'].append(clean_time)
        
        # if type(edit_layer) is list:
        # else:
        #     results['intervene_results'][edit_layer].append(clean_acc)
        #     results['intervene_time_list'][edit_layer].append(clean_time)
        results["chosen_state_num"].append(0)
        self.record_output(results, query_text, clean_output, clean_output, [], target, [])
       
        return results

    def handle_enabled_intervention(self, results, inputs, target_token_id, tv, task_label, edit_layer, query_text, clean_output, target, template):
        # NOTE: math and mathqa
        results['retrieve_acc'].extend([1 if self.args.dataset_name in tl else 0 for tl in task_label])
        results["chosen_state_num"].append(len(tv))
        
        intervention_fn, intervene_layers = self.create_intervention_function(edit_layer, tv)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        if self.args.generate_str == True:
            print("Generation Task")
            start_time.record()
            with TraceDict(self.model, layers=self.model_config['layer_hook_names'], edit_output=intervention_fn):     
                intervention_output = self.model.generate(**inputs, max_new_tokens = 1,  do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
            intervention_output = self.model.generate(intervention_output, max_new_tokens=MAX_NEW_TOKENS-1, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
            end_time.record()
            torch.cuda.synchronize()
            intervention_output = self.tokenizer.decode(intervention_output.squeeze()[inputs.input_ids.shape[1]:])
            pred_str, intervene_acc = test_answer(intervention_output, target, template)
            
            intervene_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            with TraceDict(self.model, layers=self.model_config["layer_hook_names"], edit_output=intervention_fn): 
                start_time.record()
                intervention_output = self.model(**inputs).logits[:, -1, :]
                end_time.record()
                
                torch.cuda.synchronize()
                intervene_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                intervene_acc = int((torch.argsort(intervention_output.squeeze(), descending=True)[0] == target_token_id[0]).item())
        results['intervene_time_list'].append(intervene_time)
        results['intervene_results'].append(intervene_acc)
        # if type(edit_layer) is list:
        # else:
        #     results['intervene_results'][edit_layer].append(intervene_acc)
        self.record_output(results, query_text, clean_output, intervention_output, task_label, target, intervene_layers)
    
        return results

    def create_intervention_function(self, edit_layer, tv):
        if type(edit_layer) is list:
            print("Edit Layer: ", edit_layer)
            return add_function_vector(edit_layer, tv, device=self.model.device, idx=-1, weight_fv=args.weight_fv, weight_ori=args.weight_ori, norm=True), edit_layer
        else:
            if self.args.retrieve_method == "retriever" and self.args.n_layers is not None:
                temp_tv = tv
                intervene_layers = list(range(max(0, edit_layer-self.args.n_layers), min(edit_layer+self.args.n_layers + 1, self.model_config["n_layers"])))
                factor = len(intervene_layers)
                norm = False
            else: 
                temp_tv = tv[edit_layer]
                intervene_layers = edit_layer
                factor = 1
                norm = False
        print("Edit Layer: ", intervene_layers)
        return add_function_vector(intervene_layers, temp_tv, device=self.model.device, idx=-1, 
                                   weight_fv=self.args.weight_fv / factor, weight_ori=self.args.weight_ori, norm=norm), intervene_layers

    def record_output(self, results, query_text, clean_output, intervention_output, task_label, target, intervene_layers):
        # print(clean_output, intervention_output)
        results["output"].append({
            "threshold": self.threshold,
            "input": query_text,
            "clean_output": self.tokenizer.decode((torch.argsort(clean_output.squeeze(), descending=True)[0])) if not self.args.generate_str else clean_output, 
            "intervene_output": self.tokenizer.decode((torch.argsort(intervention_output.squeeze(), descending=True)[0])) if not self.args.generate_str else intervention_output, 
            "label": target,
            "chosen_states": task_label,
            "intervene_layers": intervene_layers
        })

    def process_icl_prompt_to_shot(self):
        shots_pool = []
        for icl_prompt in self.state_db.icl_prompts:
            parts = icl_prompt.split("\n\nQ: ")[:-1]
            assert len(parts) == 16
            for part in parts:
                if part == parts[0]:
                    q, a = part[3:].split("\nA:")
                    shots_pool.append({'q': q, 'a': a})
                else:
                    q, a = part.split("\nA:")
                    shots_pool.append({'q': q, 'a': a})
                # p = part.split("A: ")
                # shots_pool.append({"input":p[0].strip("\n"), "output":p[1].strip("\n\n")})
        # print(16*10*len(self.state_db.icl_prompts))
        self.shots_pool = shots_pool
        print("Total Shots: ", len(self.shots_pool))
        self.tokenized_shots = [doc['q'].lower().split() for doc in self.shots_pool]
        self.bm25 = BM25Okapi(self.tokenized_shots)
    
    def bm25_retriever(self, query_text, results, target_id, target, tv=None, task_label=None, edit_layer=None, template=None, all_thresholds=None):
        print("BM25 retrieving")
        tokenized_query = query_text.strip().lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n = self.bm25.get_top_n(tokenized_query, self.shots_pool, n=self.args.shots_num)
        
        if self.args.use_template:
            test_sample = ''
            for shot in top_n:
                test_sample += template.format(input=shot["q"])
                test_sample += shot['a']
                test_sample += "\n\n"
            test_sample += template.format(input=query_text)
        else:
            temp_template = "Q: {input}\nA:{output}"
            processed_topn = [temp_template.format(input=t['q'],output=t['a']) for t in top_n]
            test_sample = "\n\n".join(processed_topn) + "\n\nQ: " + query_text + "\nA:"
            
        if self.args.debug: breakpoint()
        nshot_inputs = self.tokenizer(test_sample, return_tensors="pt").to(self.model.device)
        nshot_output, nshot_time = self.perform_clean_inference(nshot_inputs)
        if self.args.generate_str == True:
            pred_str, nshot_acc = test_answer(nshot_output, target, template)
        else:
            nshot_acc = int((torch.argsort(nshot_output.squeeze(), descending=True)[0] == target_id[0]).item())
            
        results[f'bm25_results']["length"].append(len(nshot_inputs.input_ids.squeeze()) - 1)
        results[f'bm25_results']["acc"].append(nshot_acc)
        results[f'bm25_results']["time"].append(nshot_time)

        # bm25 + ours
        disable = None
        if self.args.reretrieval == True:
            query_activation = calculate_natural_text_activations(test_sample, self.model, self.tokenizer, self.model_config)
            disable, tv, edit_layer, task_label = self.retrieve(test_sample, query_activation, retrieve_layer=None, all_thresholds=all_thresholds)
        
        if (self.args.reretrieval == True and disable == False) or (self.args.reretrieval == False and tv is not None):
        # if disable == False:
            intervention_fn, intervene_layers = self.create_intervention_function(edit_layer, tv)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            if self.args.generate_str == True:
                start_time.record()
                with TraceDict(self.model, layers=self.model_config['layer_hook_names'], edit_output=intervention_fn):     
                    intervention_output = self.model.generate(**nshot_inputs, max_new_tokens = 1,  do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
                intervention_output = self.model.generate(intervention_output, max_new_tokens=MAX_NEW_TOKENS-1, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
                end_time.record()
                torch.cuda.synchronize()
                intervention_output = self.tokenizer.decode(intervention_output.squeeze()[nshot_inputs.input_ids.shape[1]:])
                pred_str, intervene_acc = test_answer(intervention_output, target, template)
            else:
                with TraceDict(self.model, layers=self.model_config["layer_hook_names"], edit_output=intervention_fn):
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    intervention_output = self.model(**nshot_inputs).logits[:, -1, :]
                    end_time.record()
                    
                    torch.cuda.synchronize()
            
                    intervene_acc = int((torch.argsort(intervention_output.squeeze(), descending=True)[0] == target_id[0]).item())
            
            results['bm25_results']["+tv_acc"].append(intervene_acc)
            results['bm25_results']["generation"].append({
                    "threshold": self.threshold,
                    "input": query_text,
                    "clean_output":  self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                    "intervene_output": self.tokenizer.decode((torch.argsort(intervention_output.squeeze(), descending=True)[0])) if not self.args.generate_str else intervention_output, 
                    "label": target,
                    "chosen_states": task_label,
                    "intervene_layers": intervene_layers
                })
            
        else:
            results['bm25_results']["+tv_acc"].append(nshot_acc)
            results['bm25_results']["generation"].append({
                    "threshold": self.threshold,
                    "input": query_text,
                    "clean_output":  self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                    "intervene_output": self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                    "label": target,
                    "chosen_states": [],
                    "intervene_layers": []
                })
        

        

    def perform_nshot_inference(self, word_pairs_test, nshots, results, template, all_thresholds, edit_layer, task_label, tv=None):
        print("nshot inferencing")
        for shot in nshots:
            nshot_word_pairs = self.dataset['train'][np.random.choice(len(self.dataset['train']), shot, replace=False)]
            nshot_prompt_data = word_pairs_to_prompt_data(nshot_word_pairs, query_target_pair=word_pairs_test, 
                                                          prepend_bos_token=False, shuffle_labels=False, 
                                                          cot=False, prepend_space=True, template=template if self.args.use_template else None)
            
            nshot_query, nshot_target = nshot_prompt_data["query_target"]["input"].strip(), nshot_prompt_data["query_target"]["output"]
            nshot_sentence = [create_prompt(nshot_prompt_data)]
    

            nshot_target_token_id = get_answer_id(nshot_sentence[0], nshot_target, self.tokenizer)
            nshot_inputs = self.tokenizer(nshot_sentence, return_tensors="pt").to(self.model.device)
            

            if self.args.generate_str == True:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                output = self.model.generate(**nshot_inputs,max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
                end_time.record()
                torch.cuda.synchronize()
                nshot_time = start_time.elapsed_time(end_time) / 1000
                nshot_output = self.tokenizer.decode(output.squeeze()[nshot_inputs.input_ids.shape[1]:])
                pred_str, nshot_acc = test_answer(nshot_output, nshot_target, template)
            else:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                nshot_output = self.model(**nshot_inputs).logits[:, -1, :]
                end_time.record()
                
                torch.cuda.synchronize()
                nshot_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                
                nshot_acc = int((torch.argsort(nshot_output.squeeze(), descending=True)[0] == nshot_target_token_id[0]).item())
            results[f'{shot}shot_results']["length"].append(len(nshot_inputs.input_ids.squeeze()) - 1)
            results[f'{shot}shot_results']["acc"].append(nshot_acc)
            results[f'{shot}shot_results']["time"].append(nshot_time)
            
            # retrieve using nshot input
            # query_activation = calculate_natural_text_activations(nshot_sentence[0], self.model, self.tokenizer, self.model_config)

            # disable, tv, edit_layer, task_label = self.retrieve(nshot_sentence[0], query_activation, retrieve_layer=None, all_thresholds=all_thresholds)
            # fs + ours
            # if disable == False:
            if tv != None:
                intervention_fn, intervene_layers = self.create_intervention_function(edit_layer, tv)
                # intervention_fn, intervene_layers = self.create_intervention_function(icl_best_layer, task_vector)
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                if self.args.generate_str == True:
                    start_time.record()
                    with TraceDict(self.model, layers=self.model_config['layer_hook_names'], edit_output=intervention_fn):     
                        intervention_output = self.model.generate(**nshot_inputs, max_new_tokens = 1,  do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
                    intervention_output = self.model.generate(intervention_output, max_new_tokens=MAX_NEW_TOKENS-1, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=self.tokenizer.eos_token_id)
                    end_time.record()
                    torch.cuda.synchronize()
                    intervention_output = self.tokenizer.decode(intervention_output.squeeze()[nshot_inputs.input_ids.shape[1]:])
                    pred_str, intervene_acc = test_answer(intervention_output, nshot_target, template)
                else:
                    with TraceDict(self.model, layers=self.model_config["layer_hook_names"], edit_output=intervention_fn):
                        start_time = torch.cuda.Event(enable_timing=True)
                        end_time = torch.cuda.Event(enable_timing=True)
                        
                        start_time.record()
                        intervention_output = self.model(**nshot_inputs).logits[:, -1, :]
                        end_time.record()
                        
                        torch.cuda.synchronize()
                
                        intervene_acc = int((torch.argsort(intervention_output.squeeze(), descending=True)[0] == nshot_target_token_id[0]).item())
                results[f'{shot}shot_results']["+tv_acc"].append(intervene_acc)
                results[f'{shot}shot_results']["generation"].append({
                        "threshold": self.threshold,
                        "input": nshot_query,
                        "clean_output":  self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                        "intervene_output": self.tokenizer.decode((torch.argsort(intervention_output.squeeze(), descending=True)[0])) if not self.args.generate_str else intervention_output, 
                        "label": nshot_target,
                        "chosen_states": task_label,
                        "intervene_layers": intervene_layers
                    })
                
            else:
                results[f'{shot}shot_results']["+tv_acc"].append(nshot_acc)
                results[f'{shot}shot_results']["generation"].append({
                        "threshold": self.threshold,
                        "input": nshot_query,
                        "clean_output":  self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                        "intervene_output": self.tokenizer.decode((torch.argsort(nshot_output.squeeze(), descending=True)[0])) if not self.args.generate_str else nshot_output, 
                        "label": nshot_target,
                        "chosen_states": [],
                        "intervene_layers": []
                    })


    def process_all_layers_results(self, results, dataset_split, nshots=None):
        if self.args.skip:
            return {}
        
        final_results = {
            "test_zero-shot_acc": np.mean(results['clean_acc_list']),
            "test_acc": np.mean(results["intervene_results"]),
            "clean_time": sum(results['clean_time_list']),
            "intervene_time": sum(results["intervene_time_list"]),
            "retrieve_time": sum(results["retrieve_time_list"]),
            "retrieve_acc": np.mean(results['retrieve_acc']),
            "zs_lengths": np.mean(results['token_lengths']),
            "chosen_state_num": np.mean(results["chosen_state_num"]),
            "generation": results["output"]
        }

        if dataset_split == "test" and nshots:
            final_results.update(self._process_nshot_results(results, nshots))
            final_results["bm25_results"] = self._process_bm25_results(results)
        
        return final_results

    def load_thresholds(self):
        model_name = self.args.model_name.split("/")[-1]
        suffix = "natural_single" if self.args.prompt_file and self.args.single_layer_mode else "icl_single"
        
        if self.args.retrieve_method == "retriever":
            suffix = "natural" if self.args.prompt_file else "icl"
            with open(f"results/thresholds/retriever_{suffix}.json", "r")as f:
                thresholds = json.load(f)
            print("Recall: ", self.args.recall)
            return thresholds[str(self.args.recall)]["threshold"]

        elif self.args.single_layer_mode == False:
            return None
        print("Using single mode to retrieve..")
        with open(f"results/thresholds/{model_name}_{self.args.retrieve_method}_{suffix}_single_layer_{self.args.recall}recall.json", "r") as f:
            return json.load(f)

    def load_templates(self):
        if self.args.prompt_file:
            print("Loading Natural Prompts...")
            with open(self.args.prompt_file, "r") as f:
                natural_texts = json.load(f)
            if self.args.generate_str == False:
                all_templates = [template + "\n"+ self.answer_templates[i] for i in range(3) for template in natural_texts[self.args.dataset_name]]
                
            else:
                all_templates = [template + "\n"+ self.answer_templates[i] for i in range(3) for template in natural_texts[self.args.dataset_name]]
                # return [template + "\n"+ self.answer_templates[i] + " Let's think step by step." for i in range(3) for template in natural_texts[self.args.dataset_name]]
            if self.args.template_num != None:
                return all_templates[:self.args.template_num]
            else:
                return all_templates
        else:
            return [None]

    def get_edit_layers(self):
        if self.args.retrieve_method == "retriever":
            return None  
        return list(range(self.model_config["n_layers"]))

    def save_results(self, results):
        os.makedirs(self.args.save_dir, exist_ok=True)
        output_file = f"{self.args.save_dir}/{self.args.model_name.split('/')[-1]}_{self.args.dataset_name}_{self.args.retrieve_method}_{self.args.shots}shots_{self.args.weight_ori}ori_{self.args.weight_fv}fv_{self.args.recall}recall.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")

    def _process_nshot_results(self, results, nshots):
        nshot_results = {}
        for shot in nshots:
            shot_data = results[f"{shot}shot_results"]
            nshot_results[f'{shot}shot_results'] = {
                k: v if k == "generation" else 
                   sum(v) if "time" in k else
                   np.mean(v)
                for k,v in shot_data.items()
            }
        return {"nshots_results": nshot_results}

    def _process_bm25_results(self, results):
        return {
            k: v if k == "generation" else
               sum(v) if "time" in k else
               np.mean(v)
            for k,v in results["bm25_results"].items()
        }

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Evaluate retriever model")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to the natural prompts data file")
    parser.add_argument("--state_dir", type=str, default="results/states/llama2-7b/state_soup.pt", help="Path to the state soup file")
    parser.add_argument("--model_name", type=str, default="../../../llm-analysis/llm-analysis/spin_models/llama2-7b", help="Path to the LLaMA model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")  
    parser.add_argument("--retrieval_model", type=str, default="new_prompt_classifier_model.pth")
    parser.add_argument("--k", type=int, default=5, help="Number of top matches to retrieve")
    parser.add_argument("--shots_num", type=int, default=16, help="Number of top matches to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--recall", type=float, default=0.2, help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="english-french", help="Dataset name")
    parser.add_argument("--test_split", type=float, default=0.3, help="Test split ratio")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--root_data_dir", type=str, default="dataset_files", help="Root directory for dataset")
    parser.add_argument("--save_dir", type=str, default="results/retrieve_aug27", help="Root directory for dataset")
    parser.add_argument("--specific_data", type=str, default=None, help="Root directory for dataset")

    parser.add_argument("--single_layer_mode", action='store_true', required=False)
    parser.add_argument("--generate_str", action='store_true', required=False)
    parser.add_argument("--soft", action='store_true', required=False)
    parser.add_argument("--local", action='store_true', required=False)
    parser.add_argument("--skip", action='store_true', required=False)
    parser.add_argument("--force", action='store_true', required=False)
    parser.add_argument("--group", action='store_true', required=False)
    parser.add_argument("--calculate_nshots_bm25", action='store_false', required=False)
    parser.add_argument("--recollect_state", action='store_true', required=False)
    parser.add_argument("--reeval", action='store_true', required=False)
    parser.add_argument("--reretrieval", action='store_true', required=False)
    parser.add_argument("--debug", action='store_true', required=False)
    parser.add_argument("--use_template", action='store_true', required=False)

    parser.add_argument("--retrieve_method", type=str, default="cosine", help="Retrieval method: cosine, tsne, distance, or cluster")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--weight_ori", type=float, default=0.0)
    parser.add_argument("--weight_fv", type=float, default=1.0)
    parser.add_argument("--template_num", type=int, default=None)
    parser.add_argument("--shots", type=str, default=None)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--n_layers", type=int, default=None)
    args = parser.parse_args()
    seed_everything(seed=args.seed)
    output_file = f"{args.save_dir}/{args.model_name.split('/')[-1]}_{args.dataset_name}_{args.retrieve_method}_{args.shots}shots_{args.weight_ori}ori_{args.weight_fv}fv_{args.recall}recall.json"
    if os.path.exists(output_file) and args.reeval==False:
        print(f"Skip {output_file}")
        assert False
    evaluator = Retrieve_Evaluator(args)
    results = evaluator.evaluate()
    evaluator.save_results(results)