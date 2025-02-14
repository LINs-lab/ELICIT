import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import os
from utils.data_utils import load_dataset
from utils.model_utils import load_model_and_tokenizer, seed_everything
from utils.eval_utils import n_shot_eval, n_shot_eval_no_intervention
from utils.extract_utils import calculate_natural_text_activations
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import random
import json
import os
import torch


def load_state_soups(result_dir, shots=None, weight_fv=2.0, group=False, recollect=True):
    """Load various state vectors"""
    # ag_news_state = torch.load("results/states/llama2-7b-80k/ag_news_tv_8shots_seed65549.pt")
    # english_french_state = torch.load("results/english-french/english-french_mean_layer_activations_10shot_seed26853.pt")
    # glue_mnli_states = torch.load("results/glue-mnli/glue-mnli_mean_layer_activations_10shot_seed60806.pt")
    # return torch.stack([ag_news_state, english_french_state, glue_mnli_states], dim=0)
    
    if shots == None:
        shots_name = "all_shots"
    else:
        shots_name = f"{shots}shots"
    state_file_name = f"{result_dir}/states/state_soup_{shots_name}{'_group' if group else ''}.pt"
    if os.path.exists(state_file_name) and recollect==False:
        saved_states = torch.load(state_file_name)
    else:
        saved_states = []

        datasets = [
                "ethics_justice", "glue_sst2", "hellaswag", "math_qa", "mmlu_pro_math", 
                "openbookqa", "superglue_rte", "superglue_wic", "glue_mnli", "glue_qnli",
                "arc_challenge", "bbh_boolean_expressions", "bbh_date_understanding",
                "bbh_reasoning_about_colored_objects", "bbh_temporal_sequences", "bbq_age", 
                "boolq", "crows_pairs", "commonsense_qa", "ethics_commonsense",
                # "math_level_1", 
                # "math_level_2", "math_level_3", "math_level_4", "math_level_5", 
                # "gsm8k"
                ]
        # datasets = ["math_qa", "mmlu_pro_math"]
        idx = 0
        for dataset in datasets:
            state_file = f"{result_dir}/states/{dataset}_tv_16shots_seed42_ori1.0_fv{weight_fv}.pt"
            state = torch.load(state_file) #[10, 32, 4096]
            label = dataset+ "_16shots"
            # results_file = f"{result_dir}/{dataset}/mean_all_layers_intervention_zs_16shots_results_sweep_42_ori1.0_fv{weight_fv}.json"
            results_file = f"{result_dir}/{dataset}/mean_layer_intervention_zs_16shots_results_sweep_42_ori1.0_fv{weight_fv}.json"
            with open(results_file, "r")as f:
                result_data = json.load(f)
            if "best_layer" in result_data.keys():
                best_layer = result_data["best_layer"]
            else:
                best_layer = None
            icl_prompt_file = f"{result_dir}/states/{dataset}_tv_16shots_seed42_ori1.0_fv{weight_fv}_icl_prompts.json"
            with open(icl_prompt_file, "r")as f:
                icl_prompts = json.load(f) #10
            
            if group:
                item = {
                    "id": idx,
                    "best_layer": best_layer,
                    "icl_prompt": random.sample(icl_prompts, 1)[0],
                    "state": torch.mean(state, dim=0),
                    "label": label
                    }
                saved_states.append(item)
                idx += 1
            else:
                for i in range(len(icl_prompts)):
                    item = {
                        "id": idx,
                        "best_layer": best_layer,
                        "icl_prompt": icl_prompts[i],
                        "state": state[i],
                        "label": label
                    }
                    saved_states.append(item)
                    idx += 1
     
        torch.save(saved_states, state_file_name)
            
     
    return saved_states


def tsne_retrieval_visualization(save_name, database, query, labels, n_neighbors=1, perplexity=40, n_iter=1000, outlier_threshold=1.5):
    """
    Perform retrieval and visualization using t-SNE with labeled database points
    
    Args:
    database: torch.Tensor, shape [n, 32, 4096]
    query: torch.Tensor, shape [32, 4096]
    labels: list or numpy array, shape [n], labels for database points
    n_neighbors: int, number of nearest neighbors to retrieve
    perplexity: float, perplexity parameter for t-SNE
    n_iter: int, number of iterations for t-SNE
    """
    database_2d = database.view(database.shape[0], -1).cpu().numpy()
    query_2d = query.reshape(1, -1).cpu().numpy()
    combined_data = np.vstack((database_2d, query_2d))
    
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(combined_data)
    
    database_tsne = tsne_results[:-1]
    query_tsne = tsne_results[-1]
    
    distances = np.linalg.norm(database_tsne - query_tsne, axis=1)
    print("All Distances: ", distances)
    nearest_indices = np.argsort(distances)[:n_neighbors]
    
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot for each unique label
    # Process labels
    new_labels = [label.split(" ")[0] for label in labels]
    unique_labels = np.unique(new_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Create a color dictionary
    color_dict = dict(zip(unique_labels, colors))
    
    # Identify outliers
    pairwise_distances = pdist(database_tsne)
    distance_matrix = squareform(pairwise_distances)
    mean_distances = np.mean(distance_matrix, axis=1)
    threshold = np.mean(mean_distances) + outlier_threshold * np.std(mean_distances)
    outliers = mean_distances > threshold
    
    # Create a scatter plot for each unique label
    for label in unique_labels:
        mask = np.array(new_labels) == label
        plt.scatter(database_tsne[mask & ~outliers, 0], database_tsne[mask & ~outliers, 1], 
                    c=[color_dict[label]], alpha=0.5, label=f'Database: {label}', s=150)
    
    # Plot and label outliers with original labels
    for i, (x, y) in enumerate(database_tsne[outliers]):
        plt.scatter(x, y, c=[color_dict[new_labels[i]]], alpha=0.5, s=200, edgecolors='black')
        plt.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.scatter(query_tsne[0], query_tsne[1], c='red', s=400, label='Query')
    plt.scatter(database_tsne[nearest_indices, 0], database_tsne[nearest_indices, 1], 
                c='black', s=400, edgecolors='white', label='Nearest Neighbors')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(save_name)
    plt.tight_layout()
    os.makedirs(f"results/figures/retrieval/", exist_ok=True)
    plt.savefig(f"results/figures/retrieval/{save_name}_tsne_visualization.png")
    plt.close()
    
    return nearest_indices, distances[nearest_indices]

def evaluate_model(dataset, task_vector, model, model_config, tokenizer, template=None, edit_layers=list(range(32)), prefixes=None, separators=None, pred_filepath=None, weight_ori=0.0, weight_fv=1.0, dataset_split="test", disable=False):
    """Evaluate model performance"""

    zs_res = {}
    original_zs_results = n_shot_eval_no_intervention(dataset, 0,  model, model_config, tokenizer, generate_str=False, cot=False, template=template, prefixes=prefixes, separators=separators, dataset_split=dataset_split)
    if dataset_split == "test":
        original_4shots_results = n_shot_eval_no_intervention(dataset, 4,  model, model_config, tokenizer, generate_str=False, cot=False, template=template, prefixes=prefixes, separators=separators, dataset_split=dataset_split)
        original_8shots_results =  n_shot_eval_no_intervention(dataset, 8,  model, model_config, tokenizer, generate_str=False, cot=False, template=template, prefixes=prefixes, separators=separators, dataset_split=dataset_split)

    print("Weight Ori: ", weight_ori)
    print("Weight FV: ", weight_fv)

    if disable == False:
        for i in tqdm(edit_layers):
            zs_res[i] = n_shot_eval(dataset, task_vector[i].unsqueeze(0), i, 0, model, model_config, 
                                    tokenizer, generate_str=False,template=template, prefixes=prefixes, separators=separators, pred_filepath=pred_filepath.replace("generation.json", f"{i}layer_generation.json"), weight_fv=weight_fv, weight_ori=weight_ori, dataset_split=dataset_split)
        if dataset_split == "test":
            return zs_res, original_zs_results, original_4shots_results, original_8shots_results
        else:
            return zs_res, original_zs_results
    else:
        if dataset_split == "test":
            return original_zs_results, original_4shots_results, original_8shots_results
        else:
            return original_zs_results

       

def print_results( original_zs_results, results=None):
    """Print evaluation results"""
    max_acc = 0
    layer_id = 0
    zs_acc =  len(np.where(np.array(original_zs_results["clean_rank_list"]) == 0)[0])/ len(original_zs_results["clean_rank_list"])*100
    if results is not None:
        result_for_each_layer = {}
        print("===================Zero-shots==========================\n", zs_acc)
        for i in results.keys():
            
            clean_acc = len(np.where(np.array(results[i]['clean_rank_list']) == 0)[0]) / len(results[i]['clean_rank_list']) * 100
            intervention_acc = len(np.where(np.array(results[i]['intervention_rank_list'])==0)[0]) / len(results[i]['intervention_rank_list']) * 100
            print(f"Layer {i}: Clean Acc = {round(clean_acc, 2)}%, Intervention Acc = {round(intervention_acc, 2)}%")
            result_for_each_layer[i] = {
                "clean_acc": round(clean_acc,2),
                "intervention_acc": round(intervention_acc, 2),
                "avg_time": results[i]["avg_time"],
                "avg_length": results[i]["avg_length"]
            }
            if intervention_acc > max_acc:
                max_acc = intervention_acc
                layer_id = i
        print("=======================Best Layer & Acc======================")
        print(f"Layer: {layer_id}, ACC: {max_acc}")
        return zs_acc, layer_id, max_acc, result_for_each_layer
    else:
        return zs_acc


def plot_accuracy_vs_layer(results, query, dataset):
    """
    Plot accuracy vs layer number
    
    Args:
    results: dict, containing evaluation results for each layer
    """
    layers = sorted(results.keys())
    clean_acc = []
    intervention_acc = []
    
    for layer in layers:
        clean_acc.append(len(np.where(np.array(results[layer]['clean_rank_list']) == 0)[0]) / len(results[layer]['clean_rank_list']) * 100)
        intervention_acc.append(len(np.where(np.array(results[layer]['intervention_rank_list']) == 0)[0]) / len(results[layer]['intervention_rank_list']) * 100)
    
    plt.figure(figsize=(12, 8))
    plt.plot(layers, clean_acc, marker='o', label='Clean Accuracy')
    plt.plot(layers, intervention_acc, marker='s', label='Intervention Accuracy')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy (%)')
    plt.title(query)
    plt.legend()
    plt.grid(True)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/accuracy_vs_layer_{dataset}.png")
    plt.close()