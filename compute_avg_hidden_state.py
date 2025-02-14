import os, json
import torch, numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
# Include prompt creation helper functions
from utils.data_utils import *
# from utils.intervention_utils import *
from utils.model_utils import *
# from utils.eval_utils import *
from utils.extract_utils import *
from utils.eval_utils import *
import matplotlib.pyplot as plt

def plot_accuracy_vs_layer(results, clean_key, intervention_key, regular_few_shot_acc, save_name):
    """
    Plot accuracy vs layer number
    
    Args:
    results: dict, containing evaluation results for each layer
    """
    layers = sorted(results.keys())
    clean_acc = []
    intervention_acc = []
    
    for layer in layers:
        clean_acc.append(len(np.where(np.array(results[layer][clean_key]) == 0)[0]) / len(results[layer][clean_key]) * 100)
        intervention_acc.append(len(np.where(np.array(results[layer][intervention_key]) == 0)[0]) / len(results[layer][intervention_key]) * 100)
    
    plt.figure(figsize=(12, 8))
    plt.plot(layers, clean_acc, marker='o', label='Clean Accuracy')
    plt.plot(layers, intervention_acc, marker='s', label='Intervention Accuracy')
    plt.axhline(y=regular_few_shot_acc, color='r', linestyle='--', label='Regular ICL Accuracy')
    plt.xlabel('Layer')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/{save_name}_accuracy_vs_layer.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='meta-llama/Meta-Llama-3-8B')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='results')
    parser.add_argument('--n_seeds', help='Number of seeds', type=int, required=False, default=1)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", required=False, default=10, type=int)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", required=False, default=10)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--test_nums', help="Percentage corresponding to test set split size", required=False, type=int, default=None)
    parser.add_argument('--dataset_split', required=False, type=str, default="test")
    parser.add_argument('--selective_method', required=False, type=str, default="random")

    parser.add_argument('--device', help='Device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--model_dtype', default=16, type=int)
    parser.add_argument('--weight_fv', default=1.0, type=float)
    parser.add_argument('--weight_ori', default=0.0, type=float)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)

    parser.add_argument('--cot', help='Whether to use chain-of-thought', action='store_true', required=False)
    parser.add_argument('--plot', help='Whether to plot the activations', action='store_true', required=False)
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--save_icl', help='Whether to save icl prompts', action='store_true', required=False)
    parser.add_argument('--eval', help='Whether to eval', action='store_true', required=False)   
    parser.add_argument('--norm', help='Whether to norm', action='store_true', required=False)
    parser.add_argument('--recompute', help='Whether to norm', action='store_true', required=False)
    parser.add_argument('--all_layers', help='Whether to norm', action='store_true', required=False)
    parser.add_argument('--fluency', help='Whether to norm', action='store_true', required=False)
    parser.add_argument('--use_vllm', help='Whether to use vllm', action='store_true', required=False)

        
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    n_seeds = args.n_seeds
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    prefixes = args.prefixes
    separators = args.separators
    generate_str = args.generate_str
    cot = args.cot
    test_nums = args.test_nums
    dataset_split = args.dataset_split
    weight_fv = args.weight_fv
    weight_ori = args.weight_ori
    save_icl = args.save_icl
    eval = args.eval
    norm = args.norm
    recompute = args.recompute
    fluency = args.fluency
    all_layers = args.all_layers
    use_vllm = args.use_vllm
    selective_method = args.selective_method
    if args.model_dtype == 16:
        model_dtype = torch.float16 
    elif args.model_dtype == 32:
        model_dtype = torch.float32
    else:
        model_dtype = torch.float64
    plot = args.plot

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_model_and_tokenizer(model_name, model_dtype, use_vllm=use_vllm)

    # seeds = np.random.choice(100000, size=n_seeds)
    for seed in [42]:
        seed_everything(seed)
        # load the dataset
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_split=test_split, seed=seed)
      
        if test_nums is not None:
            if test_nums > len(dataset[dataset_split].raw_data):
                test_nums = len(dataset[dataset_split].raw_data)
            dataset[dataset_split].raw_data = dataset[dataset_split].raw_data[:test_nums]
        print("Computing Mean Activations")
        n_shots = min(n_shots, len(dataset["train"]))
        print("Change number of shots to ", n_shots)

        if not os.path.exists(f"{args.save_path_root}/states/{dataset_name}_tv_{n_shots}shots_seed{seed}_ori{weight_ori}_fv{weight_fv}.pt") or recompute:
            mean_activations, icl_prompts, activation_storage= get_mean_layer_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots, N_TRIALS=n_trials, selective_method=selective_method, cot=cot)
            print("Saving mean layer activations")
            os.makedirs(save_path_root, exist_ok=True)
            os.makedirs(f"{args.save_path_root}/states", exist_ok=True)
            
            if save_icl:
                torch.save(activation_storage, f"{args.save_path_root}/states/{dataset_name}_tv_{n_shots}shots_seed{seed}_ori{weight_ori}_fv{weight_fv}.pt")
                with open(f"{args.save_path_root}/states/{dataset_name}_tv_{n_shots}shots_seed{seed}_ori{weight_ori}_fv{weight_fv}_icl_prompts.json", "w") as f:
                    json.dump(icl_prompts, f, indent=4)
            else:
              
                print("saving only mean activations", activation_storage.shape)
                torch.save(activation_storage, f"{args.save_path_root}/states/{dataset_name}_{n_shots}shots_tv_seed{seed}_ori{weight_ori}_fv{weight_fv}.pt")
            print("States save to ",f"{args.save_path_root}/states/{dataset_name}_{n_shots}shots_tv_seed{seed}_ori{weight_ori}_fv{weight_fv}.pt")
            # write atgs to file
            args.save_path_root = save_path_root

            with open(f"{save_path_root}/mean_layer_activation_args.txt", "w")as f:
                json.dump(args.__dict__, f, indent=2)

        else:
            print("State has been computed!")  

            

        
        if eval:
            print("Evaluating Layer Avgs. Baseline")
            original_zs_results = n_shot_eval_no_intervention(dataset, 0,  model, model_config, tokenizer, generate_str=generate_str, cot=cot, fluency=fluency, dataset_split=dataset_split, prepend_space=True,  use_vllm=use_vllm)
            fs_results = n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer, generate_str=generate_str, cot=cot, fluency=fluency, dataset_split=dataset_split, prepend_space=True, use_vllm=use_vllm)
            
        
            score_key = 'score' if 'score' in fs_results.keys() else 'clean_rank_list'
            regular_few_shot_acc = len(np.where(np.array(fs_results[score_key]) == 0)[0])/ len(fs_results[score_key]) * 100
            print("===================Regular ICL==========================\n", regular_few_shot_acc)
            if fluency:
                print("Fluency: ", fs_results["fluency"])
            
            print("===================Zero-shots==========================\n", len(np.where(np.array(original_zs_results[score_key]) == 0)[0])/ len(fs_results[score_key])*100)
            if fluency:
                print("Fluency: ", original_zs_results["fluency"])
            # filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

            if all_layers:
                edit_layers = list(range(model_config['n_layers']))
                zs_res = n_shot_eval(dataset, mean_activations, edit_layers, 0, model, model_config, tokenizer, generate_str=generate_str,cot=cot, pred_filepath=f"{save_path_root}/{model_name.split('/')[-1]}_intervene_zs_all_layer_ori{weight_ori}_fv{weight_fv}_{n_shots}shots_generation.json", plot=plot, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm, fluency=fluency, dataset_split=dataset_split, prepend_space=True)
                intervention_score_key = "intervention_score" if "intervention_score" in zs_res.keys() else "intervention_rank_list"
                intervention_acc = len(np.where(np.array(zs_res[intervention_score_key])==0)[0]) / len(zs_res[intervention_score_key]) * 100
                print("intervene acc: ", round(intervention_acc,2))
                if fluency:
                    print("Fluency: ", zs_res["fluency"])
                zs_res["intervene_acc"] = intervention_acc
                zs_res["original_zs_fluency"] = original_zs_results["fluency"]
                zs_res["zero-shot_acc"] = len(np.where(np.array(original_zs_results[score_key]) == 0)[0])/ len(fs_results[score_key])*100
                with open(f'{save_path_root}/mean_all_layers_intervention_zs_{n_shots}shots_results_sweep_{seed}_ori{weight_ori}_fv{weight_fv}.json', 'w') as interv_zsres_file:
                    json.dump(zs_res, interv_zsres_file, indent=2)

            else:
                zs_res = {}
                # analysis_results = defaultdict(list)
                edit_layers = range(0, model_config['n_layers'])
                # edit_layers = [15]
                for i in tqdm(edit_layers, desc=""):
                    zs_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 0, model, model_config, tokenizer, generate_str=generate_str,cot=cot, pred_filepath=f"{save_path_root}/{model_name.split('/')[-1]}_intervene_zs_layer{i}_ori{weight_ori}_fv{weight_fv}_{n_shots}shots_generation.json", plot=plot, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm, fluency=fluency, dataset_split=dataset_split, prepend_space=True,  use_vllm=use_vllm)
                    # fss_res[i] = n_shot_eval(dataset, mean_activations[i].unsqueeze(0), i, 10, model, model_config, tokenizer, filter_set=filter_set, shuffle_labels=True)
                
                print("======================zero-shot results===================\n")
                clean_score_key = "clean_score" if "clean_score" in zs_res[0].keys() else 'clean_rank_list'
                intervention_score_key = "intervention_score" if "intervention_score" in zs_res[0].keys() else "intervention_rank_list"
                max_acc = 0
                max_layer_id = 0
                for i in zs_res.keys():
                    acc = len(np.where(np.array(zs_res[i][clean_score_key]) == 0)[0]) / len(zs_res[i][clean_score_key]) * 100
                    intervention_acc = len(np.where(np.array(zs_res[i][intervention_score_key])==0)[0]) / len(zs_res[i][intervention_score_key]) * 100
                    print("Layer", i, "Zero-Shot: ", round(acc, 2), "fv: ", round(intervention_acc,2))
                    if fluency:
                        print(zs_res[i]["fluency"])
                    if intervention_acc > max_acc:
                        max_acc = intervention_acc
                        max_layer_id = i
                    zs_res[i]["acc"] = intervention_acc
                zs_res.update({"best_layer": max_layer_id, "zero-shot acc": acc, "regular_icl_acc": regular_few_shot_acc})
                print(f"Max ACC: {max_acc} at Layer {max_layer_id}")
                # plot_accuracy_vs_layer(zs_res, clean_score_key, intervention_score_key, regular_few_shot_acc, save_name=f"{model_name.split('/')[-1]}_{dataset_name}_{n_shots}shots")
                with open(f'{save_path_root}/mean_layer_intervention_zs_{n_shots}shots_results_sweep_{seed}_ori{weight_ori}_fv{weight_fv}.json', 'w') as interv_zsres_file:
                    json.dump(zs_res, interv_zsres_file, indent=2)

            
