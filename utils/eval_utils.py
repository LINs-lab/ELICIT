from tqdm import tqdm
import numpy as np
from .data_utils import *
import re
import string
import torch
from .intervention_utils import *
from .model_utils import *
import json
import nltk
import scipy
import typing
import unicodedata
import datasets
import time

def test_answer(pred_str, ans):
    try:
        first_part = pred_str.split("\nQ:")[0]
    except:
        first_part = pred_str
    numbers = re.findall(r'\d+(?:\.\d+)?(?:/\d+)?(?:,\d{3})*(?:\.\d+)?', first_part)
    if len(numbers) >= 1:
        pred_ans = numbers[-1]
    else:
        return None, 1
    try:
        ans = find_answer(ans)
    except:
        pass
    if ans.strip() == str(pred_ans):
        return pred_ans, 0
    else:
        return pred_ans, 1
        

def get_answer_id(query, answer, tokenizer):
    source = tokenizer(query, truncation=False, padding=False).input_ids
    source_target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    answer_ids = source_target[len(source):]
    return answer_ids

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def parse_generation(output_str, target, metric_fn):
    """
    parse a generated string for the target, and score using the specified metric
    """
    ans_regex = re.compile("([\w. ]+)[\nQ]*")
    parsed_str = ans_regex.findall(output_str)
    if len(parsed_str) > 0:
        parsed_str = parsed_str[0]
        score = metric_max_over_ground_truths(metric_fn, parsed_str, target)
    else:
        score = 0.0

    return parsed_str, score

# evaluate a sentence
def sentence_eval(sentence, target, model, tokenizer, compute_nll=True, generate_str=False, pred_file=None, metric_fn=None, use_vllm=False):
    """
    Evaluate a single sentence completion for a model, comparing to the given target
    """
    # Clean Run, No Intervention
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1
    generation_time = []
    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze())
        nll_targets[:, :-target_len] = -100
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:, original_pred_idx,:]
    elif generate_str:
        MAX_NEW_TOKENS = 256
        # output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        if use_vllm:
            sampling_params = vllm.SamplingParams(
                temperature=0, 
                top_p=1.0,
                max_tokens=MAX_NEW_TOKENS,
                num_beams=1,
                repetition_penalty=1.0, 
            )
            outputs = model.generate(sentence, sampling_params)
            breakpoint()
        else:
            start_time = time.time()
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
            end_time = time.time()
            generation_time.append(end_time-start_time)
            output_str = tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:])
            
       
        parsed_str, score = test_answer(output_str, target)
        # score = 0 if score else 1
        # parsed_str, score = parse_generation(output_str, target, metric_fn)
        if pred_file:
            pred_file.write(f"{output_str.strip()}\n")
            pred_file.write(f"{parsed_str.strip()}\n\n")
    else:
        start_time = time.time()
        clean_output = model(**inputs).logits[:, -1,:]
        end_time = time.time()
        generation_time.append(end_time - start_time)
    if compute_nll:
        return clean_output, clean_nll, original_pred_idx, generation_time
    elif generate_str:
        return score, original_pred_idx, generation_time
    else:
        return clean_output , original_pred_idx, generation_time

def find_answer(s):
    if "The answer is" in s:
        return s.split("The answer is ")[-1].split(".")[0]
    assert('boxed' in s)
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'): 
                stack += 1
                a += c
            elif(c == '}'): 
                stack -= 1
                if(stack == 0): break
                a += c
            else: 
                a += c
    else:
        a = ans.split('$')[0].strip()
    return a

def n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer, compute_ppl=False, generate_str=False, shuffle_labels=False, prefixes=None, separators=None, pred_filepath=None, metric="exact_match_score", dataset_split='test', cot=False, fluency=False, template=None, prepend_space=False,  use_vllm=False):
    """
    Evaluates a model (without any interventions) on provided ICL dataset.

    Parameters:
    compute_ppl: whether to compute perplexity of teacher-forced correct completion for base model & intervened model
    generate_str: whether to generate a string of tokens or predict a single token
    pred_filepath: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)
    """
    prepend_bos =  False if model_config['prepend_bos'] else True
    clean_rank_list = []
    if compute_ppl:
        clean_nll_list = []
    if generate_str:
        score_list = []
    
    if pred_filepath:
        pred_file = open(pred_filepath, "w")
    else:
        pred_file = None

    token_lengths = []
    record_time = []
    for j in tqdm(range(len(dataset[dataset_split])), total=len(dataset[dataset_split])):
        if n_shots == 0:
            word_pairs = {"input":[], "output": []}
        else:
            word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n_shots, replace=False)]
        word_pairs_test = dataset[dataset_split][j]

        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos,  shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators, cot=cot, template=template, prepend_space=prepend_space)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels, cot=cot, template=template,prepend_space=prepend_space)
        # get relevant parts of the prompt
        query, target = prompt_data["query_target"]["input"], prompt_data["query_target"]["label"] if "label"in prompt_data["query_target"].keys() else prompt_data["query_target"]["output"]
        query = query[0] if isinstance(query, list) else query
    
        # if generate_str:
        #     target = [target] if not isinstance(query, list) else target
        # else:
        #     target = target[0] if isinstance(target, list) else target
        sentence = [create_prompt(prompt_data)]
        target_token_id = get_answer_id(sentence[0], target, tokenizer)

        if compute_ppl:
            clean_output, clean_nll, length, generation_time = sentence_eval(sentence, target=[target], model=model, tokenizer=tokenizer, compute_nll=compute_ppl, use_vllm=use_vllm)
            clean_nll_list.append(clean_nll)
        elif generate_str:
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            score, length, generation_time = sentence_eval(sentence, target=target, model=model, tokenizer=tokenizer, compute_nll=False, generate_str=True,pred_file=pred_file, metric_fn=metric_fn)
            score_list.append(score)
        else:
            clean_output, length, generation_time = sentence_eval(sentence, target=[target], model=model,tokenizer=tokenizer, compute_nll=False)
        if not generate_str:
            clean_rank = compute_individual_token_rank(clean_output, target_token_id)
            clean_rank_list.append(clean_rank)
        token_lengths.append(length)

        # calculate fluency
    if fluency:
        fluency =  ce_loss(model_config, model, tokenizer, model.device)
   
    

    if generate_str:
        results = {"score": score_list}
    else:
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)], "clean_rank_list":clean_rank_list}
    if compute_ppl:
        results["clean_ppl"] = np.exp(clean_nll_list).mean()
    if fluency:
        results["fluency"] = fluency
    if pred_filepath:
        pred_file.close()
    # print("Avg Token Length: ", sum(token_lengths)/len(token_lengths))
    results["avg_length"] = sum(token_lengths) / len(token_lengths)
    results["avg_time"] = sum(generation_time) / len(generation_time)
    return results

def n_shot_eval(dataset, fv_vector, edit_layer: int, n_shots: int, model, model_config, tokenizer, shuffle_labels: bool=False, filter_set=None, prefixes=None, separators=None, generate_str=False, pred_filepath=None, metric="f1_score", cot=False, plot=False, weight_fv=1.0, weight_ori=0.0, norm=False, fluency=False, template=None, dataset_split="test", prepend_space=False,  use_vllm=False):
    """
    Evaluate a model and FV intervention on the model using the provided ICL dataset
    """
    clean_rank_list = []
    intervention_rank_list = []
    all_generation_time = []
   
    if generate_str:
        clean_score_list = []
        intervention_score_list = []

    if fluency:
        fluency_list = []

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos =  False if model_config['prepend_bos'] else True

    if filter_set is None:
        filter_set = np.arange(len(dataset[dataset_split]))

    if pred_filepath:
        pred_file = open(pred_filepath, 'w')
        pred_results = []
    else:
        pred_file = None        

    token_lengths = []
    for j in tqdm(range(len(dataset[dataset_split])), total=len(dataset[dataset_split])):
        if j not in filter_set:
            continue
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
        word_pairs_test = dataset[dataset_split][j]
        
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators, cot=cot, template=template, prepend_space=prepend_space)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels, cot=cot, template=template, prepend_space=prepend_space)
    
        # Get relevant parts of the Prompt
        # query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query, target = prompt_data["query_target"]["input"], prompt_data["query_target"]["label"] if "label"in prompt_data["query_target"].keys() else prompt_data["query_target"]["output"]
        query = query[0] if isinstance(query, list) else query

    
        
        sentence = [create_prompt(prompt_data)]
        token_lengths.append(len(tokenizer(sentence).input_ids[0])-1)
        # Figure out token of interest        
        target_token_id = get_answer_id(sentence[0], target, tokenizer)

        if generate_str:
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            clean_output, intervention_output, generation_time = function_vector_intervention(sentence, target=target, edit_layer=edit_layer, function_vector=fv_vector, model=model, model_config=model_config, tokenizer=tokenizer, compute_nll=False, generate_str=generate_str, plot=plot, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm)
            all_generation_time.append(generation_time)
            
            clean_parsed_str, clean_score = test_answer(clean_output, target)
        #    clean_parsed_str, clean_score = parse_generation(clean_output, target, metric_fn)
            # intervention_parsed_str, intervention_score = parse_generation(intervention_output, target, metric_fn)
            intervention_parsed_str, intervention_score = test_answer(intervention_output, target)
            # print("Target: ", target, "\nClean answer: ", clean_parsed_str, clean_score, "\nIntervention Answer: ", intervention_parsed_str, intervention_score)

            clean_score_list.append(clean_score)
            intervention_score_list.append(intervention_score)

            if pred_file:
                item = {
                    "input": sentence,
                    "clean_output": clean_output.strip(),
                    "clean_answer": clean_parsed_str,
                    "intervention_output": intervention_output.strip(),
                    "intervention_answer": intervention_parsed_str,
                    "label": target.strip()
                }
                pred_results.append(item)
        else:
            clean_output, intervention_output, generation_time = function_vector_intervention(sentence, target=[target], edit_layer=edit_layer, function_vector=fv_vector, model=model, model_config=model_config, tokenizer=tokenizer, compute_nll=False, plot=plot, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm)
            all_generation_time.append(generation_time)
            if pred_file:
                item = {
                    "input": sentence,
                    "label": target,
                    "clean_output": tokenizer.decode(torch.argsort(clean_output.squeeze(), descending=True)[0]),
                    "intervention_output": tokenizer.decode(torch.argsort(intervention_output.squeeze(), descending=True)[0])
                }
                pred_results.append(item)


            clean_rank = compute_individual_token_rank(clean_output, target_token_id)
            intervention_rank = compute_individual_token_rank(intervention_output, target_token_id)
            clean_rank_list.append(clean_rank)
            intervention_rank_list.append(intervention_rank)
    if fluency:
        fluency = ce_loss(model_config, model, tokenizer, model.device, edit_layer, function_vector=fv_vector,  weight_fv=weight_fv, weight_ori=weight_ori, norm=norm)

    if pred_file:
        json.dump(pred_results, pred_file, indent=4)
    
    if generate_str:
        results = {"clean_score": clean_score_list,
                   "intervention_score": intervention_score_list} 
    else:      
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)],
                   "clean_rank_list": clean_rank_list,
                   
                   "intervention_topk": [(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1,4)],
                   "intervention_rank_list":intervention_rank_list}
    if fluency:
        results["fluency"] = fluency
    
    if pred_filepath:
        pred_file.close()
    # print("Avg Token Length: ", sum(token_lengths)/len(token_lengths))
    results["avg_length"] = sum(token_lengths) / len(token_lengths)
    results["avg_time"] = sum(all_generation_time)/len(all_generation_time)
    
    return results


def ce_loss(model_config, model, tokenizer, device, edit_layer=None, function_vector=None, weight_fv=None, weight_ori=None, norm=False, num_samples=100):
    dataset = datasets.load_dataset("stas/openwebtext-10k")['train']
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

     # tokenize
    owt = dataset.map(lambda x: {'input_ids': torch.tensor(tokenizer(x['text'], return_tensors='pt')['input_ids'][:,:128])})
    owt.set_format(type='torch', columns=['input_ids'])

    losses = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    if function_vector is not None:
        intervention_fn = add_function_vector(edit_layer, function_vector, device=model.device, idx=0, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm)
    with torch.no_grad(): 
        for i in tqdm(rand_idxs, desc="run_ce_loss"):

            input_ids = owt[i]['input_ids'][:, :128].to(device)
            if function_vector is not None:
                with TraceDict(model,  layers=model_config["layer_hook_names"], edit_output=intervention_fn) as ret:
                    loss = model(input_ids, labels=input_ids).loss
            else:
                loss = model(input_ids, labels=input_ids).loss
            
            losses.append(loss.item())
    if np.isnan(np.mean(losses)):
        breakpoint()
    return np.mean(losses)





def compute_individual_token_rank(prob_list, target_id):
    """
    Individual computation of token ranks across a single distribution
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    return torch.where(torch.argsort(prob_list.squeeze(), descending=True) == target_id)[0].item()

def compute_top_k_accuracy(target_token_ranks, k=10):
    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks)

# Logic from huggingface `evaluate` library
def normalize_answer(s):
    """
    Lowercase text and remove punctuation, articles and extra white space

    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0*num_same / len(prediction_tokens)
    recall = 1.0*num_same / len(ground_truth_tokens)
    f1 = (2*precision*recall) / (precision+recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)  

def first_word_score(prediction, ground_truth):
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) > 0 and len(ground_truth) > 0:
        return prediction[0] == ground_truth[0]
    else:
        return len(prediction) == len(ground_truth)
    
def compute_dataset_baseline(dataset, model, model_config, tokenizer, n_shots=10, seed=42, generate_str=False, metric=None, prefixes=None, separators=None) -> dict:
    """
    Computes the ICL performance of the model on the provided dataset for a varying number of shots.
    Returns:
    results_dict: dictionary containing the ICL performance results as the number of shots in ICL prompts varies.
    """
    results_dict = {}
    for N in range(n_shots+1):
        seed_everything(seed)
        results_dict[N] = n_shot_eval_no_intervention(dataset, n_shots=N, model=model,model_config=model_config, tokenizer=tokenizer,generate_str=generate_str, metric=metric, prefixes=prefixes, separators=separators)
    return results_dict

def make_valid_path_name(path: str):
    file_name , extention = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = file_name + "_("  + str(counter) + ")" +extention
        counter += 1

    return path
