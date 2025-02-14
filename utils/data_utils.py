from typing import *
import pandas as pd
from pathlib import Path
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
class ICLDataset:
    def __init__(self, dataset):    
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        cols = ['input', 'output'] + [col for col in self.raw_data.columns if col not in ['input', 'output']]
        self.raw_data = self.raw_data[cols]
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, list) or isinstance(i, np.ndarray):            
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(f"Column '{i}' not in the dataset. Current columns in the dataset: {self.raw_data.columns.to_list()}")
            else:
                return self.raw_data[i].to_list()
        else:
            raise ValueError(f"{i} is not a valid index type. Expected one of: [int, list, np.ndarray, slice, str]")


    def __len__(self):
        return len(self.raw_data)
    
    def __repr__(self):
        s = "ICLDataset" + "({\n\tfeatures: " + f"{self.raw_data.columns.to_list()}\n\tnum_rows: {self.__len__}" + "\n})"

def split_icl_dataset(dataset, train_size=None, test_size=0.3, seed=42):
    """
    Uses scikit-learn's train_test split to create train, valid, test dataset.
    """   
    if train_size is None and test_size is None:
        train_size = 0.7
        test_size = 0.3
    elif train_size is not None and test_size is None:
        test_size = 1- train_size
    elif train_size is None and test_size is not None:
        train_size = 1 - test_size
    elif train_size is not None and test_size is not None:
        assert train_size + test_size == 1
    train, valid = train_test_split(dataset.raw_data, test_size=test_size, train_size=train_size, random_state=seed)
    test, valid = train_test_split(valid, test_size=test_size, random_state=seed)

    train = ICLDataset(train.to_dict(orient='list'))
    valid = ICLDataset(valid.to_dict(orient='list'))
    test = ICLDataset(test.to_dict(orient='list'))

    return {"train": train, "valid": valid, "test": test}

    
    

def load_dataset(
        task_name: str,
        root_data_dir: str='../dataset_files',
        test_split : float= 0.3,
        seed: int=43
) -> Dict[str, ICLDataset]:
    data_folders = ["abstractive", "extractive"]
    path = Path(root_data_dir)

    d_group_map = [(dataset_type, os.path.exists(os.path.join(root_data_dir, dataset_type, task_name+'.json'))) for dataset_type in data_folders]

    d_group = list(filter(lambda x:x[1], d_group_map))

    dataset_folder = d_group[0][0]
    d_path = os.path.join(path, dataset_folder, f'{task_name}.json')
    with open(d_path, "r") as f:
        dataset = json.load(f)
   
    for key in dataset.keys():
        dataset[key] = ICLDataset(pd.DataFrame(dataset[key]).to_dict(orient='list'))
    
    return dataset

def word_pairs_to_prompt_data(
        word_pairs: dict,
        instructions: str = "",
        prefixes: dict = {"input":"Q:", "output":"A:", "instructions":""},
        separators: dict = {"input":"\n" ,"output":"\n\n", "instructions":""},
        query_target_pair: dict = None,
        prepend_bos_token=False,
        shuffle_labels=False,
        prepend_space=True,
        cot=False,
        template=None
) -> dict:
    """
    Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    prepend_space: whether to prepend a space to every input and output token
    word_pairs: dict of the form {'word1': ['a','b',...], 'word2': ['c','d',...]}
    """
    prompt_data = {}
    prompt_data['instructions'] = instructions
    prompt_data['separators'] = separators
    if prepend_bos_token:
        prefixes = {k:(v if k!= 'instructions' else '<|endoftext|>' + v) for (k,v) in prefixes.items()}
    prompt_data['prefixes'] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k: (v[0] if isinstance(v, list) else v)for k,v,in query_target_pair.items()}

    if shuffle_labels:
        randomized_pairs = [np.random.permutation(x).tolist() if i==1 else x for (i, x) in enumerate(list(word_pairs.values()))] #only shuffle labels
        if prepend_space:
            prompt_data['examples'] = [{'input': ' ' + w1, 'output': ' '+str(w2)} for(w1,w2) in list(zip(*randomized_pairs))]
            prompt_data['query_target'] = {k: " " + v for k,v in query_target_pair.items()} if query_target_pair is not None else None
        else:
            prompt_data['examples'] = [{'input': w1, 'output': str(w2)} for(w1,w2) in list(zip(*randomized_pairs))]
            prompt_data['query_target'] = {k: v for k,v in query_target_pair.items()} if query_target_pair is not None else None
    else:
    
        if prepend_space:
            prompt_data['examples'] = [{'input': ' ' + w1, 'output': ' '+str(w2)} for(w1,w2) in list(zip(*word_pairs.values()))]
            if type(query_target_pair["output"]) is int:
                prompt_data['query_target'] = {k: " " + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
                prompt_data['query_target']['output'] = prompt_data['query_target']['output'].strip()
            else:
                prompt_data['query_target'] = {k: " " + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
            
            
        else:
            prompt_data['examples'] = [{'input': w1, 'output': " " +str(w2)} for(w1,w2) in list(zip(*word_pairs.values()))]
            
            prompt_data['query_target'] = {k: str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
            if type(query_target_pair["output"]) is not int:
                prompt_data['query_target']["output"] = " " + prompt_data['query_target']["output"]
        # if cot:
            # breakpoint()
            # prompt_data['query_target'].update({"label": query_target_pair['output'].split("The answer is ")[-1]})
                # prompt_data['prefixes']['output'] = "Let's think step by step. A:" # 82->94?? 77->85
            # prompt_data['prefixes']['output'] = "A: Let's think step by step." # 68 -> / 52 -> 63
    
    if template:
        prompt_data['template'] = template
    

    return prompt_data

# Utils for generating prompt meta labels
def get_prompt_parts_and_labels(prompt_data, query_sentence=None):
    """
    Generate high-level labels for ICL prompts according to its ICL role, such as demonstration, label, separatoes, structural, etc.
    """
    if query_sentence is None and prompt_data["query_target"] is not None:
        query_sentence = prompt_data['query_target']['input']
    if isinstance(query_sentence, list):
        query_sentence = query_sentence[0]
    n_examples = len(prompt_data["examples"])
    assemble_icl_example = lambda example, prompt_data:[prompt_data["prefixes"]["input"], example["input"], prompt_data["separators"]["input"], prompt_data["prefixes"]['output'], example["output"], prompt_data["separators"]["output"]]
    assemble_icl_query = lambda query, prompt_data:[prompt_data["prefixes"]["input"], query, prompt_data["separators"]["input"], prompt_data["prefixes"]['output']]
    prompt_instructions = [prompt_data["prefixes"]["instructions"], prompt_data["instructions"], prompt_data["separators"]["instructions"]]

    prompt_icl_examples = [assemble_icl_example(prompt_data["examples"][i], prompt_data) for i in range(n_examples)]
    prompt_icl_query = [assemble_icl_query(query_sentence, prompt_data)]

    prompt_instructions_labels = ['bos_token', 'instructions_token', 'separator_token']
    prompt_icl_examples_labels = [['structural_token', f'demonstration_{i+1}_token', 'separator_token', 'structural_token', f'demonstration_{i+1}_label_token', 'separator_token'] for i in range(n_examples)]

    prompt_icl_query_labels = [['query_structural_token', 'query_demonstration_token', 'query_separator_token', 'query_structural_token']]

    prompt_parts = prompt_instructions + prompt_icl_examples + prompt_icl_query
    prompt_part_labels = prompt_instructions_labels + prompt_icl_examples_labels + prompt_icl_query_labels

    return prompt_parts, prompt_part_labels

def extend_labels(sentence_parts, text_labels, tokenizer, label_init=[]):
    """
    Extend ICL component labels across words that are tokenized into multiple tokens
    """
    zipped_up = [list(zip(x, y)) if isinstance(x, list) else [(x,y)] for x,y in list(zip(sentence_parts, text_labels))]
    prompt_builder = ''
    final_labels = label_init

    for element in zipped_up:
        for j, (word, label) in enumerate(element):
            if len(word) == 0:
                continue
            pre = len(tokenizer.tokenize(prompt_builder))
            prompt_builder += word
            post = len(tokenizer.tokenize(prompt_builder))

            actual_tokens = post-pre

            final_labels.extend([label * (actual_tokens)])

            if j == 3 or j == 2 and len(element[3]) == 0:
                final_labels[-1] = final_labels[-1].replace('structural', 'predictive').replace('separator', 'predictive')
            if j==5:
                final_labels[-actual_tokens] = final_labels[-actual_tokens].replace('separator', 'end_of_example')
    return final_labels

def tokenize_labels(sentence_parts, text_labels, tokenizer, prepend_bos=False):
    """
    Extend phrase-levels across tokenization for icl prompt
    """
    if prepend_bos:
        labels = extend_labels(sentence_parts, text_labels, tokenizer, label_init=['bos_token'])
    else:
        labels = extend_labels(sentence_parts, text_labels, tokenizer, label_init=[])

    return labels

def create_fewshot_primer(prompt_data):
    prompt = ''
    prompt += prompt_data["prefixes"]["instructions"] + prompt_data["instructions"] + prompt_data["separators"]["instructions"]

    for example in prompt_data["examples"]:
        prompt += prompt_data["prefixes"]["input"] + example["input"] + prompt_data["separators"]["input"]
        prompt += prompt_data["prefixes"]["output"] + example["output"] + prompt_data["separators"]["output"]
    return prompt
        

def create_prompt(prompt_data, sentence=None, tokenizer=None):
    """
    Creates a prompt using the specified stence
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']

    if isinstance(sentence, list):
        sentence = sentence[0]
    # natural prompt
    if "template" in prompt_data.keys():
        prompt_init = prompt_data["template"].format(input=prompt_data["query_target"]["input"].strip())
        if tokenizer is not None:
            messages = [
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": prompt_init},
                    ]
            prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False)
        else:
            prompt = prompt_init
        
        if len(prompt_data['examples']) > 0:
            n_shots = ""
            for d in prompt_data["examples"]:
                n_shots += prompt_data["template"].format(input=d["input"].strip())
                n_shots += d["output"]
                n_shots += "\n\n"
            # n_shots = create_fewshot_primer(prompt_data)
            prompt = n_shots + prompt_init
        return prompt
    
    # n-shot
    prompt_init = create_fewshot_primer(prompt_data)
    if tokenizer is not None:
        messages = [
        {"role": "user", "content": prompt_init},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = prompt_init + prompt_data["prefixes"]["input"] + sentence + prompt_data["separators"]["input"] + prompt_data["prefixes"]["output"]

    return prompt

def get_token_meta_labels(prompt_data, tokenizer, query=None, prepend_bos=False, instruct=False):
    """
    Computes the ICL meta-labels for every token in a prompt.
    """
    if query is None and prompt_data['query_target'] is not None:
        query = prompt_data['query_target']['input']
    if isinstance(query, list):
        query = query[0]
    
    prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)
    token_meta_labels = tokenize_labels(prompt_parts, prompt_part_labels, tokenizer, prepend_bos)

    prompt_string = create_prompt(prompt_data, sentence=query, tokenizer=tokenizer if instruct == True else None)
    tokens = [tokenizer.decode(x) for x in tokenizer(prompt_string).input_ids]
    token_labels = list(zip(np.arange(len(tokens)), tokens, token_meta_labels))
    return token_labels, prompt_string

def get_dummy_token_labels(n_icl_examples, tokenizer, model_config, prefixes = None, separators = None):
    """
    Computes the ground-truth meta labels & indices for an ICL prompt with the specified number of example pairs
    These GT labels assume each word gets a single token
    """

    prepend_bos = False if model_config['prepend_bos'] else True
    if prefixes is not None and separators is not None:
        dummy_prompt_data = word_pairs_to_prompt_data({'input': ['a'] *n_icl_examples, 'output': ['a'] * n_icl_examples}, query_target_pair={'input': ['a'], 'output':['a']}, prepend_bos_token=prepend_bos, prefixes=prefixes,separators=separators)
    else:
        dummy_prompt_data = word_pairs_to_prompt_data({'input': ['a'] *n_icl_examples, 'output': ['a'] * n_icl_examples}, query_target_pair={'input': ['a'], 'output':['a']}, prepend_bos_token=prepend_bos)
        
    final_token_labels, _ = get_token_meta_labels(dummy_prompt_data, tokenizer, prepend_bos=prepend_bos)
    final_token_labels = [ (x[0], x[1]) for x in final_token_labels]

    return final_token_labels

def compute_duplicated_labels(token_labels, gt_labels):
    """
    Compute a  map between duplicated labels and ground truth label positions for localized averaging

    Returns:
    index_map: a dict mapping prompt label indices to ground truth label indices
    dup_label_ranges: indices where labels shouldbe duplicated
    """
    check_inds = list(filter(lambda x: "demon" in x[2], token_labels))
    dup_ranges = pd.DataFrame(check_inds).groupby(2)[0].aggregate(lambda x: (x.min(), x.max()))
    dup_labels = [v for v,x in dup_ranges.items() if (x[1] - x[0]) > 0]
    dup_label_ranges = dup_ranges[dup_labels].to_dict()
    dup_inds = pd.DataFrame(check_inds)[pd.DataFrame(check_inds)[2].duplicated()][0].values

    index_map = {k:v[0] for (k,v) in zip([x[0] for x in token_labels if x[0] not in dup_inds], gt_labels)}

    return index_map, dup_label_ranges

def update_idx_map(idx_map, idx_avg):
    """
    Updates the idx_map to map duplicate tokens to its gt token position
    """
    update_map = {}
    for (i,j) in idx_avg.values():
        for k in range(i, j+1):
            if k not in idx_map.keys():
                update_map[k] = idx_map[i]
    update_map = {**idx_map, **update_map}
    return update_map
    

def get_natural_text(dataset_name):
    if dataset_name == "english-french":
        prefixes = {"input": '', "output":'',"instructions":''}
        separators = {"input": ' in French is', "output": '\n', 'instructions':''}
    elif dataset_name == "antonym":
        prefixes = {"input": 'The antonym of', "output": 'is',"instructions":''}
        separators = {"input": ' ', "output": '\n', 'instructions':''}
    elif dataset_name == "sentiment":
        prefixes = {"input": "The sentiment of '", "output": "' is","instructions":''}
        separators = {"input": '', "output": '\n', 'instructions':''}
    elif dataset_name == 'product-company':
        prefixes = {"input": 'The company of', "output": 'is',"instructions":''}
        separators = {"input": ' ', "output": '\n', 'instructions':''}
    return prefixes, separators

   

