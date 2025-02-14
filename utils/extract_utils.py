from .data_utils import *
import torch
from baukit import TraceDict
from tqdm import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def calculate_sentence_transformer_embedding(text_to_encode):
    num = len(text_to_encode)
    emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = []
    bar = tqdm(range(0, num, 20),desc='calculate embeddings')
    for i in range(0, num, 20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]['input']).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings

def maximize_diversity_selection(embeddings, n_shots):
    selected_indices = []
    first_id = random.choice(range(len(embeddings)))
    selected_indices.append(first_id)
    selected_representations = embeddings[first_id].reshape(1, -1)
    for count in range(n_shots - 1):
        scores = np.sum(cosine_similarity(embeddings, selected_representations), axis=1)
        for i in selected_indices:
            scores[i] = float('inf')
        min_idx = np.argmin(scores)
        selected_representations = torch.cat((selected_representations,
                                                  embeddings[min_idx].reshape(1, -1)), 0)
        selected_indices.append(min_idx.item())
    return selected_indices

def get_mean_layer_activations(dataset, model, model_config, tokenizer, n_icl_examples=10, N_TRIALS=100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None, cot=False, selective_method="random"):
    """
    Computes the average activation for each layer in the model, at the final predictive token
    """
    n_test_examples=1
    activation_storage = torch.zeros(N_TRIALS, model_config["n_layers"], model_config["hidden_dim"])
    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    if selective_method == "diversity":
        train_embedding = calculate_sentence_transformer_embedding(dataset["train"])

    prepend_bos = False if model_config["prepend_bos"] else True
    icl_prompts = []
    for n in tqdm(range(N_TRIALS)):
        word_pairs_test = dataset['valid'][np.random.choice(filter_set, n_test_examples, replace=False)]
        if selective_method == "random":
            word_pairs = dataset["train"][np.random.choice(len(dataset['train']), n_icl_examples, replace=False)]
        else:
            indices = maximize_diversity_selection(train_embedding, n_icl_examples)
            word_pairs = dataset["train"][indices]
        
       
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test,prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators, cot=cot)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos,shuffle_labels=shuffle_labels, cot=cot)
        
        activation_td, icl_prompt = gather_layer_activations(prompt_data=prompt_data, layers=model_config['layer_hook_names'], model=model, tokenizer=tokenizer,model_config=model_config)
        if isinstance(activation_td[model_config["layer_hook_names"][0]].output, tuple):
            stack_initial = torch.vstack([activation_td[layer].output[0].to("cpu") for layer in model_config["layer_hook_names"]])
        else:
            stack_initial = torch.vstack([activation_td[layer].output.to("cpu") for layer in model_config["layer_hook_names"]])
        stack_initial = stack_initial[:, -1, :] # last token
        activation_storage[n] = stack_initial.cpu()
        icl_prompts.append(icl_prompt)

    
    mean_activations = activation_storage.mean(dim=0)
    return mean_activations, icl_prompts, activation_storage


def calculate_natural_text_activations(text, model, tokenizer, model_config):
    """
    calculate the hidden states for natural text
    """
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # Access Activations
    with TraceDict(model, layers=model_config["layer_hook_names"], retain_input=False, retain_output=True) as td:
        model(**inputs) # batch_size, n_tokens, vocab_size

    if isinstance(td[model_config["layer_hook_names"][0]].output, tuple):
        stack_initial = torch.vstack([td[layer].output[0].to("cpu") for layer in model_config["layer_hook_names"]])
    else:
        stack_initial = torch.vstack([td[layer].output.to("cpu") for layer in model_config["layer_hook_names"]])
    activations = stack_initial[:, -1, :] # last token

    return activations.cpu()

# Layer activations
@torch.no_grad()
def gather_layer_activations(prompt_data, layers, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt
    """
    if "instruct" in model_config["name_or_path"]:
        instruct = True
    else:
        instruct = False
    # get sentence and token labels
    query = prompt_data["query_target"]["input"]
    _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config["prepend_bos"], instruct=instruct)
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

    # Access Activations
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:
        model(**inputs) # batch_size, n_tokens, vocab_size
    return td, prompt_string


@torch.no_grad()
def gather_attn_activations(prompt_data, layers, dummy_labels, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt

    Returns:
    td: tracedict with stored activations
    idx_map: map of token indices to respective averaged token indices
    idx_avg: dict containing token indices of multi-token words

    """
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    if "instruct" in model_config["name_or_path"]:
        instruct = True
    else:
        instruct = False
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config['prepend_bos'], instruct=instruct)
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)

    # Access Activations
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        model(**inputs)
    return td, idx_map, idx_avg

def get_mean_head_activations(dataset, model, model_config, tokenizer, n_icl_examples=10, N_TRIALS=100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging
    """
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config["n_heads"], model_config["hidden_dim"] // model_config["n_heads"])
        activations = activations.view(*new_shape)
        # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    

    n_test_examples = 1
    if prefixes is not None and separators is not None:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer, prefixes=prefixes, separators=separators, model_config=model_config)
    else:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, model_config=model_config)
    print(dummy_labels)
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), model_config["hidden_dim"] // model_config["n_heads"])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))
    
    prepend_bos = False if model_config["prepend_bos"] else True

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset["train"]), n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(len(dataset['valid']), n_test_examples, replace=False)]

        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
        
        activation_td, idx_map, idx_avg = gather_attn_activations(prompt_data=prompt_data, layers=model_config["attn_hook_names"], dummy_labels=dummy_labels, model=model, tokenizer=tokenizer, model_config=model_config)
        stack_initial = torch.vstack([split_activations_by_head(activation_td[layer].input.to("cpu"), model_config) for layer in model_config["attn_hook_names"]]).permute(0, 2, 1, 3)
        stack_filtered = stack_initial[:, :, list(idx_map.keys())]
        for (i,j) in idx_avg.values():
            stack_filtered[:,:, idx_map[i]] = stack_initial[:,:,i:j+1].mean(axis=2) # average activations of multi-token words across all its token
        activation_storage[n] = stack_filtered.cpu()
    mean_activations = activation_storage.mean(dim=0)
    return mean_activations
 

def compute_function_vector(mean_activations, indirect_effect, model, model_config, n_top_heads=10, token_class_idx=-1):
    """
    computes funtion vector that communicates the task observed in ICL examples used for downstream intervention

    Parameters:
    token_class_idx: int indicating which token class to use, -1 is default for last token computations
    indirect_effect: tensor of size (N, Layers, Heads, class(optional)) containing the indirect_effect of each head across N trials

    Returns:
    function_vector: vector representing the communication of a particular task
    top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score) 
    """
    model_hidden_dim = model_config["hidden_dim"]
    model_n_heads = model_config['n_heads']
    model_head_dim = model_hidden_dim//model_n_heads
    device = model.device

    li_dims = len(indirect_effect.shape)

    if li_dims == 3 and token_class_idx == -1:
        mean_indirect_effect = indirect_effect.mean(dim=0)
    else:
        assert(li_dims == 4)
        mean_indirect_effect = indirect_effect[:,:,:,token_class_idx].mean(dim=0) # Subset to token class of interest
    
    # Compute Top Influenctual Heads (Layer, Head)
    h_shape = mean_indirect_effect.shape
    topk_vals, topk_inds = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(), 4) for x in topk_vals]))
    top_heads = top_lh[:n_top_heads]

    # Compute Function Vector as sum of influenctiao heads
    function_vector = torch.zeros((1, 1, model_hidden_dim)).to(device)
    T = -1 # Intervention & values taken from last token
    
    for Layer, Head, _ in top_heads:
        out_proj = model.model.layers[Layer].self_attn.o_proj

        x = torch.zeros(model_hidden_dim)
        x[Head*model_head_dim: (Head+1) * model_head_dim] = mean_activations[Layer, Head, T]
        d_out = out_proj(x.reshape(1, 1, model_hidden_dim).to(device).to(model.dtype))

        function_vector += d_out
    function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_hidden_dim)
    return function_vector, top_heads