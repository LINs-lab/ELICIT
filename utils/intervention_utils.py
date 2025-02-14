from baukit import TraceDict, get_module
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
def get_module(model, name):
    for n,m in model.named_modules:
        if n == name:
            return m
    raise LookupError(name)


# dataset="sentiment"
def plot_tensor_distributions(tensor1, tensor2, name1="Tensor 1", name2="Tensor 2", bins=100, save_name=None):
    """
    Plot the distributions of two [1, 4096] tensors side by side.
    
    :param tensor1: First tensor to plot
    :param tensor2: Second tensor to plot
    :param name1: Name of the first tensor (for legend)
    :param name2: Name of the second tensor (for legend)
    :param bins: Number of bins for the histogram
    """
    # Convert tensors to numpy arrays
    array1 = tensor1.cpu().numpy().flatten()
    array2 = tensor2.cpu().numpy().flatten()
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Distribution Comparison of Two Tensors')
    
    # Plot histograms
    ax1.hist(array1, bins=bins, alpha=0.7, label=name1)
    ax1.hist(array2, bins=bins, alpha=0.7, label=name2)
    ax1.set_title('Overlapping Histogram')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Plot kernel density estimation
    from scipy.stats import gaussian_kde
    kde1 = gaussian_kde(array1)
    kde2 = gaussian_kde(array2)
    x_range = np.linspace(min(array1.min(), array2.min()), max(array1.max(), array2.max()), 1000)
    ax2.plot(x_range, kde1(x_range), label=name1)
    ax2.plot(x_range, kde2(x_range), label=name2)
    ax2.set_title('Kernel Density Estimation')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # Add some statistics
    stats_text = f'{name1} - Mean: {array1.mean():.4f}, Std: {array1.std():.4f}\n'
    stats_text += f'{name2} - Mean: {array2.mean():.4f}, Std: {array2.std():.4f}'
    fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"results/new_vs_original/{save_name}.png")
    plt.close()

def add_function_vector(edit_layer, fv_vector, device, idx=-1, plot=False, weight_fv=1.0, weight_ori=0, norm=False):
    """
    Adds a vector to the output of a specified layer in the model

    Returns:
    add_act: a function specifying how to add a function vector to a layer's output hidden states
    """
    def add_act(output, layer_name):
        nonlocal edit_layer, fv_vector
        current_layer = int(layer_name.split(".")[2])
        if isinstance(edit_layer, int):
            edit_layer = [edit_layer]
       
        if len(edit_layer) > 1:
            intert_fv = fv_vector[current_layer].unsqueeze(0).clone()
        else:
            intert_fv = fv_vector.clone()
        # fv all layers [32, 4096]
        # one layer [1, 4096]
        if current_layer in edit_layer:
            if isinstance(output, tuple):
                if plot:
                    plot_tensor_distributions(output[0][:, idx], intert_fv.to(output[0].device), "Original", "Function Vector", save_name=f"{edit_layer}")
                if norm:
                    original_vector = output[0][:, idx].clone()
                    original_norm = torch.norm(original_vector, p=2, dim=-1)
                    updated_vector = weight_ori * original_vector + weight_fv * intert_fv.to(output[0].device)
                    updated_norm = torch.norm(updated_vector, p=2, dim=-1)
                    normalized_vector = updated_vector * (original_norm / updated_norm)
                    output[0][:, idx] = normalized_vector
                else:
                    output[0][:, idx] = weight_ori * output[0][:, idx] + weight_fv * intert_fv .to(output[0].device) #
                
                return output
            else: # MAMBA
                if norm:
                    original_vector = output[:, idx].clone()
                    original_norm = torch.norm(original_vector, p=2, dim=-1)
                    updated_vector = weight_ori * original_vector + weight_fv * intert_fv.to(output.device)
                    updated_norm = torch.norm(updated_vector, p=2, dim=-1)
                    normalized_vector = updated_vector * (original_norm / updated_norm)
                    output[:, idx] = normalized_vector
                else:
                    output[:, idx] =  weight_ori * output[:, idx] + weight_fv * intert_fv .to(output.device) #intert_fv.to(output.device)
    
                return output
        else:
            return output
    return add_act

@torch.no_grad()
def function_vector_intervention(sentence, target, edit_layer, function_vector, model, model_config, tokenizer, compute_nll=False, generate_str=False, plot=False, weight_fv=1.0, weight_ori=0, norm=False):
    """
    Run the model on the sentence and adds the function vector to the output of edit_layer as a model intervention, predicting a single token.
    Return the output of the model with and without intervention.

    Returns: a tuple containing output results of a clean run and intervened run of the model
    """

    # Clean Run, No Intervention
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1
    
    if compute_nll:
        target_completion = "".join(sentence+target)
        nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze())
        nll_targets[:, :-target_len] = -100
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:,original_pred_idx,:]
        intervention_idx = -1-target_len
    elif generate_str:
        # MAX_NEW_TOKENS = 100
        # output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS)
        # clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        intervention_idx = -1
        MAX_NEW_TOKENS = 256
        # output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
        # output = model.generate(inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS)
        output = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
        clean_output = tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:])
    else:
        clean_output = model(**inputs).logits[:,-1,:]
        intervention_idx = -1
    
    # perform Intervention
    intervention_fn = add_function_vector(edit_layer, function_vector, device=model.device, idx=intervention_idx, plot=plot, weight_fv=weight_fv, weight_ori=weight_ori, norm=norm) #function_vector.reshape(1, model_config['hidden_dim'])
    
    
    if compute_nll:
        with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:, original_pred_idx,:]
    elif generate_str:
            # output = model.generate(inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS)
            # start_time = time.time()
            # output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
            # end_time = time.time()
            # generation_time = end_time-start_time
            # # output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=MAX_NEW_TOKENS)
            # intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        start_time = time.time()
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
            intervention_output = model.generate(**inputs, max_new_tokens = 1,  do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
        intervention_output = model.generate(intervention_output, max_new_tokens=MAX_NEW_TOKENS-1, do_sample=False, top_p=1.0, num_beams=1, temperature=1.0, repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id)
        end_time = time.time()
        generation_time = end_time-start_time
        intervention_output = tokenizer.decode(intervention_output.squeeze()[inputs.input_ids.shape[1]:])

    else:
        with TraceDict(model, layers=model_config["layer_hook_names"], edit_output=intervention_fn):
            start_time = time.time()
            intervention_output = model(**inputs).logits[:,-1,:]
            end_time = time.time()
            generation_time = end_time-start_time

    fvi_output = (clean_output, intervention_output, generation_time)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    return fvi_output

def replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, idx_map, batched_input=False, last_token_only=False):
    """
    An intervention function for replacing activations witha computed average value.

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    idx_map: dict mapping prompt label indices to ground truth label indices
    batched_input: whether or not to batch the intervention across all heads

    returns:
    rep_act: a function that specifies how to replace activations with an average when given a hooked pytorch module

    """
    edit_layer = [x[0] for x in layer_head_token_pairs]
    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layer:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            # shapes for intervention
            original_shape = input.shape
            new_shape = inputs.size()[:-1] + (model_config["n_heads"], model_config["hidden_dim"] // model_config["n_heads"])
            inputs = inputs.view(*new_shape)

            # Perform Intervention
            if batched_input:
                for i in range(model_config['n_heads']):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            elif last_token_only:
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, -1, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            else:
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] = avg_activations[layer, head_n, idx_map[token_n]]
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight
            new_output = torch.matmul(inputs, out_proj.T)
            return new_output
        else:
            return output
    return rep_act