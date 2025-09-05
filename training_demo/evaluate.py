from typing import Any

from training_demo.config import BaseConfig
import dictionary_learning.utils as utils
import json
import os
from dictionary_learning.activault_s3_buffer import ActivaultS3ActivationBuffer
from training_demo.precompute_activations import LocalCache, tokenized_batch_generator

import torch as t
import torch.nn.functional as F
from collections import defaultdict
from nnsight import LanguageModel


# Enlist all trainers in the trained_sae_save_dir

# for trainer
# load SAE
# evaluate SAE
# save eval results

# plot evaluation


"""
Utilities for evaluating dictionaries on a model and dataset.

Metrics
MSE
Normalized MSE aka. Fraction of variance unexplained
L1
L0
cossim
loss recovered

"""



def compute_l0_norm(f_BLS):
    l0_BL = (f_BLS != 0).float().sum(dim=-1)
    return l0_BL

def compute_l1_norm(f_BLS):
    l1_BL = f_BLS.abs().sum(dim=-1)
    return l1_BL

def compute_mse(x_hat_BLD, x_BLD):
    """Note, that only meaning over batch dimension makes this MSE."""
    squared_error_BLD = (x_hat_BLD - x_BLD) ** 2
    sum_squared_error_BL = squared_error_BLD.sum(dim=-1)
    return sum_squared_error_BL

def compute_normalized_mse(x_hat_BLD, x_BLD):
    target_mean_LD = x_BLD.mean(dim=0) # Keep seperate means for positions L
    target_centered_BLD  = x_BLD - target_mean_LD
    target_variance_BLD  = target_centered_BLD  ** 2
    target_mean_variance_BL = target_variance_BLD.sum(dim=-1)

    sum_squared_error_BL = compute_mse(x_hat_BLD, x_BLD)
    normalized_sum_squared_error_BL = sum_squared_error_BL / target_mean_variance_BL
    return normalized_sum_squared_error_BL

def compute_fraction_variance_explained(x_hat_BLD, x_BLD):
    '''The only difference to normalized mse is that residual variances are centralized at before squaring in fvu.'''
    total_variance_BL = t.var(x_BLD, dim=0, keepdim=True).sum(dim=-1)
    residual_variance_BL = t.var(x_hat_BLD - x_BLD, dim=0, keepdim=True).sum(dim=-1)
    fraction_variance_explained_BL = 1 - residual_variance_BL / total_variance_BL
    return fraction_variance_explained_BL

def compute_cosine_similarity(x_hat_BLD, x_BLD):
    x_hat_normalized_BLD = x_hat_BLD / t.linalg.norm(x_hat_BLD, dim=-1, keepdim=True)
    x_normed_BLD = x_BLD / t.linalg.norm(x_BLD, dim=-1, keepdim=True)
    cosine_similarity_BL = (x_hat_normalized_BLD * x_normed_BLD).sum(dim=-1)
    return cosine_similarity_BL

def compute_relative_reconstruction_bias(x_hat_BLD, x_BLD):
    """Eq 10 from GatedSAE paper https://arxiv.org/pdf/2404.16014"""
    x_hat_norm_squared_BL = t.linalg.norm(x_hat_BLD, dim=-1, ord=2) ** 2
    x_times_x_hat_BL = (x_BLD * x_hat_BLD).sum(dim=-1)
    relative_reconstruction_bias_BL = x_hat_norm_squared_BL / x_times_x_hat_BL
    return relative_reconstruction_bias_BL

def compute_cross_entropy_loss(logits_BLV, tokens_BL):
    loss_fn = t.nn.CrossEntropyLoss()
    loss_BL = loss_fn(input=logits_BLV[:, :-1, :].flatten(0, 1), target=tokens_BL[:, 1:].flatten(0, 1))
    return loss_BL


@t.no_grad()
def compute_loss_recovered_single_batch(
    encoding_BL,  # a batch of tokenized text, no padding tokens expected!
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    # deprecated: normalize_batch=False,  # normalize batch before passing through dictionary
    # deprecated: io="out",  # can be 'in', 'out', or 'in_and_out'
    tracer_args={
        "use_cache": False,
        "output_attentions": False,
        "scan": False,
        "validate": False,
    },  # minimize cache during model trace.
):
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder?
    """

    # Original, unmodified logits
    print(f"trace1")
    with model.trace(encoding_BL):
        logits_original = model.lm_head.output.save()
    logits_original = logits_original.detach().clone()
    
    # intervene with `x_hat`
    print(f"trace2")
    with model.trace(encoding_BL, **tracer_args):
        x = submodule.output[0]  # Output is tuple
        x_hat = dictionary(x)
        submodule.output[0][:] = x_hat
        logits_with_dictionary = model.lm_head.output.save()
    logits_with_dictionary = logits_with_dictionary.detach().clone()

    # logits when replacing component activations with zeros
    print(f"trace3")
    with model.trace(encoding_BL, **tracer_args):
        x = submodule.output[0]
        submodule.output[0][:] = t.zeros_like(x)
        logits_zero = model.lm_head.output.save()
    logits_zero = logits_zero.detach().clone()

    # compute losses
    input_ids = encoding_BL.input_ids.to(logits_original.device)
    loss_original_BL = compute_cross_entropy_loss(logits_original, encoding_BL.input_ids)
    loss_with_dictionary_BL = compute_cross_entropy_loss(logits_with_dictionary, encoding_BL.input_ids)
    loss_zero_BL = compute_cross_entropy_loss(logits_zero, encoding_BL.input_ids)

    del logits_original, logits_with_dictionary, logits_zero
    t.cuda.empty_cache()

    loss_recovered_BL = (loss_with_dictionary_BL - loss_zero_BL) / (loss_original_BL - loss_zero_BL)

    return loss_recovered_BL.cpu(), loss_original_BL.cpu(), loss_with_dictionary_BL.cpu(), loss_zero_BL.cpu()


def compute_mean_and_error(score_BL, aggregate_across_positions=True):
    B, L = score_BL.shape

    if aggregate_across_positions:
        dims = [0, 1]
        num_samples = B * L
    else:
        dims = [0]
        num_samples = B

    mean = score_BL.mean(dim=dims)
    std = score_BL.std(dim=dims)
    ci = 1.96 * std / (num_samples ** 0.5)

    return mean, ci


@t.no_grad()
def batch_compute_statistics(dictionary, activation_buffer, cfg: BaseConfig):
    mse, nmse, l1, l0, fve, cos_sim, rrb = [], [], [], [], [], [], []
    num_batches = cfg.eval_num_sequences // cfg.eval_batch_size
    active_feature_count_S = t.zeros(dictionary.dict_size, device=cfg.device)

    for batch_idx in range(num_batches):
        x_BLD = next(activation_buffer).to(cfg.device)
        x_hat_BLD, f_BLS = dictionary(x_BLD, output_features=True)

        active_feature_count_S += (f_BLS != 0).float().sum(dim=(0,1))

        l0.append(compute_l0_norm(f_BLS))
        l1.append(compute_l1_norm(f_BLS))
        mse.append(compute_mse(x_hat_BLD, x_BLD))
        nmse.append(compute_normalized_mse(x_hat_BLD, x_BLD))
        fve.append(compute_fraction_variance_explained(x_hat_BLD, x_BLD))
        cos_sim.append(compute_cosine_similarity(x_hat_BLD, x_BLD))
        rrb.append(compute_relative_reconstruction_bias(x_hat_BLD, x_BLD))

    l0 = t.cat(l0, dim=0)
    l1 = t.cat(l1, dim=0)
    mse = t.cat(mse, dim=0)
    nmse = t.cat(nmse, dim=0)
    fve = t.cat(fve, dim=0)
    cos_sim = t.cat(cos_sim, dim=0)
    rrb = t.cat(rrb, dim=0)
    
    per_token_metrics = {
        "l0": l0,
        "l1": l1,
        "mse": mse,
        "normalized_mse": nmse,
        "fraction_variance_explained": fve,
        "cosine_similarity": cos_sim,
        "relative_reconstruction_bias": rrb,
    }

    miscellaneous_metrics = {
        "fraction_alive_features": (active_feature_count_S != 0).float() / dictionary.dict_size
    }

    results = {}
    results["mean"] = {}

    for name, scores_BL in per_token_metrics.items():
        print(name)
        mean, ci = compute_mean_and_error(scores_BL, aggregate_across_positions=True)
        results["mean"].update({
            f"{name}_mean": mean,
            f"{name}_ci": ci
        })

    results["mean_per_position"] = {}

    for name, scores_BL in per_token_metrics.items():
        mean, ci = compute_mean_and_error(scores_BL, aggregate_across_positions=False)
        results["mean_per_position"].update({
            f"{name}_mean_pos": mean,
            f"{name}_ci_pos": ci
        })

    results["miscellaneous"] = miscellaneous_metrics
        
    return results

# Computing downstream CE loss recovered is more complex than all other metrics: requires loading a model and text.
# We therefore treat it as separate functions for easy separability
@t.no_grad()
def batch_compute_loss_recovered(
    tokenized_batch_generator, model, submodule, dictionary, cfg
):
    ce_reco, ce_orig, ce_dict, ce_zero = [], [], [], []
    num_batches = cfg.eval_num_sequences // cfg.eval_batch_size

    for batch_idx in range(num_batches):
        encoding_BL = next(tokenized_batch_generator).to(cfg.device)
        loss_recovered, loss_original, loss_with_dictionary, loss_zero = compute_loss_recovered_single_batch(
            encoding_BL, model, submodule, dictionary
        )
        
        ce_reco.append(loss_recovered)
        ce_orig.append(loss_original)
        ce_dict.append(loss_with_dictionary)
        ce_zero.append(loss_zero)

    ce_reco = t.cat(ce_reco, dim=0)
    ce_orig = t.cat(ce_orig, dim=0)
    ce_dict = t.cat(ce_dict, dim=0)
    ce_zero = t.cat(ce_zero, dim=0)

    per_token_metrics = {
        "downstream_ce_loss_recovered": ce_reco,
        "downstream_ce_loss_original": ce_orig,
        "downstream_ce_loss_with_dict": ce_dict,
        "downstream_ce_loss_zero": ce_zero,
    }

    results = {}
    results["mean"] = {}

    for name, scores_BL in per_token_metrics.items():
        mean, ci = compute_mean_and_error(scores_BL, aggregate_across_positions=True)
        results["mean"].update({
            f"{name}_mean": mean,
            f"{name}_ci": ci
        })

    results["mean_per_position"] = {}

    for name, scores_BL in per_token_metrics.items():
        mean, ci = compute_mean_and_error(scores_BL, aggregate_across_positions=False)
        results["mean_per_position"].update({
            f"{name}_mean_pos": mean,
            f"{name}_ci_pos": ci
        })
        
    return results



def eval_saes(
    cfg: BaseConfig,
    activation_buffer: ActivaultS3ActivationBuffer,
    model: LanguageModel | None = None,
    submodule: Any | None = None,
    token_generator: Any | None = None,
    do_downstream_ce_loss_evaluation: bool = False,
    overwrite_prev_results: bool = False,
) -> dict:

    ae_paths = utils.get_nested_folders(cfg.eval_sae_dir)

    for ae_path in ae_paths:
        # Create results file
        print(f"running {ae_path}")
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        # Load dictionary
        dictionary, config = utils.load_dictionary(ae_path, cfg.device)
        dictionary = dictionary.to(dtype=cfg.dtype)

        # Run evaluations
        eval_results = batch_compute_statistics(
            dictionary,
            activation_buffer,
            cfg
        )

        if do_downstream_ce_loss_evaluation:
            assert model is not None
            assert submodule is not None
            assert token_generator is not None

            eval_results.update(batch_compute_loss_recovered(
                token_generator,
                model,
                submodule,
                dictionary,
                cfg
            ))

        hyperparameters = {
            "n_inputs": cfg.eval_num_sequences,
            "context_length": cfg.ctx_len,
        }
        eval_results["hyperparameters"] = hyperparameters

        # Save results
        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

        del dictionary
        t.cuda.empty_cache()

    # return the final eval_results for testing purposes
    return eval_results

def main():
    env_config = BaseConfig()

    model, tokenizer, submodule = utils.get_model_tokenizer_submodule(
        env_config, do_truncate_model=False, load_with_nnsight=True
    )

    token_generator = tokenized_batch_generator(
        dataset_name=env_config.dataset_name,
        split=env_config.dataset_split,
        tokenizer=tokenizer,
        batch_size=env_config.eval_batch_size,
        ctx_len=env_config.ctx_len,
    )

    local_cache = LocalCache(
        save_dir=env_config.precomputed_act_save_dir,
        seed=env_config.seed,
    )

    activation_buffer = ActivaultS3ActivationBuffer(
        cache=local_cache,
        batch_size=env_config.eval_batch_size,
        device=env_config.device,
        io="out",
        do_flatten_batch_pos=False
    )

    eval_saes(
        cfg=env_config,
        token_generator=token_generator,
        model=model,
        submodule=submodule,
        activation_buffer=activation_buffer,
        do_downstream_ce_loss_evaluation=True,
        overwrite_prev_results=True,
    )

if __name__ == "__main__":
    main()
