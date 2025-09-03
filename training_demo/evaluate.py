from training_demo.config import EnvironmentConfig
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



def compute_l0_norm(f_BLD):
    l0_BL = (f_BLD != 0).sum(dim=-1)
    return l0_BL

def compute_mean_L1_norm(f):
    return f.abs().sum(dim=-1).mean(dim=0)

def compute_mse(x_hat, x):
    diff_squared = (x_hat - x) ** 2
    return diff_squared.sum(dim=-1).mean(dim=0)

def compute_normalized_mse(x_hat, x):
    """Identical to fraction of variance explained if mean_dims=[0]"""
    mse = compute_mse(x_hat, x)

    target_mean_LD = x.mean(dim=0) # Keep seperate means for positions L
    target_centered = x - target_mean_LD
    target_variance = target_centered ** 2
    target_mean_variance = target_variance.sum(dim=-1).mean(dim=0)

    return mse / target_mean_variance

def compute_fraction_variance_explained(x_hat, x):
    total_variance = t.var(x_hat, dim=0).sum(dim=-1)
    residual_variance = t.var(x_hat - x, dim=0).sum(dim=-1)
    return 1 - residual_variance / total_variance

def compute_cosine_similarity(x_hat, x):
    f_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
    x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
    return (f_normed * x_normed).sum(dim=-1).mean(dim=0)

def compute_relative_reconstruction_bias(x_hat, x):
    x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2) ** 2
    x_dot_x_hat = (x * x_hat).sum(dim=-1)
    relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()
    return relative_reconstruction_bias

def mean_across_seq_position(list_of_tensors):
    """Compute_metric X functions above already compute mean over batch dim.
    Therefore, dim=0 corresponds to seq pos here."""
    return [a.mean(dim=0) for a in list_of_tensors]
    

@t.no_grad()
def batch_compute_metrics(dictionary, activation_buffer, cfg: EnvironmentConfig):
    mse, nmse, l1, l0, fve, cos_sim, rrb = [], [], [], [], [], [], []
    num_batches = cfg.eval_num_sequences // cfg.eval_batch_size

    for batch_idx in range(num_batches):
        x = next(activation_buffer).to(cfg.device)
        x_hat, f = dictionary(x, output_features=True)

        l0.append(compute_mean_L0_norm(f))
        l1.append(compute_mean_L1_norm(f))
        mse.append(compute_mse(x_hat, x))
        nmse.append(compute_normalized_mse(x_hat, x))
        fve.append(compute_fraction_variance_explained(x_hat, x))
        cos_sim.append(compute_cosine_similarity(x_hat, x))
        rrb.append(compute_relative_reconstruction_bias(x_hat, x))

    if cfg.eval_mean_metric_over_sequence:
        l0 = mean_across_seq_position(l0)
        l1 = mean_across_seq_position(l1)
        mse = mean_across_seq_position(mse)
        nmse = mean_across_seq_position(nmse)
        fve = mean_across_seq_position(fve)
        cos_sim = mean_across_seq_position(cos_sim)
        rrb = mean_across_seq_position(rrb)

    l0 = t.cat(l0, dim=0).mean(dim)

    

    
    



def compute_cross_entropy_loss(logits_BLD, tokens_BL):
    loss_fn = t.nn.CrossEntropyLoss()
    loss = loss_fn(input=logits_BLD[:, :-1, :].flatten(0, 1), target=tokens_BL[:, 1:].flatten(0, 1))
    return loss

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
    with model.trace(encoding_BL):
        logits_original = model.lm_head.output.save()
    logits_original = logits_original.detach().clone()
    
    # intervene with `x_hat`
    with model.trace(encoding_BL, **tracer_args):
        x = submodule.output[0]  # Output is tuple
        x_hat = dictionary(x)
        submodule.output[0][:] = x_hat
        logits_with_dictionary = model.lm_head.output.save()
    logits_with_dictionary = logits_with_dictionary.detach().clone()

    # logits when replacing component activations with zeros
    with model.trace(encoding_BL, **tracer_args):
        x = submodule.output[0]
        submodule.output[0][:] = t.zeros_like(x)
        logits_zero = model.lm_head.output.save()
    logits_zero = logits_zero.detach().clone()

    # compute losses
    loss_original = compute_cross_entropy_loss(logits_original, encoding_BL.input_ids)
    loss_with_dictionary = compute_cross_entropy_loss(logits_with_dictionary, encoding_BL.input_ids)
    loss_zero = compute_cross_entropy_loss(logits_zero, encoding_BL.input_ids)

    loss_recovered = (loss_with_dictionary - loss_zero) / (loss_original - loss_zero)

    del logits_original, logits_with_dictionary, logits_zero
    t.cuda.empty_cache()

    return loss_recovered, loss_original, loss_with_dictionary, loss_zero

@t.no_grad()
def batch_compute_loss_recovered(
    tokenized_batch_generator, model, submodule, dictionary, num_total_sequences, batch_size, cfg
):
    num_batches = num_total_sequences // batch_size
    loss_recovered_B = t.zeros(num_batches)
    loss_original_B = t.zeros(num_batches)
    loss_with_dictionary_B = t.zeros(num_batches)
    loss_zero_B = t.zeros(num_batches)

    print(num_batches)
    for batch_idx in range(num_batches):
        print(f'batch idx {batch_idx}')
        encoding_BL = next(tokenized_batch_generator).to(cfg.device)
        loss_recovered, loss_original, loss_with_dictionary, loss_zero = compute_loss_recovered_single_batch(
            encoding_BL, model, submodule, dictionary
        )
        loss_recovered_B[batch_idx] = loss_recovered.mean(dim=0).cpu()
        loss_original_B[batch_idx] = loss_original.mean(dim=0).cpu()
        loss_with_dictionary_B[batch_idx] = loss_with_dictionary.mean(dim=0).cpu()
        loss_zero_B[batch_idx] = loss_zero.mean(dim=0).cpu()

        t.cuda.empty_cache()

    return {
        "downstream_loss_recovered": loss_recovered_B.mean(dim=0).item(),
        "downstream_loss_original": loss_original_B.mean(dim=0).item(),
        "downstream_loss_with_dictionary": loss_with_dictionary_B.mean(dim=0).item(),
        "downstream_loss_zero": loss_zero_B.mean(dim=0).item(),
    }








def evaluate(
    dictionary,  # a dictionary
    activations,  # a generator of activations; if an ActivationBuffer, also compute loss recovered
    max_len=128,  # max context length for loss recovered
    batch_size=128,  # batch size for loss recovered
):
    
    for _ in range(n_batches):
        try:
            x = next(activations).to(device)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim**0.5)
        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )

        features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(
            dtype=t.float32
        )  # If f is shape (B, L, D), flatten to (B*L, D)
        assert features_BF.shape[-1] == dictionary.dict_size
        assert len(features_BF.shape) == 2

        active_features += features_BF.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        # compute variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2) ** 2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        out["frac_variance_explained"] += frac_variance_explained.item()
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out["relative_reconstruction_bias"] += relative_reconstruction_bias.item()

        if not isinstance(activations, (ActivationBuffer, NNsightActivationBuffer)):
            continue

        # compute loss recovered
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            activations.model,
            activations.submodule,
            dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
            io=io,
            tracer_args=tracer_args,
        )
        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)

        out["loss_original"] += loss_original.item()
        out["loss_reconstructed"] += loss_reconstructed.item()
        out["loss_zero"] += loss_zero.item()
        out["frac_recovered"] += frac_recovered.item()

    out = {key: value / n_batches for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    return out


def eval_saes(
    cfg: EnvironmentConfig,
    token_generator,
    model,
    submodule,
    activation_buffer: ActivaultS3ActivationBuffer,
    overwrite_prev_results: bool = False,
) -> dict:

    ae_paths = utils.get_nested_folders(env_config.trained_sae_save_dir)

    for ae_path in ae_paths:
        print(f"running {ae_path}")
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, cfg.device)
        dictionary = dictionary.to(dtype=cfg.dtype)

        eval_results = compute_loss_recovered(
            token_generator,
            model,
            submodule,
            dictionary,
            num_total_sequences=env_config.eval_num_sequences,
            batch_size=env_config.eval_batch_size,
            cfg=cfg,
        )

        hyperparameters = {
            "n_inputs": cfg.eval_num_sequences,
            "context_length": cfg.ctx_len,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

        del dictionary
        t.cuda.empty_cache()

    # return the final eval_results for testing purposes
    return eval_results


if __name__ == "__main__":
    env_config = EnvironmentConfig()

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
    )

    eval_saes(
        cfg=env_config,
        token_generator=token_generator,
        model=model,
        submodule=submodule,
        activation_buffer=activation_buffer,
        overwrite_prev_results=True,
    )
