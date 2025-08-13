This is a lightweight demonstration of running a training sweep.

## Configure hyperparameters
`training_demo/config.py` has a central hyperparameter class `EnvironmentConfig` as well as individual configs for each SAE architecture.

## Precomputing activations
Caching activations on-the-fly significantly increases SAE training time. Precomputing activations saves time, this is especially useful for training a large sweep of hyperparameters.

<details><summary>Details on Activation Storage Structure</summary>

Precomputed activations are stored on S3 with the following structure:

```
bucket/
├── run_name/
│   ├── metadata.json          # Contains shape, dtype, bytes_per_file info
│   ├── statistics.json        # Training statistics
│   ├── cfg.json              # Configuration file
│   └── data_files/           # Serialized activation tensors
```

The activation data is stored as PyTorch tensors using `torch.save()`. Each file contains dictionary with keys `"states"` and `"input_ids"` (for backward compatibility)

The `metadata.json` file contains:
- `shape`: Tensor dimensions 
- `dtype`: Data type (e.g., `torch.float16`)
- `bytes_per_file`: Size of each data file
- `input_ids_shape`: Shape of input token IDs (if applicable)
</details>

## Run training 

`run_training.py`