# Base Autoencoder with configurable sparsity
from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops

import t as t
from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder
from collections import namedtuple

from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK

# Apply sparsity is a separate function in splinterp

### Reference implementation

class SplinterpTrainerBase(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update. Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients. This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(
            self,
            steps: int, # total number of steps to train for
            activation_dim: int,
            dict_size: int,
            layer: int,
            lm_name: str,
            dict_class=AutoEncoder,
            lr:float=1e-3,
            l1_penalty:float=1e-1,
            warmup_steps:int=1000, # lr warmup period at start of training
            sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
            decay_start:Optional[int]=None, # decay learning rate after this many steps
            seed:Optional[int]=None,
            device=None,
            wandb_name:Optional[str]='StandardTrainerAprilUpdate',
            submodule_name:Optional[str]=None,

            # Splinterp custom arguments
            mu_enc:float=0.001,
            nu_enc:float=0.001,
            mu_dec:float=0.001,
            nu_dec:float=0.001,
            alpha_w:float=1e-6,
            beta_b:float=1e-6,
            decoder_reg:float=1e-5,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.encoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)


        # splinterp specific
        self.mu_enc = mu_enc
        self.nu_enc = nu_enc
        self.mu_dec = mu_dec
        self.nu_dec = nu_dec
        self.alpha_w = alpha_w
        self.beta_b = beta_b
        self.decoder_reg = decoder_reg

    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = (f * self.ae.decoder.weight.norm(p=2, dim=0)).sum(dim=-1).mean()

        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )

    def update_decoder_analytically(self, x):
        """SINGLE-BATCH decoder update"""
        x = x.to(self.device)
        # Algorithm C.1, Line 1: Compute batchwise average x̄ = (1/N) Σ x^r
        x_bar = x.mean(0)

        # Algorithm C.1, Line 2: for t = 0 to t_max - 1 do (we do this once per epoch)
        # Algorithm C.1, Line 3: SGD Encoder update 
        # (this is done before, see training loop in self.update)

        # Algorithm C.1, Lines 4-9: Optimal Decoder update
        # Algorithm C.1, Line 4: foreach batch B do
        # Only doing once
        
        # Algorithm C.1, Line 5: foreach r ∈ B do
        # Algorithm C.1, Line 6: z^r_{t+1} = ρ(W^{t+1}_{enc} x^r + b^{t+1}_{enc})
        z_sparse = self.ae.encode(x)
        # z_sparse == z^r_{t+1} in algorithm
        
        # Algorithm C.1, Line 7: end (batch processing)
        
        # Algorithm C.1, Line 8: z̄^B_{t+1} = (1/|B|) Σ_{r∈B} z^r_{t+1}
        z_bar_B = z_sparse.mean(dim=0)  # [latent_dim]
        
        # Algorithm C.1, Line 9: end (batch processing)
        
        # Algorithm C.1, Line 10: z̄_{t+1} = (1/N) Σ_B |B| z̄^B_{t+1}
        # Not applicable, since we're working with a single batch

        # Complete the z̄_{t+1} computation ??
        # z_bar_t1 = z_bar_total / total_batch_size Not needed when working with a single batch
        z_bar_t1 = z_bar_B # equivalent to z_bar_total

        A_t = self.decoder_reg * t.eye(self.latent_dim, device=self.device, dtype=self.dtype)
        C_t = t.zeros(self.latent_dim, self.input_dim, device=self.device, dtype=self.dtype)

        # Algorithm C.1, Line 13: foreach batch B do
        # Algorithm C.1, Line 14: foreach r ∈ B do
        # reusing encoding z_sparse from above
        
        # Algorithm C.1, Lines 15-16: Compute ψ^r and φ^r
        # ψ^r = z^r_{t+1} - (N/(N+ν_{dec})) z̄_{t+1}
        batch_size = x.shape[0]  # |B| in the algorithm
        psi_r = z_sparse - (batch_size / (batch_size + self.nu_dec)) * z_bar_t1.unsqueeze(0)

        # φ^r = x^r - (ν_{dec}/(N+ν_{dec})) b^t_{dec} - (N/(N+ν_{dec})) x̄
        nu_dec_term = (self.nu_dec / (batch_size + self.nu_dec)) * (self.prev_decoder_bias if self.prev_decoder_bias is not None else 0)
        phi_r = x - nu_dec_term.unsqueeze(0) - (batch_size / (batch_size + self.nu_dec)) * x_bar.unsqueeze(0)

        # Algorithm C.1, Line 17: end (sample processing)
                
        # Algorithm C.1, Lines 18-19: Concatenate ψ^r and φ^r into matrices
        # Ψ^B = cat(ψ^r) into d × |B| matrix
        # Φ^B = cat(φ^r) into n × |B| matrix
        Psi_B = psi_r.t()  # [latent_dim, batch_size]
        Phi_B = phi_r.t()  # [input_dim, batch_size]
        
        # Algorithm C.1, Line 19: A_t = Ψ_t Ψ_t^T + (α + μ'_{dec}) I_d and C_t = Ψ_t Φ_t^T + μ'_{dec} b^t_{dec}
        # we're just doing one batch, so we could do this much simpler, right?
        A_t += Psi_B @ Psi_B.t()  # [latent_dim, latent_dim]
        C_t += Psi_B @ Phi_B.t()  # [latent_dim, input_dim]
        
        # Algorithm C.1, Line 20: end (batch processing)

        ## NOTE Condition number??
        # Skip condition number calculation for large matrices as it's expensive
        if A_t.shape[0] < 5000:
            print(f"[DEBUG] A_t condition number estimate: {t.linalg.cond(A_t.float()).item():.2e}")
        else:
            print(f"[DEBUG] Matrix too large ({A_t.shape[0]}x{A_t.shape[0]}) - skipping condition number calculation")
        
        # For very large matrices, use a more direct approach
        if A_t.shape[0] > 10000:
            print(f"[DEBUG] Large matrix detected ({A_t.shape[0]}x{A_t.shape[0]}) - using aggressive regularization")
            # Add much stronger regularization for large matrices
            large_matrix_reg = 1e-2
            A_t += large_matrix_reg * t.eye(self.latent_dim, device=self.device, dtype=self.dtype)


        ##         
        try:
            print("[DEBUG] Attempting Cholesky decomposition...")
            # Convert to float32 for numerical stability in Cholesky decomposition
            A_t_float32 = A_t.float()
            C_t_float32 = C_t.float()
            
            # Use Cholesky decomposition for numerical stability
            L = t.linalg.cholesky(A_t_float32)
            print("[DEBUG] Cholesky decomposition successful, solving triangular systems...")
            Y = t.linalg.solve_triangular(L, C_t_float32, upper=False)
            self.decoder.weight = t.linalg.solve_triangular(L.t(), Y, upper=True)
            print("[DEBUG] Cholesky solve completed successfully")
        except t.linalg.LinAlgError as e:
            # Fallback to general solver if Cholesky fails
            print(f"[DEBUG] Cholesky decomposition failed: {e}")
            print("[DEBUG] Attempting general solver...")
            try:
                A_t_float32 = A_t.float()
                C_t_float32 = C_t.float()
                self.decoder.weight = t.linalg.solve(A_t_float32, C_t_float32)
                print("[DEBUG] General solver completed successfully")
            except t.linalg.LinAlgError as e2:
                # Final fallback: use pseudo-inverse for singular matrices
                print(f"[DEBUG] General solver failed: {e2}")
                
                # For very large matrices, use a simpler fallback
                if A_t.shape[0] > 5000:
                    print("[DEBUG] Large matrix - using simple regularized solve instead of pseudo-inverse")
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()
                    
                    # Add very strong regularization and solve
                    strong_reg = 1e-1
                    A_t_float32 += strong_reg * t.eye(self.latent_dim, device=self.device, dtype=t.float32)
                    
                    try:
                        self.decoder.weight = t.linalg.solve(A_t_float32, C_t_float32)
                        print("[DEBUG] Strong regularization solve completed successfully")
                    except:
                        print("[DEBUG] All matrix solvers failed, using identity initialization")
                        self.decoder.weight = t.eye(self.latent_dim, self.input_dim, device=self.device, dtype=t.float32)
                else:
                    print("[DEBUG] Using pseudo-inverse (this may be slow)...")
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()
                    
                    # Add more aggressive regularization for pseudo-inverse
                    reg_strength = 1e-3
                    A_t_float32 += reg_strength * t.eye(self.latent_dim, device=self.device, dtype=t.float32)
                    print(f"[DEBUG] Added regularization {reg_strength} for pseudo-inverse")
                    
                    # Try pseudo-inverse with timeout protection
                    try:
                        A_t_pinv = t.linalg.pinv(A_t_float32, rcond=1e-6)
                        self.decoder.weight = A_t_pinv @ C_t_float32
                        print("[DEBUG] Pseudo-inverse completed successfully")
                    except Exception as e3:
                        print(f"[DEBUG] Pseudo-inverse failed: {e3}")
                        print("[DEBUG] Using identity initialization for decoder weights")
                        self.decoder.weight = t.eye(self.latent_dim, self.input_dim, device=self.device, dtype=t.float32)

        print("[DEBUG] Matrix solve completed, converting back to original dtype...")
        self.decoder.weight = self.decoder.weight.to(self.dtype)
        


        print("[DEBUG] Computing decoder bias...")
        # Algorithm C.1, Line 22: b^{t+1}_{dec} = (ν_{dec}/(N+ν_{dec})) b^t_{dec} + (N/(N+ν_{dec})) (x̄ - z̄_{t+1}^T W^{t+1}_{dec})
        bias_term1 = (self.nu_dec / (batch_size + self.nu_dec)) * (self.prev_decoder_bias if self.prev_decoder_bias is not None else 0)
        bias_term2 = (batch_size / (batch_size + self.nu_dec)) * (x_bar - z_bar_t1 @ self.decoder.weight)
        self.decoder.bias = bias_term1 + bias_term2
        
        print(f"[DEBUG] Decoder update completed successfully!")
        print(f"[DEBUG] Decoder weights norm: {t.norm(self.decoder.weight).item():.6f}")
        print(f"[DEBUG] Decoder bias norm: {t.norm(self.decoder_bias).item():.6f}")
        # Algorithm C.1, Line 23: end

    def update(self, step, activations):

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()

        self.update_decoder_analytically()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainerAprilUpdate',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }











class SparseAutoencoderBase(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        activation_type=ACTIVATION_TYPE,
        k=K,
        do_abs_topk=True,
        l1_lambda=L1_LAMBDA,
        mu_enc=MU_ENC,
        nu_enc=NU_ENC,
        dtype=t.float32,
    ):
        super(SparseAutoencoderBase, self).__init__()
        self.encoder_weights = nn.Parameter(
            t.randn(input_dim, latent_dim, dtype=dtype) / np.sqrt(input_dim)
        )
        self.encoder_bias = nn.Parameter(
            t.zeros(latent_dim, dtype=dtype)
        )  # Set bias to zero after normalization
        self.decoder_bias = nn.Parameter(t.zeros(input_dim, dtype=dtype))

        # Store previous parameters for "stay-close" regularization
        self.prev_encoder_weights = None
        self.prev_encoder_bias = None
        self.prev_decoder_bias = None

        # Configuration parameters
        self.k = k
        self.do_abs_topk = do_abs_topk
        self.l1_lambda = l1_lambda
        self.activation_type = activation_type
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Quadratic update cost parameters
        self.mu_enc = mu_enc
        self.nu_enc = nu_enc

    def store_previous_params(self):
        """Store current parameters for "stay-close" regularization"""
        self.prev_encoder_weights = self.encoder_weights.detach().clone()
        self.prev_encoder_bias = self.encoder_bias.detach().clone()
        self.prev_decoder_bias = self.decoder_bias.detach().clone()

    def encode(self, x):
        # Linear encoder
        out = x @ self.encoder_weights + self.encoder_bias
        # Debug: print pre-activation stats for the first batch in each epoch
        if hasattr(self, "debug_print") and self.debug_print:
            print("[DEBUG] Encoder pre-activation stats:")
            print("  Mean:", out.mean().item())
            print("  Std:", out.std().item())
            print("  Min:", out.min().item())
            print("  Max:", out.max().item())
            print("[DEBUG] Encoder bias stats:")
            print("  Mean:", self.encoder_bias.data.mean().item())
            print("  Min:", self.encoder_bias.data.min().item())
            print("  Max:", self.encoder_bias.data.max().item())
            self.debug_print = False  # Only print once per epoch
        return out

    def apply_sparsity(self, z):
        """Apply sparsity based on activation_type"""
        if self.activation_type == "topk":
            # Apply k-sparsity: keep only the k largest activations per sample
            if self.do_abs_topk:
                z_topk = z.abs()
            else:
                z_topk = z
            values, indices = t.topk(z_topk, k=self.k, dim=-1)
            mask = t.zeros_like(z).scatter_(-1, indices, 1)
            return z * mask
        elif self.activation_type == "relu":
            # Apply ReLU activation for sparsity
            return t.relu(z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def get_l1_loss(self, z_sparse):
        """Calculate L1 sparsity loss for ReLU activation"""
        if self.activation_type == "relu":
            # Use the instance's l1_lambda value which may be different for PAM-SGD vs regular SGD
            # For PAM-SGD with ReLU, we use a reduced L1_LAMBDA_PAM_RELU value to avoid excessive penalties
            return self.l1_lambda * z_sparse.abs().sum(dim=1).mean()
        return 0.0  # No L1 loss for TopK

    def get_stay_close_loss(self):
        """Calculate quadratic 'stay-close' loss for encoder parameters"""
        loss = 0
        if self.prev_encoder_weights is not None and self.mu_enc > 0:
            # μ_enc * ||W_enc - W_enc_prev||²_F
            loss += (
                self.mu_enc
                * ((self.encoder_weights - self.prev_encoder_weights) ** 2).sum()
            )

        if self.prev_encoder_bias is not None and self.nu_enc > 0:
            # ν_enc * ||b_enc - b_enc_prev||²_2
            loss += (
                self.nu_enc * ((self.encoder_bias - self.prev_encoder_bias) ** 2).sum()
            )

        return loss

    def decode(self, z):
        # Tied weights: use transpose of encoder weights
        return z @ self.encoder_weights.t() + self.decoder_bias

    def forward(self, x):
        # Encode (linear)
        z = self.encode(x)

        # Apply sparsity
        z_sparse = self.apply_sparsity(z)

        # Decode and return
        recon = self.decode(z_sparse)

        return recon, z_sparse


# SGD Version (standard autodiff)
class SGDAutoencoder(SparseAutoencoderBase):
    def __init__(
        self,
        input_dim,
        latent_dim,
        activation_type=ACTIVATION_TYPE,
        k=K,
        do_abs_topk=False,
        l1_lambda=L1_LAMBDA,
        mu_enc=MU_ENC,
        nu_enc=NU_ENC,
        dtype=t.float32,
    ):
        super(SGDAutoencoder, self).__init__(
            input_dim,
            latent_dim,
            activation_type,
            k,
            do_abs_topk,
            l1_lambda,
            mu_enc,
            nu_enc,
            dtype,
        )
        # This is the baseline version using standard SGD


# EM Version (PAM-SGD)
class EMAutoencoder(SparseAutoencoderBase):
    def __init__(
        self,
        input_dim, #sae
        latent_dim, #sae
        activation_type=ACTIVATION_TYPE, #relu
        k=K, # relu first
        do_abs_topk=True, # default to true
        l1_lambda=None, # trainer
        mu_enc=MU_ENC,
        nu_enc=NU_ENC,
        mu_dec=MU_DEC,
        nu_dec=NU_DEC,
        alpha_w=ALPHA_W,
        beta_b=BETA_B,
        dtype=t.float32,
        decoder_reg=DECODER_REG,
    ):
        # Use reduced L1 lambda specifically for PAM-SGD with ReLU activation
        if l1_lambda is None:
            if activation_type == "relu":
                l1_lambda = L1_LAMBDA_PAM_RELU  # Use lower lambda for PAM-SGD with ReLU
            else:
                l1_lambda = L1_LAMBDA  # Use default lambda for other cases

        super(EMAutoencoder, self).__init__(
            input_dim,
            latent_dim,
            activation_type,
            k,
            do_abs_topk,
            l1_lambda,
            mu_enc,
            nu_enc,
            dtype,
        )

        # For EM, handle the decoder weights separately (not as Parameter)
        self.decoder_weights = t.randn(
            latent_dim, input_dim, device=self.device, dtype=dtype
        ) / np.sqrt(latent_dim)

        # Store previous decoder parameters for "stay-close" regularization
        self.prev_decoder_weights = None

        # Decoder regularization parameters
        self.mu_dec = mu_dec  # Weight "stay-close" penalty
        self.nu_dec = nu_dec  # Bias "stay-close" penalty
        self.alpha_w = alpha_w  # Weight decay for weights
        self.beta_b = beta_b  # Weight decay for biases
        self.decoder_reg = decoder_reg
        self.dtype = dtype

    def store_previous_params(self):
        """Store current parameters for "stay-close" regularization"""
        super().store_previous_params()
        self.prev_decoder_weights = self.decoder_weights.detach().clone()

    def decode(self, z):
        # explicitly maintained decoder weights instead of tied weights
        return z @ self.decoder_weights + self.decoder_bias

    def apply_sparsity(self, z):
        """Apply sparsity based on activation_type"""
        if self.activation_type == "topk":
            # Apply k-sparsity: keep only the k largest activations per sample
            if self.do_abs_topk:
                z_topk = z.abs()
            else:
                z_topk = z
            values, indices = t.topk(z_topk, k=self.k, dim=-1)
            mask = t.zeros_like(z).scatter_(-1, indices, 1)
            return z * mask
        elif self.activation_type == "relu":
            # Apply ReLU activation for sparsity
            return t.relu(z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def update_decoder_analytically(
        self,
        data_loader,
        num_batches=None,
        use_decoder_stay_close=False,
    ):
        """
        Batchwise PAM-SGD decoder update method (Algorithm C.1).
        
        This implements the algorithm with 
        batch-wise processing and matrix accumulation.

        Args:
            data_loader: DataLoader for the M-step.
            num_batches: Number of batches to use (optional).
            use_decoder_stay_close: If True, apply decoder weight stay-close and weight decay (for ablation only).
        """
        print(f"[DEBUG] Starting decoder update with {len(data_loader)} batches")
        self.eval()  # Set to eval mode 

        # Algorithm C.1, Line 1: Compute batchwise average x̄ = (1/N) Σ x^r
        print(f"[DEBUG] Computing x̄ and z̄_t1 from {len(data_loader)} batches in single pass...")
        total_samples = 0
        x_bar = t.zeros(self.input_dim, device=self.device, dtype=self.dtype)
        
        # Store batch data for second pass to avoid recomputation
        batch_data_store = []
        
        with t.no_grad():
            for batch_idx, data in enumerate(data_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                data = get_batch_data(data).to(device)
                batch_size = data.shape[0]
                
                # Store for second pass
                batch_data_store.append(data)
                
                # Compute x_bar
                x_bar += data.sum(dim=0)
                total_samples += batch_size
        
        x_bar = x_bar / total_samples
        N = total_samples
        print(f"[DEBUG] Computed x̄ from {N} total samples")
        
        # Algorithm C.1, Line 2: for t = 0 to t_max - 1 do (we do this once per epoch)
        # Algorithm C.1, Line 3: SGD Encoder update (this is done in the main training loop)
        
        # Algorithm C.1, Lines 4-9: Optimal Decoder update
        print(f"[DEBUG] Computing z̄_t1 from stored batches...")
        # Initialize accumulation matrices A_t and C_t
        A_t = t.zeros(self.latent_dim, self.latent_dim, device=self.device, dtype=self.dtype)
        C_t = t.zeros(self.latent_dim, self.input_dim, device=self.device, dtype=self.dtype)
        
        with t.no_grad():
            # Algorithm C.1, Line 4: foreach batch B do
            for batch_idx, data in enumerate(batch_data_store):
                batch_size = data.shape[0]  # |B| in the algorithm
                
                # Algorithm C.1, Line 5: foreach r ∈ B do
                # Algorithm C.1, Line 6: z^r_{t+1} = ρ(W^{t+1}_{enc} x^r + b^{t+1}_{enc})
                z = self.encode(data)
                z_sparse = self.apply_sparsity(z)  # z^r_{t+1} in algorithm
                
                # Algorithm C.1, Line 7: end (batch processing)
                
                # Algorithm C.1, Line 8: z̄^B_{t+1} = (1/|B|) Σ_{r∈B} z^r_{t+1}
                z_bar_B = z_sparse.mean(dim=0)  # [latent_dim]
                
                # Algorithm C.1, Line 9: end (batch processing)
                
                # Algorithm C.1, Line 10: z̄_{t+1} = (1/N) Σ_B |B| z̄^B_{t+1}
                # We accumulate this across batches
                if batch_idx == 0: # NOTE Can shouldn't we work with a single batch here?
                    z_bar_total = batch_size * z_bar_B
                    total_batch_size = batch_size
                else:
                    z_bar_total += batch_size * z_bar_B
                    total_batch_size += batch_size
        
        # Complete the z̄_{t+1} computation
        z_bar_t1 = z_bar_total / total_batch_size  # [latent_dim]
        print(f"[DEBUG] Computed z̄_t1 from {total_batch_size} total samples")
        
        # Algorithm C.1, Lines 11-12: Compute A_t and C_t
        # A_t = Σ_{r∈B} μ'_{dec} I_d + N²/(N+ν_{dec})² z̄_{t+1} z̄^T_{t+1}
        # C_t = Σ_{r∈B} μ'_{dec} b^{t}_{dec} + N²/(N+ν_{dec})² z̄_{t+1} (β x̄ - ν_{dec} μ'_{dec})^T
        
        # Start with regularization terms (increase for numerical stability)
        reg_coeff = max(self.decoder_reg, 1e-3)  # Ensure minimum regularization
        A_t = reg_coeff * t.eye(self.latent_dim, device=self.device, dtype=self.dtype)
        C_t = t.zeros(self.latent_dim, self.input_dim, device=self.device, dtype=self.dtype)




        # Now process stored batches again to compute the main A_t and C_t terms
        print(f"[DEBUG] Computing A_t and C_t matrices from stored batches...")
        with t.no_grad():
            # Algorithm C.1, Line 13: foreach batch B do
            for batch_idx, data in enumerate(batch_data_store):
                batch_size = data.shape[0]
                
                # Algorithm C.1, Line 14: foreach r ∈ B do
                z = self.encode(data)
                z_sparse = self.apply_sparsity(z)
                
                # Algorithm C.1, Lines 15-16: Compute ψ^r and φ^r
                # ψ^r = z^r_{t+1} - (N/(N+ν_{dec})) z̄_{t+1}
                psi_r = z_sparse - (N / (N + self.nu_dec)) * z_bar_t1.unsqueeze(0)
                
                # φ^r = x^r - (ν_{dec}/(N+ν_{dec})) b^t_{dec} - (N/(N+ν_{dec})) x̄
                nu_dec_term = (self.nu_dec / (N + self.nu_dec)) * (self.prev_decoder_bias if self.prev_decoder_bias is not None else 0)
                phi_r = data - nu_dec_term.unsqueeze(0) - (N / (N + self.nu_dec)) * x_bar.unsqueeze(0)
                
                # Algorithm C.1, Line 17: end (sample processing)
                
                # Algorithm C.1, Lines 18-19: Concatenate ψ^r and φ^r into matrices
                # Ψ^B = cat(ψ^r) into d × |B| matrix
                # Φ^B = cat(φ^r) into n × |B| matrix
                Psi_B = psi_r.t()  # [latent_dim, batch_size]
                Phi_B = phi_r.t()  # [input_dim, batch_size]
                
                # Algorithm C.1, Line 19: A_t = Ψ_t Ψ_t^T + (α + μ'_{dec}) I_d and C_t = Ψ_t Φ_t^T + μ'_{dec} b^t_{dec}
                A_t += Psi_B @ Psi_B.t()  # [latent_dim, latent_dim]
                C_t += Psi_B @ Phi_B.t()  # [latent_dim, input_dim]
                
                # Algorithm C.1, Line 20: end (batch processing)
        
        print(f"[DEBUG] Completed A_t and C_t computation")
        
        # Add weight decay regularization if enabled
        if use_decoder_stay_close and self.alpha_w > 0:
            A_t += self.alpha_w * t.eye(self.latent_dim, device=self.device, dtype=self.dtype)
        
        print(f"[DEBUG] Starting matrix solve for decoder weights...")
        print(f"[DEBUG] A_t shape: {A_t.shape}, C_t shape: {C_t.shape}")
        
        # Skip condition number calculation for large matrices as it's expensive
        if A_t.shape[0] < 5000:
            print(f"[DEBUG] A_t condition number estimate: {t.linalg.cond(A_t.float()).item():.2e}")
        else:
            print(f"[DEBUG] Matrix too large ({A_t.shape[0]}x{A_t.shape[0]}) - skipping condition number calculation")
        
        # For very large matrices, use a more direct approach
        if A_t.shape[0] > 10000:
            print(f"[DEBUG] Large matrix detected ({A_t.shape[0]}x{A_t.shape[0]}) - using aggressive regularization")
            # Add much stronger regularization for large matrices
            large_matrix_reg = 1e-2
            A_t += large_matrix_reg * t.eye(self.latent_dim, device=self.device, dtype=self.dtype)
        
        # Algorithm C.1, Line 21: W^{t+1}_{dec} = (A_t^T C_t)^T
        # This is equivalent to solving A_t @ W_dec = C_t
        try:
            print("[DEBUG] Attempting Cholesky decomposition...")
            # Convert to float32 for numerical stability in Cholesky decomposition
            A_t_float32 = A_t.float()
            C_t_float32 = C_t.float()
            
            # Use Cholesky decomposition for numerical stability
            L = t.linalg.cholesky(A_t_float32)
            print("[DEBUG] Cholesky decomposition successful, solving triangular systems...")
            Y = t.linalg.solve_triangular(L, C_t_float32, upper=False)
            self.decoder_weights = t.linalg.solve_triangular(L.t(), Y, upper=True)
            print("[DEBUG] Cholesky solve completed successfully")
        except t.linalg.LinAlgError as e:
            # Fallback to general solver if Cholesky fails
            print(f"[DEBUG] Cholesky decomposition failed: {e}")
            print("[DEBUG] Attempting general solver...")
            try:
                A_t_float32 = A_t.float()
                C_t_float32 = C_t.float()
                self.decoder_weights = t.linalg.solve(A_t_float32, C_t_float32)
                print("[DEBUG] General solver completed successfully")
            except t.linalg.LinAlgError as e2:
                # Final fallback: use pseudo-inverse for singular matrices
                print(f"[DEBUG] General solver failed: {e2}")
                
                # For very large matrices, use a simpler fallback
                if A_t.shape[0] > 5000:
                    print("[DEBUG] Large matrix - using simple regularized solve instead of pseudo-inverse")
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()
                    
                    # Add very strong regularization and solve
                    strong_reg = 1e-1
                    A_t_float32 += strong_reg * t.eye(self.latent_dim, device=self.device, dtype=t.float32)
                    
                    try:
                        self.decoder_weights = t.linalg.solve(A_t_float32, C_t_float32)
                        print("[DEBUG] Strong regularization solve completed successfully")
                    except:
                        print("[DEBUG] All matrix solvers failed, using identity initialization")
                        self.decoder_weights = t.eye(self.latent_dim, self.input_dim, device=self.device, dtype=t.float32)
                else:
                    print("[DEBUG] Using pseudo-inverse (this may be slow)...")
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()
                    
                    # Add more aggressive regularization for pseudo-inverse
                    reg_strength = 1e-3
                    A_t_float32 += reg_strength * t.eye(self.latent_dim, device=self.device, dtype=t.float32)
                    print(f"[DEBUG] Added regularization {reg_strength} for pseudo-inverse")
                    
                    # Try pseudo-inverse with timeout protection
                    try:
                        A_t_pinv = t.linalg.pinv(A_t_float32, rcond=1e-6)
                        self.decoder_weights = A_t_pinv @ C_t_float32
                        print("[DEBUG] Pseudo-inverse completed successfully")
                    except Exception as e3:
                        print(f"[DEBUG] Pseudo-inverse failed: {e3}")
                        print("[DEBUG] Using identity initialization for decoder weights")
                        self.decoder_weights = t.eye(self.latent_dim, self.input_dim, device=self.device, dtype=t.float32)
        
        print("[DEBUG] Matrix solve completed, converting back to original dtype...")
        self.decoder_weights = self.decoder_weights.to(self.dtype)
        


        print("[DEBUG] Computing decoder bias...")
        # Algorithm C.1, Line 22: b^{t+1}_{dec} = (ν_{dec}/(N+ν_{dec})) b^t_{dec} + (N/(N+ν_{dec})) (x̄ - z̄_{t+1}^T W^{t+1}_{dec})
        bias_term1 = (self.nu_dec / (N + self.nu_dec)) * (self.prev_decoder_bias if self.prev_decoder_bias is not None else 0)
        bias_term2 = (N / (N + self.nu_dec)) * (x_bar - z_bar_t1 @ self.decoder_weights)
        self.decoder_bias.data = bias_term1 + bias_term2
        
        print(f"[DEBUG] Decoder update completed successfully!")
        print(f"[DEBUG] Processed {N} samples across {batch_idx + 1} batches.")
        print(f"[DEBUG] Decoder weights norm: {t.norm(self.decoder_weights).item():.6f}")
        print(f"[DEBUG] Decoder bias norm: {t.norm(self.decoder_bias).item():.6f}")
        # Algorithm C.1, Line 23: end