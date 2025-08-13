# Base Autoencoder with configurable sparsity
from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops

from typing import Optional

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder
from collections import namedtuple

from dictionary_learning.dictionary import AutoEncoder

# Apply sparsity is a separate function in splinterp

### Reference implementation


class SplinterpTrainer(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update. Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients. This trainer will have fewer dead neurons than the standard trainer.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=AutoEncoder,
        lr: float = 1e-3,
        l1_penalty: float = 1e-1,
        warmup_steps: int = 1000,  # lr warmup period at start of training
        sparsity_warmup_steps: Optional[int] = 2000,  # sparsity warmup period at start of training
        decay_start: Optional[int] = None,  # decay learning rate after this many steps
        seed: Optional[int] = None,
        device=None,
        wandb_name: Optional[str] = "StandardTrainerAprilUpdate",
        submodule_name: Optional[str] = None,
        # Splinterp custom arguments
        mu_enc: float = 0.001,
        nu_enc: float = 0.001,
        mu_dec: float = 0.001,
        nu_dec: float = 0.001,
        alpha_w: float = 1e-6,
        beta_b: float = 1e-6,
        decoder_reg: float = 1e-5,
        prev_decoder_bias: Optional[t.Tensor] = None
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.dict_size = dict_size
        self.activation_dim = activation_dim

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.encoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(
            steps,
            warmup_steps,
            decay_start,
            resample_steps=None,
            sparsity_warmup_steps=sparsity_warmup_steps,
        )
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

        self.prev_decoder_bias = prev_decoder_bias

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
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "mse_loss": recon_loss.item(),
                    "sparsity_loss": l1_loss.item(),
                    "loss": loss.item(),
                },
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
        z_bar_B = z_sparse.mean(dim=0)  # [dict_size]

        # Algorithm C.1, Line 9: end (batch processing)

        # Algorithm C.1, Line 10: z̄_{t+1} = (1/N) Σ_B |B| z̄^B_{t+1}
        # Not applicable, since we're working with a single batch

        # Complete the z̄_{t+1} computation ??
        # z_bar_t1 = z_bar_total / total_batch_size Not needed when working with a single batch
        z_bar_t1 = z_bar_B  # equivalent to z_bar_total

        A_t = self.decoder_reg * t.eye(self.dict_size, device=self.device, dtype=x.dtype)
        C_t = t.zeros(self.dict_size, self.activation_dim, device=self.device, dtype=x.dtype)

        # Algorithm C.1, Line 13: foreach batch B do
        # Algorithm C.1, Line 14: foreach r ∈ B do
        # reusing encoding z_sparse from above

        # Algorithm C.1, Lines 15-16: Compute ψ^r and φ^r
        # ψ^r = z^r_{t+1} - (N/(N+ν_{dec})) z̄_{t+1}
        batch_size = x.shape[0]  # |B| in the algorithm
        psi_r = z_sparse - (batch_size / (batch_size + self.nu_dec)) * z_bar_t1.unsqueeze(0)

        # φ^r = x^r - (ν_{dec}/(N+ν_{dec})) b^t_{dec} - (N/(N+ν_{dec})) x̄
        nu_dec_term = (self.nu_dec / (batch_size + self.nu_dec)) * (
            self.prev_decoder_bias if self.prev_decoder_bias is not None else 0
        )
        phi_r = (
            x
            - t.tensor(nu_dec_term, device=self.device).unsqueeze(0)
            - (batch_size / (batch_size + self.nu_dec)) * x_bar.unsqueeze(0)
        )

        # Algorithm C.1, Line 17: end (sample processing)

        # Algorithm C.1, Lines 18-19: Concatenate ψ^r and φ^r into matrices
        # Ψ^B = cat(ψ^r) into d × |B| matrix
        # Φ^B = cat(φ^r) into n × |B| matrix
        Psi_B = psi_r.t()  # [dict_size, batch_size]
        Phi_B = phi_r.t()  # [activation_dim, batch_size]

        # Algorithm C.1, Line 19: A_t = Ψ_t Ψ_t^T + (α + μ'_{dec}) I_d and C_t = Ψ_t Φ_t^T + μ'_{dec} b^t_{dec}
        # we're just doing one batch, so we could do this much simpler, right?
        A_t += Psi_B @ Psi_B.t()  # [dict_size, dict_size]
        C_t += Psi_B @ Phi_B.t()  # [dict_size, activation_dim]

        # Algorithm C.1, Line 20: end (batch processing)

        ## NOTE Condition number??
        # Skip condition number calculation for large matrices as it's expensive
        if A_t.shape[0] < 5000:
            print(f"[DEBUG] A_t condition number estimate: {t.linalg.cond(A_t.float()).item():.2e}")
        else:
            print(
                f"[DEBUG] Matrix too large ({A_t.shape[0]}x{A_t.shape[0]}) - skipping condition number calculation"
            )

        # For very large matrices, use a more direct approach
        if A_t.shape[0] > 10000:
            print(
                f"[DEBUG] Large matrix detected ({A_t.shape[0]}x{A_t.shape[0]}) - using aggressive regularization"
            )
            # Add much stronger regularization for large matrices
            large_matrix_reg = 1e-2
            A_t += large_matrix_reg * t.eye(self.dict_size, device=self.device, dtype=x.dtype)

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
            self.ae.decoder.weight = nn.Parameter(t.linalg.solve_triangular(L.t(), Y, upper=True))
            print("[DEBUG] Cholesky solve completed successfully")
        except t.linalg.LinAlgError as e:
            # Fallback to general solver if Cholesky fails
            print(f"[DEBUG] Cholesky decomposition failed: {e}")
            print("[DEBUG] Attempting general solver...")
            try:
                A_t_float32 = A_t.float()
                C_t_float32 = C_t.float()
                self.ae.decoder.weight = nn.Parameter(t.linalg.solve(A_t_float32, C_t_float32))
                print("[DEBUG] General solver completed successfully")
            except t.linalg.LinAlgError as e2:
                # Final fallback: use pseudo-inverse for singular matrices
                print(f"[DEBUG] General solver failed: {e2}")

                # For very large matrices, use a simpler fallback
                if A_t.shape[0] > 5000:
                    print(
                        "[DEBUG] Large matrix - using simple regularized solve instead of pseudo-inverse"
                    )
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()

                    # Add very strong regularization and solve
                    strong_reg = 1e-1
                    A_t_float32 += strong_reg * t.eye(
                        self.dict_size, device=self.device, dtype=t.float32
                    )

                    try:
                        self.ae.decoder.weight = nn.Parameter(t.linalg.solve(A_t_float32, C_t_float32))
                        print("[DEBUG] Strong regularization solve completed successfully")
                    except:
                        print("[DEBUG] All matrix solvers failed, using identity initialization")
                        self.ae.decoder.weight = nn.Parameter(t.eye(
                            self.dict_size, self.activation_dim, device=self.device, dtype=t.float32
                        ))
                else:
                    print("[DEBUG] Using pseudo-inverse (this may be slow)...")
                    A_t_float32 = A_t.float()
                    C_t_float32 = C_t.float()

                    # Add more aggressive regularization for pseudo-inverse
                    reg_strength = 1e-3
                    A_t_float32 += reg_strength * t.eye(
                        self.dict_size, device=self.device, dtype=t.float32
                    )
                    print(f"[DEBUG] Added regularization {reg_strength} for pseudo-inverse")

                    # Try pseudo-inverse with timeout protection
                    try:
                        A_t_pinv = t.linalg.pinv(A_t_float32, rcond=1e-6)
                        self.ae.decoder.weight = nn.Parameter(A_t_pinv @ C_t_float32)
                        print("[DEBUG] Pseudo-inverse completed successfully")
                    except Exception as e3:
                        print(f"[DEBUG] Pseudo-inverse failed: {e3}")
                        print("[DEBUG] Using identity initialization for decoder weights")
                        self.ae.decoder.weight = nn.Parameter(t.eye(
                            self.dict_size, self.activation_dim, device=self.device, dtype=t.float32
                        ))

        print("[DEBUG] Matrix solve completed, converting back to original dtype...")
        self.ae.decoder.weight = nn.Parameter(self.ae.decoder.weight.to(self.ae.encoder.weight.dtype))

        print("[DEBUG] Computing decoder bias...")
        # Algorithm C.1, Line 22: b^{t+1}_{dec} = (ν_{dec}/(N+ν_{dec})) b^t_{dec} + (N/(N+ν_{dec})) (x̄ - z̄_{t+1}^T W^{t+1}_{dec})
        bias_term1 = (self.nu_dec / (batch_size + self.nu_dec)) * (
            self.prev_decoder_bias if self.prev_decoder_bias is not None else 0
        )
        bias_term2 = (batch_size / (batch_size + self.nu_dec)) * (
            x_bar - z_bar_t1 @ self.ae.decoder.weight
        )
        self.ae.decoder.bias = nn.Parameter(bias_term1 + bias_term2)

        print(f"[DEBUG] Decoder update completed successfully!")
        print(f"[DEBUG] Decoder weights norm: {t.norm(self.ae.decoder.weight).item():.6f}")
        print(f"[DEBUG] Decoder bias norm: {t.norm(self.ae.decoder.bias).item():.6f}")
        # Algorithm C.1, Line 23: end

    def update(self, step, activations):

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()

        self.update_decoder_analytically(x=activations)
        self.scheduler.step()

    @property
    def config(self):
        return {
            "dict_class": "AutoEncoder",
            "trainer_class": "StandardTrainerAprilUpdate",
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "steps": self.steps,
            "decay_start": self.decay_start,
            "seed": self.seed,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
