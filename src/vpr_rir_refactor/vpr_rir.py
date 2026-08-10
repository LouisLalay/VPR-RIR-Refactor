from . import swft_operators
from .evaluate import log_metrics

from datetime import datetime
from pathlib import Path
from torch import Tensor
from torch.nn import Parameter
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch


class VPR_RIR2:
    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        reference_rir: Tensor,
        Lh: int,
        Lg: int,
        sr: int,
        device: torch.device,
        dtype: torch.dtype,
        experiment_dir: str | Path = "runs/tests",
    ) -> None:
        # torch default tensor args
        self.tensor_args = {
            "dtype": dtype,
            "device": device,
        }
        # Desired RIR length
        self.Lh = Lh
        # Desired mode filter length
        self.Lg = Lg
        # Sampling rate
        self.sr = sr

        (
            self.nfft,
            self.taylor_arange,
            self.omega,
        ) = swft_operators.init_taylor_mul(Lh, **self.tensor_args)

        # RIR parameters
        self.mu_epsilon = Parameter(torch.empty(Lh, **self.tensor_args))
        self.r_epsilon = Parameter(torch.empty(Lh, **self.tensor_args))
        self.sigma_epsilon_2 = Parameter(torch.empty(1, **self.tensor_args))

        # SWFT parameters
        self.alpha = Parameter(torch.empty(self.nfft // 2 + 1, **self.tensor_args))
        self.g = Parameter(torch.empty(Lg, **self.tensor_args))

        # Noise parameter
        self.sigma_w_2 = Parameter(torch.empty(1, **self.tensor_args))

        # Known variables
        self.y = reverberant_signal.to(**self.tensor_args)
        self.s = source.to(**self.tensor_args)
        self.rir_ref = reference_rir.to(**self.tensor_args)

        # Init tensorboard
        exp_name = datetime.now().strftime("%Y-%m-%d %Hh%M-%S")
        self.full_exp_path = Path(experiment_dir) / exp_name
        self.logger = SummaryWriter(self.full_exp_path)

        # Init the parameters
        self.init_parameters()

    def __repr__(self) -> str:
        # RIR parameters
        rir_text = f"Parameters of the RIR:\n"
        rir_text += f"mu_epsilon: {self.mu_epsilon.shape}, {self.mu_epsilon}\n"
        rir_text += f"r_epsilon: {self.r_epsilon.shape}, {self.r_epsilon}\n"
        rir_text += (
            f"sigma_epsilon_2: {self.sigma_epsilon_2.shape}, {self.sigma_epsilon_2}\n"
        )

        # SWFT parameters
        swft_text = f"Parameters of the physical model:\n"
        swft_text += f"alpha: {self.alpha.shape}, {self.alpha}\n"
        swft_text += f"g: {self.g.shape}, {self.g}"

        # Noise parameter
        noise_text = f"Noise parameter:\n"
        noise_text += f"sigma_w_2: {self.sigma_w_2.shape}, {self.sigma_w_2}\n"

        return rir_text + "\n" + swft_text + "\n" + noise_text

    def __str__(self) -> str:
        return self.__repr__()

    def init_parameters(self):
        self.mu_epsilon.data = torch.randn_like(self.mu_epsilon)
        self.r_epsilon.data = torch.ones_like(self.r_epsilon)
        self.sigma_epsilon_2.data[0] = 1

        self.alpha.data = swft_operators.rt_profile_to_alpha(
            torch.ones_like(self.alpha)
        )
        self.g.data = torch.zeros_like(self.g)
        self.g.data[0] = 1

        self.sigma_w_2.data[0] = 1

    def log_step(self, current_step: int, log_freq: int):
        if current_step % log_freq:
            return
        with torch.no_grad():
            current_p_fourier = swft_operators.alpha_to_p_fourier(self.alpha, self.sr)
            current_rir_estimate = swft_operators.taylor_mul(
                self.mu_epsilon,
                self.taylor_arange,
                self.omega,
                current_p_fourier,
                self.nfft,
            )
            log_metrics(
                self.logger,
                self.rir_ref,
                current_rir_estimate,
                self.sr,
                current_step,
            )
            swft_operators.log_swft_parameters(
                self.logger,
                self.alpha,
                self.g,
                current_step,
                self.sr,
            )
            swft_operators.log_rirs(
                self.logger,
                self.rir_ref,
                current_rir_estimate,
                self.sr,
                current_step,
            )

    def update_step(self):
        pass

    def fit(self, n_step: int):
        for k in tqdm(range(n_step)):
            self.update_step()
            self.log_step(k, 10)
