from . import constants
from . import swft_operators
from .evaluate import log_metrics

from datetime import datetime
from pathlib import Path
from torch import Tensor
from torch.nn import Parameter
from torch.utils.tensorboard.writer import SummaryWriter
from torchaudio.functional import fftconvolve
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
        Lalpha: int,
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
        # Desired RT profile length
        self.Lp = Lalpha
        # Source signal length
        self.Ls = source.shape[0]
        # Full signal length
        self.Ly = self.Ls + self.Lh - 1
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
        self.alpha = Parameter(torch.empty(self.Lp, **self.tensor_args))
        self.alpha_freqs = torch.linspace(0, sr / 2, self.Lp, **self.tensor_args)
        self.g = Parameter(torch.empty(Lg, **self.tensor_args))

        # Noise parameter
        self.sigma_w_2 = Parameter(torch.empty(1, **self.tensor_args))

        # Known variables
        self.y = reverberant_signal[: self.Ly].to(**self.tensor_args)
        self.s = source.to(**self.tensor_args)
        self.rir_ref = reference_rir.to(dtype=dtype)

        # Init tensorboard
        exp_name = datetime.now().strftime("%Y-%m-%d %Hh%M-%S")
        self.full_exp_path = Path(experiment_dir) / exp_name
        self.logger = SummaryWriter(self.full_exp_path)

        # Init constants for loss computation
        self.init_constants()

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

    def init_constants(self):
        self.y_norm = self.y @ self.y

    def log_step(self, current_step: int, log_freq: int):
        if current_step % log_freq:
            return
        with torch.no_grad():
            current_rir_estimate = self.mu_h().detach().cpu()
            log_metrics(
                self.logger,
                self.rir_ref,
                current_rir_estimate,
                self.sr,
                current_step,
            )
            swft_operators.log_swft_parameters(
                self.logger,
                self.alpha.detach().cpu(),
                self.g.detach().cpu(),
                current_step,
                self.sr,
                tag="Parameters/SWFT_Parameters",
            )
            swft_operators.log_rirs(
                self.logger,
                self.rir_ref,
                current_rir_estimate,
                self.sr,
                current_step,
            )
            swft_operators.log_epsilon(
                self.logger,
                self.mu_epsilon.detach().cpu(),
                self.r_epsilon.detach().cpu(),
                current_step,
                self.sr,
                tag="Parameters/Epsilon",
            )
            self.logger.add_scalar("Loss/loss", self.last_loss, current_step)
            self.logger.add_scalar("Loss/elbo_y", self.last_elbo_y, current_step)
            self.logger.add_scalar(
                "Loss/elbo_epsilon",
                self.last_elbo_epsilon,
                current_step,
            )
            self.logger.add_scalar(
                "Parameters/sigma_w_2",
                self.sigma_w_2.item(),
                current_step,
            )
            self.logger.add_scalar(
                "Parameters/sigma_epsilon_2",
                self.sigma_epsilon_2.item(),
                current_step,
            )

    def mu_h(self) -> Tensor:
        current_rt = swft_operators.alpha_to_rt_profile(self.alpha)
        full_rt = swft_operators.create_linear_RT_profile(
            current_rt,
            self.alpha_freqs,
            self.nfft,
        )
        full_alpha = swft_operators.rt_profile_to_alpha(full_rt)
        mu_h_bar = swft_operators.taylor_mul(
            self.mu_epsilon,
            self.taylor_arange,
            self.omega,
            swft_operators.alpha_to_p_fourier(full_alpha, self.sr),
            self.nfft,
        )
        mu_h = fftconvolve(mu_h_bar, self.g, mode="same")
        return mu_h

    def elbo_y(self) -> Tensor:
        mu_h = self.mu_h()

        # E[ || y - s * h ||^2 ] = ||y||^2 - 2 y^T E[s * h] + E[ || s * h ||^2 ]
        #  E[s * h] = mu_h * s
        expec_first_order = fftconvolve(mu_h, self.s, mode="full")

        # E[ || s * h ||^2 ] = || mu_h * s ||^2 + t2
        # || mu_h * s ||^2
        expec_second_order_t1 = expec_first_order.pow(2).sum()

        # t2 approx || s * g ||^2 <r_epsilon, p_diag^2> under circular convolution assumption
        # || s * g ||^2
        kernel = fftconvolve(self.s, self.g, mode="full").pow(2).sum()
        # p_diag^2
        approximate_P_diag = swft_operators.alpha_to_p_fourier(
            self.alpha.mean() * self.taylor_arange, self.sr
        ).pow(2)
        expec_second_order_t2 = kernel * approximate_P_diag @ self.r_epsilon

        full_expec = (
            self.y_norm
            - 2 * self.y @ expec_first_order
            + expec_second_order_t1
            + expec_second_order_t2
        )

        elbo_y = -0.5 * (
            self.Ly * torch.log(self.sigma_w_2) + full_expec / self.sigma_w_2
        )
        self.last_elbo_y = elbo_y.item()
        return elbo_y

    def elbo_epsilon(self) -> Tensor:
        elbo_epsilon = -0.5 * (
            self.Lh * torch.log(self.sigma_epsilon_2)
            + (self.mu_epsilon @ self.mu_epsilon + self.r_epsilon.sum())
            / self.sigma_epsilon_2
            - self.r_epsilon.log().sum()
        )
        self.last_elbo_epsilon = elbo_epsilon.item()
        return elbo_epsilon

    def loss(self) -> Tensor:
        loss = -(self.elbo_y() + self.elbo_epsilon())
        self.last_loss = loss.item()
        return loss

    def parameters_sanity_check(self):
        with torch.no_grad():
            # sigma_w_2 must be positive
            if self.sigma_w_2 < 0:
                self.sigma_w_2.data[0] = constants.ALMOST_ZERO
            # sigma_epsilon_2 must be positive
            if self.sigma_epsilon_2 < 0:
                self.sigma_epsilon_2.data[0] = constants.ALMOST_ZERO
            # r_epsilon must be positive
            self.r_epsilon.clamp_(min=constants.ALMOST_ZERO)
            # alpha must be positive
            self.alpha.clamp_(min=constants.ALMOST_ZERO)

    def init_optimizer(self, lr: float):
        self.optimizer = torch.optim.Adam(
            [
                self.mu_epsilon,
                self.r_epsilon,
                self.sigma_epsilon_2,
                self.alpha,
                self.g,
                self.sigma_w_2,
            ],
            lr=lr,
        )

    def update_step(self):
        self.optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.optimizer.step()
        self.parameters_sanity_check()

    def fit(self, n_step: int, log_freq: int, lr: float = 1e-3):
        self.init_optimizer(lr=lr)
        with torch.no_grad():
            self.loss()
        for k in tqdm(range(n_step)):
            self.log_step(k, log_freq)
            self.update_step()
        self.log_step(n_step, 1)
