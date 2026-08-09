from taylor_swft.evaluation.metrics import distance, METRICS_UNITS
from torch.utils.tensorboard.writer import SummaryWriter
from torch import Tensor


def log_metrics(
    logger: SummaryWriter,
    rir_ref: Tensor,
    rir_test: Tensor,
    sr: int,
    current_step: int,
):
    d = distance(rir_ref, rir_test, sr)
    for metric in d:
        if METRICS_UNITS[metric] != "":
            tag = f"{metric}_{METRICS_UNITS[metric]}"
        else:
            tag = metric
        logger.add_scalar(
            tag=f"METRIC/{tag}",
            scalar_value=d[metric],
            global_step=current_step,
        )
