from torch import Tensor


class Baseline1:
    def __init__(
        self,
        source: Tensor,
        reverberant_signal: Tensor,
        reference_rir: Tensor,
        nfft: int,
    ):
        self.reference_rir = reference_rir

    def fit_experiment(self):
        pass

    def get_estimates(self):
        return self.reference_rir, {}


def validate_baseline1_kwargs(kwargs: dict):
    required_keys = {
        "nfft": int,
    }
    validated_kwargs = {}
    for key, expected_type in required_keys.items():
        if key not in kwargs:
            raise ValueError(f"Missing required key '{key}' in kwargs.")
        if expected_type is int:
            if not isinstance(kwargs[key], int):
                raise TypeError(
                    f"Expected '{key}' to be of type {expected_type.__name__}, "
                    f"but got {type(kwargs[key]).__name__}."
                )
            else:
                validated_kwargs[key] = kwargs[key]
    return validated_kwargs
