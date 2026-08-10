import torch

from vpr_rir_refactor.swft_operators import (
    taylor_mul,
    create_linear_RT_profile,
    alpha_to_p_fourier,
    rt_profile_to_alpha,
    init_taylor_mul,
)
from taylor_swft import make_demo_room, Reverberator


def test_interpolated_rt():
    """
    Test the package implementation of interpolated_rt against the reference implementation from Taylor-SWFT.
    """
    # Simple demo room with known walls
    swft_room = make_demo_room()
    # Initialize all the utilities
    tft = Reverberator(swft_room)
    # Initialize the default length of the RIR
    tft.compute_late_rir()

    # Retrieve the intermediate vectors from the reference implementation
    p_op_ref = tft.late_rir_operator
    interpolated_rt_ref = p_op_ref.interpolated_rt
    nfft = p_op_ref.nfft

    frequency_bands = torch.from_numpy(swft_room.center_freqs)
    rt60_profile = torch.from_numpy(swft_room.get_rt60_profile())
    interpolated_rt_test = create_linear_RT_profile(
        rt60_profile=rt60_profile,
        frequency_bands=frequency_bands,
        n_fft=nfft,
    )

    # The interpolation method slightly differs, so the check is done with a tolerance
    assert torch.allclose(
        interpolated_rt_ref,
        interpolated_rt_test,
        rtol=1e-2,
        atol=1e-2,
    )


def test_p_fourier():
    """
    Test the package implementation of p_fourier against the reference implementation from Taylor-SWFT.
    """
    # Simple demo room with known walls
    swft_room = make_demo_room()
    # Initialize all the utilities
    tft = Reverberator(swft_room)
    # Initialize the default length of the RIR
    tft.compute_late_rir()

    # Retrieve the intermediate vectors from the reference implementation
    p_op_ref = tft.late_rir_operator
    p_fourier_ref = p_op_ref.P_fourier
    interpolated_rt = p_op_ref.interpolated_rt

    alpha = rt_profile_to_alpha(interpolated_rt)
    p_fourier_test = alpha_to_p_fourier(alpha, sr=swft_room.room.fs)

    assert torch.allclose(p_fourier_ref, p_fourier_test)


def test_init_taylor_mul():
    """
    Test the package implementation of init_taylor_mul against the reference implementation from Taylor-SWFT.
    """
    # Simple demo room with known walls
    swft_room = make_demo_room()
    # Initialize all the utilities
    tft = Reverberator(swft_room)
    # Initialize the default length of the RIR
    tft.compute_late_rir()

    # Retrieve the intermediate vectors from the reference implementation
    p_op_ref = tft.late_rir_operator
    Lh = p_op_ref.Lh
    nfft_ref = p_op_ref.nfft
    omega_ref = p_op_ref.omega
    taylor_arange_ref = p_op_ref.taylor_arange

    nfft_test, taylor_arange_test, omega_test = init_taylor_mul(
        Lh,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert nfft_ref == nfft_test
    assert torch.allclose(omega_ref, omega_test)
    assert torch.allclose(taylor_arange_ref, taylor_arange_test)


def test_taylor_mul():
    """
    Test the package implementation of taylor_mul against the reference implementation from Taylor-SWFT.
    """
    # Simple demo room with known walls
    room = make_demo_room()
    # Initialize all the utilities
    tft = Reverberator(room)
    # Initialize the default length of the RIR
    tft.compute_late_rir()

    # Retrieve the intermediate vectors from the reference implementation
    p_op_ref = tft.late_rir_operator
    Lh = p_op_ref.Lh
    omega = p_op_ref.omega
    taylor_arange = p_op_ref.taylor_arange
    p_fourier = p_op_ref.P_fourier
    nfft = p_op_ref.nfft

    epsilon = torch.randn(Lh)
    result_ref = p_op_ref.taylor_mul(epsilon)
    result_test = taylor_mul(
        epsilon=epsilon,
        taylor_arange=taylor_arange,
        omega=omega,
        p_fourier=p_fourier,
        nfft=nfft,
    )

    assert torch.allclose(result_ref, result_test)
