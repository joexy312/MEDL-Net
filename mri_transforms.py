
"""
Mostly copied from https://github.com/facebookresearch/fastMRI/
"""

import numpy as np
import torch


def fft2_2c(t, dim=(1, 2)):
    t = torch.view_as_complex(t)
    t = torch.fft.fftn(t, dim=dim, norm='ortho')
    t = torch.view_as_real(t)
    return t


def ifft2_2c(t, dim=(1, 2)):
    t = torch.view_as_complex(t)
    t = torch.fft.ifftn(t, dim=dim, norm='ortho')
    t = torch.view_as_real(t)
    return t


def conjugate(data: torch.Tensor) -> torch.Tensor:

    data = data.clone()
    data[..., 1] = data[..., 1] * -1.0
    return data


def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:

    complex_index = -1

    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

    multiplication = torch.cat(
        [
            real_part.unsqueeze(dim=complex_index),
            imaginary_part.unsqueeze(dim=complex_index),
        ],
        dim=complex_index,
    )

    return multiplication


def modulus(data: torch.Tensor) -> torch.Tensor:
    """Compute modulus of complex input data. Assumes there is a complex axis (of dimension 2) in the data.

    Parameters
    ----------
    data: torch.Tensor

    Returns
    -------
    output_data: torch.Tensor
        Modulus of data.
    """

    # assert_complex(data, complex_last=False)
    complex_axis = -1 if data.size(-1) == 2 else 1

    return (data**2).sum(complex_axis).sqrt()  # noqa


def reduce_operator(
    coil_data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Parameters
    ----------
    coil_data: torch.Tensor
        Zero-filled reconstructions from coils. Should be a complex tensor (with complex dim of size 2).
    sensitivity_map: torch.Tensor
        Coil sensitivity maps. Should be complex tensor (with complex dim of size 2).
    dim: int
        Coil dimension. Default: 0.

    Returns
    -------
    torch.Tensor:
        Combined individual coil images.
    """

    return complex_multiplication(conjugate(sensitivity_map), coil_data).sum(dim)


def expand_operator(
    data: torch.Tensor,
    sensitivity_map: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Returns
    -------
    torch.Tensor:
        Zero-filled reconstructions from each coil.
    """

    return complex_multiplication(sensitivity_map, data.unsqueeze(dim))


def safe_divide(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    """Divide input_tensor and other_tensor safely, set the output to zero where the divisor b is zero.

    Parameters
    ----------
    input_tensor: torch.Tensor
    other_tensor: torch.Tensor

    Returns
    -------
    torch.Tensor: the division.
    """

    data = torch.where(
        other_tensor == 0,
        torch.tensor([0.0], dtype=input_tensor.dtype).to(input_tensor.device),
        input_tensor / other_tensor,
    )
    return data


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)
