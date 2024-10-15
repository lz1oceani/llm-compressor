from typing import Any, Iterable, List, Tuple, Union

import torch

__all__ = ["get_output_error"]


def get_output_error(
    unquantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
    quantized: List[Tuple[Union[Iterable, torch.Tensor], Any]],
) -> torch.Tensor:
    """
    Calculate mean l1 loss between weight-unquantized outputs and weight-quantized
    outputs

    :param unquantized: unquantized-weight outputs
    :param quantized: quantized-weight outputs
    :return: mean l1 loss between outputs
    """
    unquantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in unquantized
        ],
        start=[],
    )

    quantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in quantized
        ],
        start=[],
    )

    if len(unquantized_outputs) != len(quantized_outputs):
        raise ValueError(
            "Number of samples of weight-unquantized and weight-quantized "
            "outputs differs"
        )
    num_unq = sum([1 for _ in unquantized_outputs if isinstance(_, torch.Tensor)])
    num_q = sum([1 for _ in quantized_outputs if isinstance(_, torch.Tensor)])
    assert num_q == num_unq, (
        f"Number of samples of weight-unquantized and weight-quantized "
        f"outputs differs: {num_unq} vs {num_q}"
    )
    return sum(
        [
            torch.nn.functional.l1_loss(unq, q)
            for unq, q in zip(unquantized_outputs, quantized_outputs) if isinstance(unq, torch.Tensor)
        ]
    ) / num_q
