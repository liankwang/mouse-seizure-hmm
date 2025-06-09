import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence

Number = Union[int, float]

# Helper function to convert between numpy arrays and tensors
def to_t(array, device=None, dtype=torch.float32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    return torch.tensor(array, device=device, dtype=dtype)

def from_t(tensor):
    return tensor.to("cpu").detach().numpy().astype(np.float64)

def combine(Ta: Number,
            a: Optional[Tuple[torch.Tensor, ...]],
            Tb: Number,
            b: Optional[Tuple[torch.Tensor, ...]]
            ) -> Tuple[Number, Optional[Tuple[torch.Tensor, ...]]]:
    """
    Combines two (optional) tuples of statistics, 'a' and 'b', weighted by Ta and Tb.

    If 'a' is None, the function returns (Tb, b). If 'b' is None, it returns (Ta, a).
    Otherwise, it computes a new tuple where each element is a weighted average
    of the corresponding elements in 'a' and 'b'. The weights Ta and Tb are used
    for elements from 'a' and 'b' respectively, and (Ta + Tb) serves as the
    denominator for the average.

    Args:
        Ta: The numerical weight for the first tuple of statistics 'a'.
        a: An optional tuple of torch.Tensors or similar numerical sequences.
        Tb: The numerical weight for the second tuple of statistics 'b'.
        b: An optional tuple of torch.Tensors or similar numerical sequences.

    Returns:
        The return structure varies:
        - If `a` is None: Returns a 2-tuple `(Tb, b)`, where `Tb` is the
          weight and `b` is the original sequence of statistics.
        - If `b` is None: Returns a 2-tuple `(Ta, a)`, where `Ta` is the
          weight and `a` is the original sequence of statistics.
        - If both `a` and `b` are provided: Returns a single tuple containing
          the element-wise weighted average of statistics from `a` and `b`.
          Note that in this specific case, unlike the others, a combined weight
          is not returned alongside the resulting sequence.

    Note:
        The function asserts that at least one of 'a' or 'b' must be truthy
        (i.e., not None and not empty if it's a sequence that Python evaluates as False when empty).
        It is assumed that if both 'a' and 'b' are provided:
        - They are sequences (e.g., tuples of torch.Tensors) of the same length
          for a meaningful element-wise combination. `zip` will truncate to the
          length of the shorter sequence if lengths differ.
        - The sum of weights (Ta + Tb) is non-zero to avoid division by zero errors.
    """
    assert a or b
    if a is None:
        return Tb, b
    elif b is None:
        return Ta, a
    else:
        return tuple((Ta * ai + Tb * bi) / (Ta + Tb) for ai, bi in zip(a, b))