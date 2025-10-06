from typing import Optional

from frigate.detectors.detector_config import InputTensorEnum


def tensor_transform(input_shape: InputTensorEnum) -> Optional[tuple[int, ...]]:
    """Convert tensor input shape to transpose indices for numpy array transformation."""
    if input_shape == InputTensorEnum.nchw:
        return (0, 3, 1, 2)
    elif input_shape == InputTensorEnum.hwnc:
        return (2, 0, 1, 3)
    elif input_shape == InputTensorEnum.hwcn:
        return (3, 2, 0, 1)
    
    return None