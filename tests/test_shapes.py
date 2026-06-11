import torch

from mos2_psd_inversion.data import compute_input_stats
from mos2_psd_inversion.model import PSDDefectRegressor
from mos2_psd_inversion.synthetic import generate_synthetic_psd


def test_synthetic_data_shapes() -> None:
    inputs, targets = generate_synthetic_psd(samples=8, seed=1)
    assert inputs.shape == (8, 1, 3000)
    assert targets.shape == (8, 2)
    assert torch.isfinite(inputs).all()
    assert torch.isfinite(targets).all()


def test_model_forward_shape() -> None:
    inputs, _ = generate_synthetic_psd(samples=2, seed=2)
    mean, std = compute_input_stats(inputs)
    model = PSDDefectRegressor(mean=mean, std=std)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (2, 2)
