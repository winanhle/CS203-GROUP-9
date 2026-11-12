import torch

from spaq_iqa.models import CachedFeatureRegressor, CachedGatedRegressor


def test_fully_missing_metadata_falls_back_to_generic_prediction() -> None:
    model = CachedFeatureRegressor(metadata=True)
    model.eval()
    features = torch.randn(3, 2048)
    metadata = torch.zeros(3, 16)
    generic, combined = model(features, metadata)
    torch.testing.assert_close(generic, combined)


def test_metadata_variant_adds_only_sixteen_parameters() -> None:
    image_only = CachedFeatureRegressor(metadata=False)
    metadata = CachedFeatureRegressor(metadata=True)
    image_count = sum(parameter.numel() for parameter in image_only.parameters())
    metadata_count = sum(parameter.numel() for parameter in metadata.parameters())
    assert metadata_count - image_count == 16


def test_zero_initialized_gate_starts_as_identity() -> None:
    model = CachedGatedRegressor()
    model.eval()
    features = torch.randn(3, 2048)
    metadata = torch.randn(3, 16)
    generic, combined = model(features, metadata)
    torch.testing.assert_close(generic, combined)


def test_gated_parameter_difference_is_explicit() -> None:
    image_only = CachedFeatureRegressor(metadata=False)
    gated = CachedGatedRegressor()
    image_count = sum(parameter.numel() for parameter in image_only.parameters())
    gated_count = sum(parameter.numel() for parameter in gated.parameters())
    assert gated_count - image_count == 8704
