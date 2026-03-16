import pytest
from smoke_test import make_synthetic_inputs as _make_synthetic_inputs


@pytest.fixture
def make_synthetic_inputs():
    return _make_synthetic_inputs()
