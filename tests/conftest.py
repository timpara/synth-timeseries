"""Shared test fixtures and configuration."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_random():
    """Seed the random number generator for reproducible tests."""
    np.random.seed(42)
    yield
