"""Shared pytest configuration for tests/."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: test is slower than unit tests (does I/O or benchmarks)",
    )
