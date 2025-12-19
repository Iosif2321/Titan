"""Test configuration for shared asyncio event loop."""

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async DB tests."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()
