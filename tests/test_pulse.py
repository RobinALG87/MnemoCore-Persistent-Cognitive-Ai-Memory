import pytest
import asyncio
from mnemocore.core.pulse import PulseLoop
from mnemocore.core.config import PulseConfig
from mnemocore.core.container import Container
from mnemocore.core.engine import HAIMEngine

@pytest.fixture
def mock_container():
    c = Container(config=None) # type: ignore
    return c

@pytest.fixture
def mock_engine():
    return HAIMEngine()

def test_pulse_initialization(mock_container):
    config = PulseConfig(enabled=True, interval_seconds=1)
    pulse = PulseLoop(config=config, container=mock_container)
    
    assert getattr(pulse.config, "enabled", False) is True
    assert pulse._running is False

@pytest.mark.asyncio
async def test_pulse_start_stop(mock_container):
    config = PulseConfig(enabled=True, interval_seconds=1)
    pulse = PulseLoop(config=config, container=mock_container)
    
    task = asyncio.create_task(pulse.start())
    await asyncio.sleep(0.1)
    
    assert pulse._running is True
    
    pulse.stop()
    await task
    assert pulse._running is False
