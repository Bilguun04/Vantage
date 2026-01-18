"""
Testing Guide for mchacks Project

This document explains how to run and write tests for the mchacks application.
"""

# Test Structure

The test suite is organized in the `tests/` directory with the following structure:

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures and configuration
├── test_consumers.py        # Tests for WebSocket consumer
├── test_models.py           # Tests for MongoDB models
└── test_integration.py      # Integration tests
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest
# or
python run_tests.py
```

### Run with Coverage Report

```bash
pytest --cov=assistant --cov=app --cov-report=html --cov-report=term-missing
# or
python run_tests.py coverage
```

### Run Specific Test File

```bash
pytest tests/test_consumers.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_consumers.py::TestGeminiLiveConsumerConnect -v
```

### Run Specific Test

```bash
pytest tests/test_consumers.py::TestGeminiLiveConsumerConnect::test_connect_success -v
```

## Test Categories

### Unit Tests

Tests for individual components in isolation:

```bash
pytest -m unit -v
```

### Integration Tests

Tests for complete workflows and interactions:

```bash
pytest -m integration -v
```

## Test Files Overview

### test_consumers.py

Tests for the `GeminiLiveConsumer` WebSocket consumer:

- **TestGeminiLiveConsumerInit**: Consumer initialization
- **TestGeminiLiveConsumerConnect**: WebSocket connection and Gemini setup
- **TestGeminiLiveConsumerDisconnect**: Cleanup on disconnect
- **TestGeminiLiveConsumerReceiveAudio**: Audio message reception
- **TestGeminiLiveConsumerReceiveText**: Text message reception and saving
- **TestGeminiLiveConsumerReceiveImages**: Screenshot/image reception
- **TestGeminiLiveConsumerAudioHandling**: Sending audio to Gemini
- **TestGeminiLiveConsumerImageHandling**: Sending images to Gemini
- **TestGeminiLiveConsumerTextHandling**: Sending text to Gemini
- **TestGeminiLiveConsumerResponseHandling**: Handling Gemini responses
- **TestGeminiLiveConsumerDatabaseOperations**: MongoDB save operations
- **TestGeminiLiveConsumerErrorHandling**: Error handling and recovery

### test_models.py

Tests for MongoDB models:

- **TestConversationImage**: Image/screenshot documents
- **TestConversationMessage**: Message documents
- **TestGeminiConversation**: Conversation session documents
- **TestUserModel**: User account documents
- **TestSessionModel**: Session documents
- **TestLogModel**: Audit log documents

### test_integration.py

Integration and end-to-end tests:

- **TestIntegrationConversationFlow**: Complete conversation cycles
- **TestIntegrationErrorScenarios**: Error handling in real scenarios
- **TestIntegrationDataPersistence**: Data persistence across operations

## Writing New Tests

### Test Naming Convention

```python
def test_feature_specific_behavior():
    """Test description"""
    pass
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used by adding them as parameters:

```python
def test_with_mock_conversation(mock_conversation):
    """Test using mock conversation fixture"""
    assert mock_conversation is not None
```

### Async Test Example

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async function"""
    consumer = GeminiLiveConsumer()
    await consumer._send_text_to_gemini("Hello")
```

### Mocking External Dependencies

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_with_mocked_api(mock_gemini_client):
    """Test with mocked Gemini API"""
    with patch('assistant.consumers.genai.Client', return_value=mock_gemini_client):
        consumer = GeminiLiveConsumer()
        # ... test code
```

## Coverage Goals

The project aims for:
- **Overall coverage**: 80%+
- **Critical modules**: 90%+
  - `assistant/consumers.py`
  - `app/models.py`

## Common Test Fixtures

### mock_gemini_client
Mock Gemini API client with async methods

### mock_conversation
Mock MongoDB GeminiConversation document

### mock_session
Mock Gemini Live session

### sample_audio_bytes
Sample audio data for testing

### sample_image_bytes_jpeg
Sample JPEG image data

### sample_image_bytes_png
Sample PNG image data

## Debugging Tests

### Run with Print Statements

```bash
pytest tests/test_consumers.py -v -s
```

The `-s` flag shows print statements and logging output.

### Run Single Test with Debugging

```bash
pytest tests/test_consumers.py::TestGeminiLiveConsumerConnect::test_connect_success -v -s --pdb
```

The `--pdb` flag drops into Python debugger on failures.

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```bash
pytest --cov=assistant --cov=app --cov-report=xml --cov-report=term-missing
```

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Avoid repeated setup code
3. **Mock External Calls**: Don't call real APIs in tests
4. **Test Edge Cases**: Include error scenarios
5. **Clear Assertions**: Make test intent obvious
6. **Keep Tests Fast**: Avoid slow operations
7. **Update Tests**: Keep tests in sync with code changes

## Troubleshooting

### AsyncIO Issues

If you see "RuntimeError: no running event loop":
- Ensure `@pytest.mark.asyncio` decorator is present
- Check `asyncio_mode = auto` in `pytest.ini`

### Import Errors

If tests can't import modules:
- Run from project root directory
- Ensure `tests/__init__.py` exists
- Check Python path includes project root

### Fixture Not Found

If a fixture is not recognized:
- Verify it's defined in `conftest.py`
- Check spelling matches exactly
- Ensure `conftest.py` is in the tests directory

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
