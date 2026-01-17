# Screen-Aware Background Voice Assistant

A real-time voice assistant powered by Google Gemini Multimodal Live API that can see your screen and respond with audio. Built with Django Channels for WebSocket streaming.

## Features

- **Real-time Audio Streaming** - Bidirectional audio with Gemini Live API
- **Screen Awareness** - Send screenshots for visual context
- **Google Search Integration** - Model can search the web when needed
- **WebSocket Communication** - Low-latency streaming via Django Channels

## Tech Stack

- **Backend:** Python 3.11+, Django 5.x
- **Async:** Django Channels + Daphne (ASGI)
- **AI:** Google Gemini 2.0 Flash (Multimodal Live API)
- **SDK:** `google-genai`

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### 2. Installation

```bash
# Clone and enter the project
cd mchacks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the Server

```bash
# Apply migrations (first time only)
python manage.py migrate

# Start the ASGI server
daphne -b 0.0.0.0 -p 8000 app.asgi:application
```

The WebSocket endpoint will be available at: `ws://localhost:8000/ws/live-session/`

## Testing with Mock Client

The included `mock_client.py` captures your screen and microphone to test the full pipeline.

### Install Additional Dependencies

```bash
# macOS - Install portaudio first (required for pyaudio)
brew install portaudio

# Then install Python packages
pip install pyaudio mss pillow websockets
```

<details>
<summary>Installation for other platforms</summary>

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio mss pillow websockets
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
pip install mss pillow websockets
```

</details>

### Run the Client

```bash
# In a new terminal (with server running)
python mock_client.py
```

Options:
- `--server ws://your-server:8000/ws/live-session/` - Custom server URL

The client will:
1. Connect to your Django WebSocket server
2. Capture screen every 2 seconds
3. Stream microphone audio continuously
4. Play back Gemini's audio responses
5. Accept text input for testing without a microphone

## WebSocket Protocol

### Client → Server

```json
{
  "audio_chunk": "<base64 encoded PCM audio at 16kHz>",
  "screen_frame": "<base64 encoded JPEG/PNG>",
  "text": "optional text message"
}
```

### Server → Client

```json
{"type": "connection_established", "message": "Connected to Gemini Live session"}
{"type": "audio_response", "audio_data": "<base64>", "mime_type": "audio/pcm"}
{"type": "text_response", "text": "Assistant's text response"}
{"type": "turn_complete"}
{"type": "tool_call", "tool": "google_search", "status": "executing"}
{"type": "error", "message": "Error description"}
```

## Project Structure

```
mchacks/
├── app/                    # Django project settings
│   ├── asgi.py            # ASGI config with Channels routing
│   ├── settings.py        # Django settings + Channels config
│   └── urls.py            # HTTP URL routing
├── assistant/             # Voice assistant app
│   ├── consumers.py       # GeminiLiveConsumer WebSocket handler
│   └── routing.py         # WebSocket URL routing
├── mock_client.py         # Test client with screen/audio capture
├── requirements.txt       # Python dependencies
├── manage.py
└── .env                   # Your API key (not in git)
```

## Configuration

### Model Settings (in `assistant/consumers.py`)

```python
MODEL_ID = "gemini-2.0-flash-exp"
SYSTEM_INSTRUCTION = "You are a helpful background assistant..."
```

### Audio Settings (in `mock_client.py`)

```python
SAMPLE_RATE = 16000          # Input audio sample rate
OUTPUT_SAMPLE_RATE = 24000   # Gemini output sample rate
SCREEN_CAPTURE_INTERVAL = 2.0 # Seconds between screenshots
```

## Production Deployment

For production, update `settings.py`:

```python
# Use Redis for channel layers
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("redis", 6379)],
        },
    },
}
```

Install Redis backend:
```bash
pip install channels-redis
```

## Troubleshooting

### "GOOGLE_API_KEY environment variable not set"
- Ensure `.env` file exists in project root
- Check the key format: `GOOGLE_API_KEY=AIza...`
- Restart the server after adding the key

### pyaudio installation fails
- **macOS:** `brew install portaudio` first
- **Linux:** `sudo apt-get install portaudio19-dev`
- **Windows:** Use `pipwin install pyaudio`

### WebSocket connection refused
- Ensure server is running with `daphne` (not `runserver`)
- Check `ALLOWED_HOSTS` includes your client's host
- Verify the WebSocket URL path: `ws://host:port/ws/live-session/`
