ORDERSCAN – EXOTEL VOICEBOT PILOT (STAGE 1)

This service handles Exotel's bidirectional Voicebot WebSocket, caller speech
transcription, AI conversation and spoken responses. It uses Exotel's 8 kHz,
16-bit, mono PCM format and returns 100 ms audio chunks.

IMPORTANT LIMITATION:
This stage is for controlled media/conversation testing only. It does NOT save
a draft order and does NOT submit orders. Draft persistence must be connected
through a separate authenticated PHP endpoint after checking your existing
order format and review workflow. Do not connect this service to save_order.php.

DEPLOY AS A NEW, SEPARATE RENDER SERVICE. Do not overwrite the existing
orderscan-api service or root app.py/render.yaml/requirements.txt.

Files:
- ordscn_voice_app.py
- ordscn_voice_requirements.txt
- ordscn_voice_render.yaml

Render build command:
  pip install -r ordscn_voice_requirements.txt

Render start command:
  uvicorn ordscn_voice_app:app --host 0.0.0.0 --port $PORT

Set these private Render environment variables:
  OPENAI_API_KEY       your OpenAI API key
  EXOTEL_STREAM_USER   a new strong username for WSS Basic Auth
  EXOTEL_STREAM_PASS   a new long random password for WSS Basic Auth

Optional variables and defaults:
  ORDSCN_VOICE_MODEL=gpt-4.1-mini
  ORDSCN_STT_MODEL=gpt-4o-mini-transcribe
  ORDSCN_TTS_MODEL=gpt-4o-mini-tts
  ORDSCN_TTS_VOICE=alloy
  ORDSCN_MAX_CALL_SECONDS=900

After deploy, check:
  https://YOUR-NEW-SERVICE.onrender.com/health

In Exotel Voicebot applet, configure the URL as:
  wss://EXOTEL_STREAM_USER:EXOTEL_STREAM_PASS@YOUR-NEW-SERVICE.onrender.com/media

Use Exotel's 8 kHz default for the first test. Keep the Basic Auth credentials
private; never paste them into chat or commit them to GitHub.

Before calling any real customer, implement and enforce documented consent,
customer-level opt-out, and a human-reviewed draft order save path. KYC approval
alone does not establish that an individual customer has consented to calls.
