import os
import json
import base64
import asyncio
import logging
import wave
import struct
import math
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from openai import OpenAI


# ------------------------------------------------------------
# Logging and configuration
# ------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ordscn-voice")

app = FastAPI(title="OrderScan Exotel Voicebot")
client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

CHAT_MODEL = os.getenv("ORDSCN_VOICE_MODEL", "gpt-4.1-mini")
STT_MODEL = os.getenv("ORDSCN_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("ORDSCN_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("ORDSCN_TTS_VOICE", "alloy")

AUTH_USER = os.getenv("EXOTEL_STREAM_USER", "")
AUTH_PASS = os.getenv("EXOTEL_STREAM_PASS", "")

MAX_SECONDS = int(os.getenv("ORDSCN_MAX_CALL_SECONDS", "900"))
SILENCE_MS = int(os.getenv("ORDSCN_SILENCE_MS", "700"))
MIN_SPEECH_MS = int(os.getenv("ORDSCN_MIN_UTTERANCE_MS", "250"))
SPEECH_RMS_THRESHOLD = int(os.getenv("ORDSCN_SPEECH_RMS_THRESHOLD", "180"))

RATE = 8000
BYTES_PER_MS = 16
OUTGOING_CHUNK_BYTES = 3200


# ------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------

def pcm_rms(buf: bytes) -> int:
    sample_count = len(buf) // 2
    if sample_count == 0:
        return 0

    values = struct.unpack("<%dh" % sample_count, buf[:sample_count * 2])
    return int(math.sqrt(sum(value * value for value in values) / sample_count))


def wav_bytes(pcm: bytes) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE)
        wav_file.writeframes(pcm)
    return output.getvalue()


def resample_24k_to_8k(data: bytes) -> bytes:
    """Convert 24 kHz signed 16-bit mono PCM to 8 kHz signed 16-bit mono PCM."""
    sample_count = len(data) // 2
    if sample_count < 3:
        return b""

    samples = struct.unpack("<%dh" % sample_count, data[:sample_count * 2])
    output = []

    for index in range(0, (sample_count // 3) * 3, 3):
        average = int(sum(samples[index:index + 3]) / 3)
        output.append(max(-32768, min(32767, average)))

    if not output:
        return b""

    return struct.pack("<%dh" % len(output), *output)


def audio_chunks(data: bytes, size: int = OUTGOING_CHUNK_BYTES):
    """Yield PCM chunks aligned to 320-byte boundaries."""
    size = max(3200, min(100000, size))
    size -= size % 320

    usable_length = len(data) - (len(data) % 320)
    for offset in range(0, usable_length, size):
        part = data[offset:min(offset + size, usable_length)]
        if len(part) >= 320:
            yield part


# ------------------------------------------------------------
# Authentication and health endpoints
# ------------------------------------------------------------

def authorized(ws: WebSocket) -> bool:
    import base64 as b64

    expected = "Basic " + b64.b64encode(
        f"{AUTH_USER}:{AUTH_PASS}".encode()
    ).decode()

    return bool(
        AUTH_USER
        and AUTH_PASS
        and ws.headers.get("authorization", "") == expected
    )


@app.get("/")
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "ordscn-exotel-voice",
        "openai_configured": client is not None,
        "websocket_path": "/media",
        "audio_format": "pcm_s16le_8000_mono",
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
    }


@app.get("/ready")
async def ready():
    if client is None:
        raise HTTPException(503, "OPENAI_API_KEY is missing")
    if not AUTH_USER or not AUTH_PASS:
        raise HTTPException(503, "Exotel WebSocket Basic Auth is not configured")
    return {"ok": True}


# ------------------------------------------------------------
# Order conversation instructions
# ------------------------------------------------------------

SYSTEM_PROMPT = """You are OrderScan's outbound pharmacy-order AI assistant.
Identify yourself as an AI calling on behalf of the supplier.

Speak simple natural Hinglish by default; understand Hindi, Marathi,
English and Hinglish.

First confirm you are speaking to the intended medical store/customer.
Ask whether they want to order.

Collect one product and quantity at a time, repeat it and ask for confirmation.

Never guess medicine, strength, pack, quantity, price, stock, scheme
or delivery.

Preserve spoken words and units; never convert strips, boxes or tablets.

If unclear, ask the caller to repeat or spell the product.
If asked not to call, acknowledge and end.

Explain this is only a draft for human review, not a placed order.
Keep replies short and ask one question at a time.

Return JSON only:
{
  "reply": "short spoken reply",
  "intent": "continue|finish|opt_out|wrong_person",
  "items": [
    {
      "product_text": "as spoken",
      "quantity_text": "as spoken"
    }
  ],
  "notes": ["optional note"]
}
"""


# ------------------------------------------------------------
# OpenAI functions
# ------------------------------------------------------------

async def transcribe(pcm: bytes) -> str:
    wav = wav_bytes(pcm)

    def run():
        return client.audio.transcriptions.create(
            model=STT_MODEL,
            file=("caller.wav", wav, "audio/wav"),
        )

    result = await asyncio.to_thread(run)
    return (getattr(result, "text", "") or "").strip()


async def make_reply(history, transcript: str, meta: dict) -> dict:
    payload = {
        "call": {
            "call_sid": meta.get("call_sid", ""),
            "from": meta.get("from", ""),
            "to": meta.get("to", ""),
        },
        "recent_conversation": history[-12:],
        "caller_said": transcript,
    }

    def run():
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    data = await asyncio.to_thread(run)

    if not isinstance(data, dict) or not isinstance(data.get("reply"), str):
        raise ValueError("Invalid voice model response")

    return data


async def synthesize_pcm_8k(text: str) -> bytes:
    def run():
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="pcm",
        )
        return response.read()

    pcm_24k = await asyncio.to_thread(run)
    return resample_24k_to_8k(pcm_24k)


# ------------------------------------------------------------
# Exotel outbound audio
#
# Important protocol detail:
# Exotel's documented bot -> Exotel media message contains:
# event, stream_sid, and media.payload.
# Do not add inbound-only sequence_number/chunk/timestamp fields.
# ------------------------------------------------------------

async def send_speech(ws: WebSocket, text: str, stream_sid: str, state: dict):
    if state.get("send_task") is not None:
        old_task = state["send_task"]
        if not old_task.done():
            log.warning("Previous TTS task still running; not starting another")
            return

    state["tts_active"] = True
    mark_name = f"orderscan-{state['turn_number']}-done"
    mark_event = asyncio.Event()
    state["expected_mark"] = mark_name
    state["mark_event"] = mark_event

    try:
        log.info(
            "TTS generation starting stream=%s turn=%s chars=%d",
            stream_sid,
            state["turn_number"],
            len(text),
        )

        pcm_8k = await synthesize_pcm_8k(text)
        if not pcm_8k:
            raise ValueError("TTS returned empty audio")

        duration = len(pcm_8k) / (RATE * 2)
        chunks = list(audio_chunks(pcm_8k))

        log.info(
            "TTS generated stream=%s bytes=%d duration_seconds=%.2f chunks=%d",
            stream_sid,
            len(pcm_8k),
            duration,
            len(chunks),
        )

        loop = asyncio.get_running_loop()
        state["speaking_until"] = loop.time() + duration + 2.0

        for chunk_number, part in enumerate(chunks, start=1):
            # This is the documented bot -> Exotel media message shape.
            message = {
                "event": "media",
                "stream_sid": stream_sid,
                "media": {
                    "payload": base64.b64encode(part).decode("ascii"),
                },
            }

            log.info(
                "TTS media send begin stream=%s chunk=%d/%d bytes=%d",
                stream_sid,
                chunk_number,
                len(chunks),
                len(part),
            )

            await ws.send_json(message)

            log.info(
                "TTS media send ok stream=%s chunk=%d/%d",
                stream_sid,
                chunk_number,
                len(chunks),
            )

            # 8 kHz * 16-bit mono = 16,000 bytes per second.
            # Pace according to the actual PCM duration of this chunk.
            await asyncio.sleep(len(part) / (RATE * 2))

        # Send a mark after all audio chunks. Exotel echoes it after playback.
        await ws.send_json({
            "event": "mark",
            "stream_sid": stream_sid,
            "mark": {"name": mark_name},
        })

        log.info(
            "TTS all chunks sent stream=%s chunks=%d duration_seconds=%.2f mark=%s",
            stream_sid,
            len(chunks),
            duration,
            mark_name,
        )

        # The receive loop remains active while this task waits for the mark.
        try:
            await asyncio.wait_for(mark_event.wait(), timeout=5.0)
            log.info("TTS playback mark confirmed stream=%s mark=%s", stream_sid, mark_name)
        except asyncio.TimeoutError:
            log.warning(
                "TTS playback mark timeout stream=%s mark=%s; continuing after audio send",
                stream_sid,
                mark_name,
            )

    except asyncio.CancelledError:
        log.warning("TTS task cancelled stream=%s", stream_sid)
        raise
    except Exception:
        log.exception("TTS send failed stream=%s", stream_sid)
    finally:
        state["tts_active"] = False
        state["expected_mark"] = None
        state["mark_event"] = None
        state["speaking_until"] = 0.0


def start_speech_task(ws: WebSocket, text: str, stream_sid: str, state: dict):
    task = asyncio.create_task(send_speech(ws, text, stream_sid, state))
    state["send_task"] = task
    return task


# ------------------------------------------------------------
# Main Exotel WebSocket
# ------------------------------------------------------------

@app.websocket("/media")
async def media(ws: WebSocket):
    if not authorized(ws):
        log.warning("Rejected WebSocket: Basic Auth missing or invalid")
        await ws.close(code=1008)
        return

    await ws.accept()

    if client is None:
        log.error("OPENAI_API_KEY is missing")
        await ws.close(code=1011)
        return

    stream_sid = ""
    meta = {}
    history = []
    audio_buffer = bytearray()

    silence_ms = 0
    speech_ms = 0
    media_count = 0
    peak_rms = 0

    loop = asyncio.get_running_loop()
    call_started_at = loop.time()
    last_diagnostic_at = call_started_at

    state = {
        "tts_active": False,
        "speaking_until": 0.0,
        "expected_mark": None,
        "mark_event": None,
        "send_task": None,
        "turn_number": 0,
    }

    try:
        async for raw in ws.iter_text():
            now = loop.time()

            if now - call_started_at > MAX_SECONDS:
                log.info("Maximum call duration reached stream=%s", stream_sid)
                break

            try:
                message = json.loads(raw)
            except Exception:
                log.warning("Received non-JSON WebSocket message")
                continue

            event = message.get("event", "")

            if event == "connected":
                log.info(
                    "Exotel connected event stream=%s",
                    message.get("stream_sid", ""),
                )
                continue

            if event == "start":
                start_data = message.get("start") or {}
                stream_sid = message.get("stream_sid") or start_data.get("stream_sid", "")

                meta = {
                    "call_sid": start_data.get("call_sid", ""),
                    "from": start_data.get("from", ""),
                    "to": start_data.get("to", ""),
                    "custom_parameters": start_data.get("custom_parameters", {}),
                }

                log.info(
                    "Call started call_sid=%s from=%s to=%s stream=%s media_format=%s",
                    meta.get("call_sid"),
                    meta.get("from"),
                    meta.get("to"),
                    stream_sid,
                    start_data.get("media_format"),
                )

                greeting = (
                    "Namaste! Main OrderScan ki AI assistant hoon. "
                    "Kya main aapke medical store se baat kar rahi hoon? "
                    "Aap bol sakte hain, main sun rahi hoon."
                )

                history.append({"role": "assistant", "content": greeting})
                state["turn_number"] = 1
                start_speech_task(ws, greeting, stream_sid, state)
                continue

            if event == "mark":
                mark_name = (message.get("mark") or {}).get("name", "")
                log.info("Playback mark received stream=%s mark=%s", stream_sid, mark_name)

                if (
                    state.get("expected_mark") == mark_name
                    and state.get("mark_event") is not None
                ):
                    state["mark_event"].set()
                continue

            if event == "clear":
                log.info("Exotel clear event stream=%s", stream_sid)
                audio_buffer.clear()
                silence_ms = 0
                speech_ms = 0
                continue

            if event == "dtmf":
                log.info(
                    "DTMF received stream=%s digit=%s",
                    stream_sid,
                    (message.get("dtmf") or {}).get("digit", ""),
                )
                continue

            if event == "stop":
                log.info(
                    "Exotel stop event stream=%s detail=%s",
                    stream_sid,
                    message.get("stop"),
                )
                break

            if event != "media":
                log.info("Other Exotel event stream=%s event=%s", stream_sid, event)
                continue

            media_count += 1

            try:
                part = base64.b64decode(
                    (message.get("media") or {}).get("payload", ""),
                    validate=True,
                )
            except Exception:
                log.warning("Invalid media base64 stream=%s", stream_sid)
                continue

            part = part[:len(part) - (len(part) % 2)]
            if not part:
                continue

            rms = pcm_rms(part)
            peak_rms = max(peak_rms, rms)

            # Do not mistake the bot's own playback/echo for caller speech.
            if state["tts_active"] or now < state["speaking_until"]:
                if media_count % 50 == 0:
                    log.info(
                        "Caller media ignored during TTS stream=%s frames=%d rms=%d",
                        stream_sid,
                        media_count,
                        rms,
                    )
                continue

            duration_ms = max(1, len(part) // BYTES_PER_MS)

            if rms >= SPEECH_RMS_THRESHOLD:
                audio_buffer.extend(part)
                speech_ms += duration_ms
                silence_ms = 0
            elif speech_ms:
                audio_buffer.extend(part)
                silence_ms += duration_ms

            if media_count % 50 == 0 or now - last_diagnostic_at >= 5:
                log.info(
                    "Audio diagnostic stream=%s frames=%d rms=%d peak=%d speech_ms=%d silence_ms=%d bytes=%d",
                    stream_sid,
                    media_count,
                    rms,
                    peak_rms,
                    speech_ms,
                    silence_ms,
                    len(audio_buffer),
                )
                last_diagnostic_at = now

            if speech_ms >= MIN_SPEECH_MS and silence_ms >= SILENCE_MS:
                utterance = bytes(audio_buffer)
                audio_buffer.clear()

                captured_speech_ms = speech_ms
                silence_ms = 0
                speech_ms = 0
                peak_rms = 0

                log.info(
                    "Utterance captured stream=%s bytes=%d speech_ms=%d",
                    stream_sid,
                    len(utterance),
                    captured_speech_ms,
                )

                try:
                    transcript = await transcribe(utterance)

                    if not transcript:
                        reply = (
                            "Maaf kijiye, awaaz clear nahi aayi. "
                            "Kripya ek baar phir boliye."
                        )
                        history.append({"role": "assistant", "content": reply})
                    else:
                        log.info(
                            "Caller transcription call_sid=%s text=%s",
                            meta.get("call_sid", ""),
                            transcript[:400],
                        )
                        history.append({"role": "user", "content": transcript})

                        answer = await make_reply(history, transcript, meta)
                        reply = answer["reply"].strip()
                        history.append({"role": "assistant", "content": reply})

                        log.info(
                            "Assistant reply call_sid=%s intent=%s text=%s",
                            meta.get("call_sid", ""),
                            answer.get("intent", "continue"),
                            reply[:400],
                        )

                    state["turn_number"] += 1
                    start_speech_task(ws, reply, stream_sid, state)

                except Exception:
                    log.exception("Voice turn failed stream=%s", stream_sid)
                    fallback = (
                        "Maaf kijiye, thodi technical dikkat aa gayi. "
                        "Kripya apni baat ek baar phir batayein."
                    )
                    state["turn_number"] += 1
                    start_speech_task(ws, fallback, stream_sid, state)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected stream=%s", stream_sid)
    except Exception:
        log.exception("WebSocket session error stream=%s", stream_sid)
    finally:
        send_task = state.get("send_task")
        if send_task is not None and not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("Error while cleaning up TTS task stream=%s", stream_sid)

        # Draft-only pilot: deliberately no order submission or persistence.
        log.info(
            "Call ended call_sid=%s stream=%s turns=%d media_frames=%d",
            meta.get("call_sid", ""),
            stream_sid,
            len(history),
            media_count,
        )
