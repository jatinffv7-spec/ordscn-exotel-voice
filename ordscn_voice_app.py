import os, json, base64, asyncio, logging, wave, struct, math
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from openai import OpenAI

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ordscn-voice")
app = FastAPI(title="OrderScan Exotel Voicebot")
client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

CHAT_MODEL = os.getenv("ORDSCN_VOICE_MODEL", "gpt-4.1-mini")
STT_MODEL = os.getenv("ORDSCN_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("ORDSCN_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("ORDSCN_TTS_VOICE", "alloy")
AUTH_USER, AUTH_PASS = os.getenv("EXOTEL_STREAM_USER", ""), os.getenv("EXOTEL_STREAM_PASS", "")
MAX_SECONDS = int(os.getenv("ORDSCN_MAX_CALL_SECONDS", "900"))
SILENCE_MS = int(os.getenv("ORDSCN_SILENCE_MS", "700"))
MIN_SPEECH_MS = int(os.getenv("ORDSCN_MIN_UTTERANCE_MS", "250"))
SPEECH_RMS_THRESHOLD = int(os.getenv("ORDSCN_SPEECH_RMS_THRESHOLD", "180"))
RATE = 8000
BYTES_PER_MS = 16

def pcm_rms(buf):
    n = len(buf) // 2
    if not n: return 0
    vals = struct.unpack("<%dh" % n, buf[:n * 2])
    return int(math.sqrt(sum(v * v for v in vals) / n))

def wav_bytes(pcm):
    out = BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)
    return out.getvalue()

def resample_24k_to_8k(data):
    n = len(data) // 2
    if n < 3: return b""
    samples = struct.unpack("<%dh" % n, data[:n * 2])
    out = [max(-32768, min(32767, int(sum(samples[i:i+3]) / 3)))
           for i in range(0, (n // 3) * 3, 3)]
    return struct.pack("<%dh" % len(out), *out) if out else b""

def audio_chunks(data, size=3200):
    size = max(3200, min(100000, size)); size -= size % 320
    usable = len(data) - len(data) % 320
    # Keep every transmitted chunk aligned to 320 bytes.
    for i in range(0, usable, size):
        part = data[i:min(i + size, usable)]
        if len(part) >= 320: yield part

def authorized(ws):
    import base64 as b64
    expected = "Basic " + b64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode()).decode()
    return bool(AUTH_USER and AUTH_PASS and ws.headers.get("authorization", "") == expected)

@app.get("/")
@app.get("/health")
async def health():
    return {"ok": True, "service": "ordscn-exotel-voice", "openai_configured": client is not None,
            "websocket_path": "/media", "audio_format": "pcm_s16le_8000_mono",
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD}

@app.get("/ready")
async def ready():
    if client is None: raise HTTPException(503, "OPENAI_API_KEY is missing")
    if not AUTH_USER or not AUTH_PASS: raise HTTPException(503, "Exotel WebSocket Basic Auth is not configured")
    return {"ok": True}

@app.get("/exotel/resolve")
async def exotel_resolve(request: Request):
    """Return this call's WSS URL for Exotel Voicebot dynamic-URL mode.

    Exotel supplies CallSid/CallFrom/CallTo as query parameters. Only forward
    minimal per-call metadata; never expose API credentials in this URL.
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit, quote
    call_sid = (request.query_params.get("CallSid") or "").strip()
    call_to = (request.query_params.get("CallTo") or request.query_params.get("To") or "").strip()
    call_from = (request.query_params.get("CallFrom") or request.query_params.get("From") or "").strip()
    direction = (request.query_params.get("Direction") or "").strip()
    if not call_sid or not call_to:
        raise HTTPException(400, "Missing Exotel CallSid or CallTo")
    # Exotel documents a maximum of three custom stream parameters.
    params = {"sample-rate": "8000", "customer_phone": call_from,
              "exotel_call_sid": call_sid, "customer_from": call_from}
    ws_base = os.getenv("EXOTEL_MEDIA_WSS_URL", "wss://ordscn-exotel-voicebot.onrender.com/media")
    parts = urlsplit(ws_base)
    if parts.scheme != "wss" or not parts.netloc:
        raise HTTPException(503, "EXOTEL_MEDIA_WSS_URL must be a valid wss:// URL")
    # Exotel's Voicebot applet supports Basic Auth credentials in the WSS URL;
    # Exotel forwards them as an Authorization header to the WebSocket service.
    if AUTH_USER and AUTH_PASS and "@" not in parts.netloc:
        host = parts.netloc
        netloc = quote(AUTH_USER, safe="") + ":" + quote(AUTH_PASS, safe="") + "@" + host
        parts = parts._replace(netloc=netloc)
    query = dict([item.split("=", 1) for item in parts.query.split("&") if "=" in item])
    query.update(params)
    resolved = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    log.info("Resolved Exotel Voicebot endpoint call_sid=%s direction=%s", call_sid, direction)
    return {"url": resolved}

SYSTEM_PROMPT = """You are OrderScan's outbound pharmacy-order AI assistant. Identify yourself as an AI calling on behalf of the supplier.
Speak simple natural Hinglish by default; understand Hindi, Marathi, English and Hinglish.
First confirm you are speaking to the intended medical store/customer. Ask whether they want to order.
Collect one product and quantity at a time, repeat it and ask for confirmation.
Never guess medicine, strength, pack, quantity, price, stock, scheme or delivery.
Preserve spoken words and units; never convert strips, boxes or tablets.
If unclear, ask the caller to repeat or spell the product. If asked not to call, acknowledge and end.
Explain this is only a draft for human review, not a placed order. Keep replies short and ask one question at a time.
Return JSON only: {"reply": string, "intent": "continue|finish|opt_out|wrong_person",
"items": [{"product_text": string, "quantity_text": string}], "notes": [string]}."""

async def transcribe(pcm):
    wav = wav_bytes(pcm)
    def run():
        return client.audio.transcriptions.create(model=STT_MODEL, file=("caller.wav", wav, "audio/wav"))
    result = await asyncio.to_thread(run)
    return (getattr(result, "text", "") or "").strip()

async def make_reply(history, transcript, meta):
    payload = {"call": {"call_sid": meta.get("call_sid", ""), "from": meta.get("from", ""),
                        "to": meta.get("to", "")},
               "recent_conversation": history[-12:], "caller_said": transcript}
    def run():
        r = client.chat.completions.create(model=CHAT_MODEL, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        return json.loads(r.choices[0].message.content)
    data = await asyncio.to_thread(run)
    if not isinstance(data, dict) or not isinstance(data.get("reply"), str):
        raise ValueError("Invalid voice model response")
    return data

async def speak(ws, text, stream_sid, seq):
    def synth():
        r = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="pcm"
        )
        return r.read()

    pcm8 = resample_24k_to_8k(await asyncio.to_thread(synth))
    if not pcm8:
        raise ValueError("TTS returned empty audio")

    duration = len(pcm8) / (RATE * 2)

    for part in audio_chunks(pcm8):
        await ws.send_json({
            "event": "media",
            "stream_sid": stream_sid,
            "media": {
                "payload": base64.b64encode(part).decode("ascii")
            }
        })

    seq += 1
    mark_name = f"reply-{seq}"
    await ws.send_json({
        "event": "mark",
        "stream_sid": stream_sid,
        "mark": {
            "name": mark_name
        }
    })

    log.info(
        "TTS sent stream=%s chars=%d audio_seconds=%.2f mark=%s",
        stream_sid, len(text), duration, mark_name
    )
    return seq, duration

@app.websocket("/media")
async def media(ws: WebSocket):
    log.info("WebSocket request reached /media")
    if not authorized(ws):
        log.warning("Rejected WebSocket: Basic Auth missing or invalid")
        await ws.close(code=1008); return
    await ws.accept()
    if client is None:
        log.error("OPENAI_API_KEY is missing"); await ws.close(code=1011); return

    stream_sid, seq, meta = "", 1, {}
    history, audio = [], bytearray()
    silence_ms = speech_ms = media_count = 0
    speaking_until, expected_mark = 0.0, None
    started = asyncio.get_running_loop().time()
    last_diag, peak_rms = started, 0

    try:
        async for raw in ws.iter_text():
            now = asyncio.get_running_loop().time()
            if now - started > MAX_SECONDS:
                log.info("Maximum call duration reached stream=%s", stream_sid); break
            try: msg = json.loads(raw)
            except Exception:
                log.warning("Received non-JSON WebSocket message"); continue
            event = msg.get("event", "")

            if event == "connected":
                log.info("Exotel connected stream=%s", msg.get("stream_sid", "")); continue

            if event == "start":
                st = msg.get("start") or {}
                stream_sid = msg.get("stream_sid") or st.get("stream_sid", "")
                meta = {"call_sid": st.get("call_sid", ""), "from": st.get("from", ""),
                        "to": st.get("to", ""), "custom_parameters": st.get("custom_parameters", {})}
                cp = meta["custom_parameters"] if isinstance(meta["custom_parameters"], dict) else {}
                meta["customer_phone"] = cp.get("customer_phone") or meta.get("from", "")
                meta["exotel_call_sid"] = cp.get("exotel_call_sid", meta.get("call_sid", ""))
                meta["customer_from"] = cp.get("customer_from") or meta.get("from", "")
                log.info("Call started call_sid=%s from=%s to=%s stream=%s media_format=%s",
                         meta.get("call_sid"), meta.get("from"), meta.get("to"), stream_sid, st.get("media_format"))
                greeting = ("Namaste! Main OrderScan ki AI assistant hoon. "
                            "Kya main aapke medical store se baat kar rahi hoon? "
                            "Aap bol sakte hain, main sun rahi hoon.")
                history.append({"role": "assistant", "content": greeting})
                seq, duration = await speak(ws, greeting, stream_sid, seq)
                # Time fallback is important: don't remain deaf if a mark echo is missed.
                speaking_until = asyncio.get_running_loop().time() + duration + 0.6
                expected_mark = f"reply-{seq}"
                continue

            if event == "mark":
                mark = (msg.get("mark") or {}).get("name", "")
                log.info("Playback mark received stream=%s mark=%s", stream_sid, mark)
                if not expected_mark or mark == expected_mark:
                    speaking_until = min(speaking_until, asyncio.get_running_loop().time())
                    expected_mark = None
                continue
            if event == "clear":
                log.info("Exotel clear event stream=%s", stream_sid)
                speaking_until = 0; expected_mark = None; audio.clear(); silence_ms = speech_ms = 0
                continue
            if event == "dtmf":
                log.info("DTMF received stream=%s digit=%s", stream_sid, (msg.get("dtmf") or {}).get("digit", "")); continue
            if event == "stop":
                log.info("Exotel stop event stream=%s detail=%s", stream_sid, msg.get("stop")); break
            if event != "media":
                log.info("Other Exotel event stream=%s event=%s", stream_sid, event); continue

            media_count += 1
            try: part = base64.b64decode((msg.get("media") or {}).get("payload", ""), validate=True)
            except Exception:
                log.warning("Invalid media base64 stream=%s", stream_sid); continue
            part = part[:len(part) - (len(part) % 2)]
            if not part: continue
            rms = pcm_rms(part); peak_rms = max(peak_rms, rms)

            # Ignore playback echo only for the estimated TTS duration.
            # After that, accept caller audio even if Exotel never echoes mark.
            if now < speaking_until:
                if media_count % 50 == 0:
                    log.info("Media during TTS stream=%s frames=%d rms=%d", stream_sid, media_count, rms)
                continue

            duration_ms = max(1, len(part) // 16)
            if rms >= SPEECH_RMS_THRESHOLD:
                audio.extend(part); speech_ms += duration_ms; silence_ms = 0
            elif speech_ms:
                audio.extend(part); silence_ms += duration_ms

            if media_count % 50 == 0 or now - last_diag >= 5:
                log.info("Audio diagnostic stream=%s frames=%d rms=%d peak=%d speech_ms=%d silence_ms=%d bytes=%d",
                         stream_sid, media_count, rms, peak_rms, speech_ms, silence_ms, len(audio))
                last_diag = now

            if speech_ms >= MIN_SPEECH_MS and silence_ms >= SILENCE_MS:
                utterance = bytes(audio); audio.clear()
                captured_speech_ms = speech_ms
                silence_ms = speech_ms = 0; peak_rms = 0
                log.info("Utterance captured stream=%s bytes=%d speech_ms=%d",
                         stream_sid, len(utterance), captured_speech_ms)
                text = ""
                answer = {}
                try:
                    text = await transcribe(utterance)
                    if not text:
                        reply = "Maaf kijiye, awaaz clear nahi aayi. Kripya ek baar phir boliye."
                        history.append({"role": "assistant", "content": reply})
                    else:
                        log.info("Caller transcription call_sid=%s text=%s", meta.get("call_sid", ""), text[:400])
                        history.append({"role": "user", "content": text})
                        answer = await make_reply(history, text, meta)
                        reply = answer["reply"].strip()
                        history.append({"role": "assistant", "content": reply})
                        log.info("Assistant reply call_sid=%s intent=%s text=%s",
                                 meta.get("call_sid", ""), answer.get("intent", "continue"), reply[:400])
                    seq, duration = await speak(ws, reply, stream_sid, seq)
                    speaking_until = asyncio.get_running_loop().time() + duration + 0.6
                    expected_mark = f"reply-{seq}"
                    if answer.get("intent") in ("finish", "opt_out", "wrong_person"):
                        log.info("Conversation intent=%s call=%s", answer.get("intent"), meta.get("call_sid", ""))
                except Exception:
                    log.exception("Voice turn failed stream=%s", stream_sid)
                    fallback = "Maaf kijiye, thodi technical dikkat aa gayi. Kripya apni baat ek baar phir batayein."
                    seq, duration = await speak(ws, fallback, stream_sid, seq)
                    speaking_until = asyncio.get_running_loop().time() + duration + 0.6
                    expected_mark = f"reply-{seq}"

    except WebSocketDisconnect:
        log.info("WebSocket disconnected stream=%s", stream_sid)
    except Exception:
        log.exception("WebSocket session error stream=%s", stream_sid)
    finally:
        # Draft-only pilot: deliberately no order submission or persistence.
        log.info("Call ended call_sid=%s stream=%s turns=%d media_frames=%d",
                 meta.get("call_sid", ""), stream_sid, len(history), media_count)
