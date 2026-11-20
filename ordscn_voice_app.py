import os, json, base64, asyncio, logging, wave, struct, math
from io import BytesIO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from openai import OpenAI

logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'))
log=logging.getLogger('ordscn-voice')
app=FastAPI(title='OrderScan Exotel Voicebot')
client=OpenAI() if os.getenv('OPENAI_API_KEY') else None
CHAT_MODEL=os.getenv('ORDSCN_VOICE_MODEL','gpt-4.1-mini')
STT_MODEL=os.getenv('ORDSCN_STT_MODEL','gpt-4o-mini-transcribe')
TTS_MODEL=os.getenv('ORDSCN_TTS_MODEL','gpt-4o-mini-tts')
TTS_VOICE=os.getenv('ORDSCN_TTS_VOICE','alloy')
AUTH_USER=os.getenv('EXOTEL_STREAM_USER','')
AUTH_PASS=os.getenv('EXOTEL_STREAM_PASS','')
MAX_SECONDS=int(os.getenv('ORDSCN_MAX_CALL_SECONDS','900'))
SILENCE_MS=int(os.getenv('ORDSCN_SILENCE_MS','850'))
MIN_SPEECH_MS=int(os.getenv('ORDSCN_MIN_UTTERANCE_MS','350'))
RATE=8000

def pcm_rms(buf):
    n=len(buf)//2
    if not n: return 0
    vals=struct.unpack('<%dh'%n,buf[:n*2])
    return int(math.sqrt(sum(v*v for v in vals)/n))

def wav_bytes(pcm):
    out=BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)
    return out.getvalue()

def resample_24k_to_8k(data):
    # TTS PCM is 24 kHz s16le; average each three samples for 8 kHz.
    n=len(data)//2
    if n<3: return b''
    samples=struct.unpack('<%dh'%n,data[:n*2])
    out=[]
    for i in range(0,(n//3)*3,3):
        v=int((samples[i]+samples[i+1]+samples[i+2])/3)
        out.append(max(-32768,min(32767,v)))
    return struct.pack('<%dh'%len(out),*out) if out else b''

def chunks(data,size=3200):
    size-=size%320
    for i in range(0,len(data),size):
        part=data[i:i+size]
        part=part[:len(part)-(len(part)%320)]
        if len(part)>=320: yield part

def authorized(ws):
    # Exotel supports Basic Auth headers for Voicebot WSS endpoints.
    import base64 as b64
    expected='Basic '+b64.b64encode(f'{AUTH_USER}:{AUTH_PASS}'.encode()).decode()
    return bool(AUTH_USER and AUTH_PASS and ws.headers.get('authorization','')==expected)

@app.get('/')
@app.get('/health')
async def health():
    return {'ok':True,'service':'ordscn-exotel-voice','openai_configured':client is not None,'websocket_path':'/media','audio_format':'pcm_s16le_8000_mono'}

@app.get('/ready')
async def ready():
    if client is None: raise HTTPException(503,'OPENAI_API_KEY is missing')
    return {'ok':True}

SYSTEM_PROMPT='''You are OrderScan's outbound pharmacy-order AI assistant. Identify yourself as an AI calling on behalf of the supplier. Speak simple natural Hinglish by default; understand Hindi, Marathi, English and Hinglish. First confirm you are speaking to the intended medical store/customer. Ask whether they want to order. Collect one product and quantity at a time, repeat it and ask for confirmation. Never guess medicine, strength, pack, quantity, price, stock, scheme or delivery. Preserve the words and units as spoken. If unclear, ask to repeat/spell. If asked not to call, acknowledge and end. Explain that this is only a draft for human review, not a placed order. Keep responses short. Return JSON only: {"reply": string, "intent": "continue|finish|opt_out|wrong_person", "items": [{"product_text": string, "quantity_text": string}], "notes": [string]}.'''

async def transcribe(pcm):
    wav=wav_bytes(pcm)
    result=await asyncio.to_thread(client.audio.transcriptions.create,model=STT_MODEL,file=('caller.wav',wav,'audio/wav'))
    return (getattr(result,'text','') or '').strip()

async def make_reply(history, transcript, meta):
    payload={'call':{'call_sid':meta.get('call_sid',''),'from':meta.get('from',''),'to':meta.get('to','')},'recent_conversation':history[-12:],'caller_said':transcript}
    def run():
        r=client.chat.completions.create(model=CHAT_MODEL,temperature=0.2,response_format={'type':'json_object'},messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
        return json.loads(r.choices[0].message.content)
    data=await asyncio.to_thread(run)
    if not isinstance(data,dict) or not isinstance(data.get('reply'),str): raise ValueError('Invalid voice model response')
    return data

async def speak(ws,text,stream_sid,seq):
    def synth():
        r=client.audio.speech.create(model=TTS_MODEL,voice=TTS_VOICE,input=text,response_format='pcm')
        return r.read()
    pcm8=resample_24k_to_8k(await asyncio.to_thread(synth))
    chunk_no=0
    for part in chunks(pcm8):
        seq+=1; chunk_no+=1
        await ws.send_json({'event':'media','sequence_number':seq,'stream_sid':stream_sid,'media':{'chunk':chunk_no,'timestamp':str((chunk_no-1)*100),'payload':base64.b64encode(part).decode()}})
    if chunk_no:
        seq+=1
        await ws.send_json({'event':'mark','sequence_number':seq,'stream_sid':stream_sid,'mark':{'name':f'reply-{seq}'}})
    return seq

@app.websocket('/media')
async def media(ws:WebSocket):
    if not authorized(ws):
        await ws.close(code=1008); return
    await ws.accept()
    if client is None:
        await ws.close(code=1011); return
    stream_sid=''; seq=1; meta={}; history=[]; audio=bytearray(); silence=0; speech=0; speaking=False
    started=asyncio.get_running_loop().time()
    try:
        async for raw in ws.iter_text():
            if asyncio.get_running_loop().time()-started>MAX_SECONDS: break
            try: msg=json.loads(raw)
            except Exception: continue
            event=msg.get('event','')
            if event=='connected': continue
            if event=='start':
                st=msg.get('start') or {}; stream_sid=msg.get('stream_sid') or st.get('stream_sid','')
                meta={'call_sid':st.get('call_sid',''),'from':st.get('from',''),'to':st.get('to',''),'custom_parameters':st.get('custom_parameters',{})}
                greeting='Namaste! Main OrderScan ki AI assistant hoon. Kya main aapke medical store se baat kar rahi hoon?'
                history.append({'role':'assistant','content':greeting}); speaking=True
                seq=await speak(ws,greeting,stream_sid,seq); continue
            if event=='mark': speaking=False; continue
            if event=='clear': speaking=False; audio.clear(); silence=speech=0; continue
            if event=='stop': break
            if event!='media' or speaking: continue
            try: part=base64.b64decode((msg.get('media') or {}).get('payload',''),validate=True)
            except Exception: continue
            part=part[:len(part)-(len(part)%2)]
            if not part: continue
            dur=max(1,len(part)//16)  # 16 bytes/ms at 8 kHz, 16-bit mono
            if pcm_rms(part)>=420:
                audio.extend(part); speech+=dur; silence=0
            elif speech:
                audio.extend(part); silence+=dur
            if speech>=MIN_SPEECH_MS and silence>=SILENCE_MS:
                utterance=bytes(audio); audio.clear(); speech=silence=0
                try:
                    text=await transcribe(utterance)
                    if not text: continue
                    log.info('call=%s caller=%s',meta.get('call_sid',''),text[:400])
                    history.append({'role':'user','content':text})
                    answer=await make_reply(history,text,meta); reply=answer['reply'].strip()
                    history.append({'role':'assistant','content':reply}); speaking=True
                    seq=await speak(ws,reply,stream_sid,seq)
                    if answer.get('intent') in ('finish','opt_out','wrong_person'):
                        log.info('conversation intent=%s call=%s',answer.get('intent'),meta.get('call_sid',''))
                except Exception:
                    log.exception('Voice turn failed')
                    speaking=True; seq=await speak(ws,'Maaf kijiye, awaaz clear nahi aayi. Kripya ek baar phir boliye.',stream_sid,seq)
    except WebSocketDisconnect: pass
    except Exception: log.exception('WebSocket session error')
    finally:
        # Deliberately no order submission and no draft persistence in this pilot stage.
        log.info('call ended call_sid=%s turns=%d',meta.get('call_sid',''),len(history))
