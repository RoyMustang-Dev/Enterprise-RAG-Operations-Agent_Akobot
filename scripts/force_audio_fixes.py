import re

# 1. Fix vision.py ModelsLab payload
vision_path = r"d:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\app\multimodal\vision.py"
with open(vision_path, "r", encoding="utf-8") as f:
    vtext = f.read()

# Modelslab Vision uses standard 'model' key, not 'model_id', in standard OpenAI format
vtext = vtext.replace(
    '                    "model_id": self.modelslab_model_id,',
    '                    "model": self.modelslab_model_id,'
)

with open(vision_path, "w", encoding="utf-8") as f:
    f.write(vtext)


# 2. Fix routes.py STT/TTS looping
routes_path = r"d:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\app\api\routes.py"
with open(routes_path, "r", encoding="utf-8") as f:
    rtext = f.read()

# ModelsLab TTS is taking longer to render than the 8-second fetch backoff
tts_fix = '''                # Some URLs are not immediately ready; retry on 404/5xx.
                backoff = 3
                last_err = None
                for attempt in range(12):
                    try:
                        async with session.get(audio_url, timeout=60) as audio_resp:
                            if audio_resp.status == 404:
                                last_err = f"404 for {audio_url}"
                                await asyncio.sleep(backoff)
                                backoff = min(backoff * 1.5, 12)
                                continue
                            audio_resp.raise_for_status()
                            return await audio_resp.read(), audio_url
                    except Exception as e:
                        last_err = str(e)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 1.5, 12)'''

rtext = re.sub(
    r"# Some URLs are not immediately ready; retry on 404.*?backoff = min\(backoff \* 2, 8\)",
    tts_fix,
    rtext,
    flags=re.DOTALL
)

# Fix STT processing loop missing output due to edge case where 'status' is missing but 'output' exists
stt_fix = '''            status = stt_result.get("status")
            if status == "processing" or stt_result.get("fetch_result"):
                fetch_url = stt_result.get("fetch_result")
                if fetch_url:
                    backoff = 3
                    last_err = None
                    for attempt in range(15):
                        await asyncio.sleep(backoff)
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(fetch_url, timeout=60) as fresp:
                                    fresp.raise_for_status()
                                    stt_result = await fresp.json()
                            status = stt_result.get("status")
                            logger.info(f"[STT] fetch_result poll {attempt+1}/15 status={status}")
                            if status == "success" or "output" in stt_result or "text" in stt_result:
                                break
                        except Exception as fe:
                            last_err = fe
                            logger.warning(f"[STT] fetch_result retry failed: {fe}")
                        backoff = min(backoff * 1.5, 15)
                    if status != "success" and not stt_result.get("output") and not stt_result.get("text"):
                        raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT fetch failed: {last_err or 'Timeout'}"))'''

rtext = re.sub(
    r"            status = stt_result\.get\(\"status\"\)\n            if status == \"processing\" and stt_result\.get\(\"fetch_result\"\):.*?raise HTTPException\(status_code=502, detail=_mask_secret\(f\"Modelslab STT fetch failed: \{last_err\}\"\)\)",
    stt_fix,
    rtext,
    flags=re.DOTALL
)

with open(routes_path, "w", encoding="utf-8") as f:
    f.write(rtext)

print("Applied Vision+STT+TTS fixes.")
