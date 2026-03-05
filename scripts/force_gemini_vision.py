import re

vision_path = r"d:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\app\multimodal\vision.py"
with open(vision_path, "r", encoding="utf-8") as f:
    vtext = f.read()

# Replace Modelslab init with Gemini properties
vtext = vtext.replace(
    'self.modelslab_key = os.getenv("MODELSLAB_API_KEY")',
    'self.gemini_key = os.getenv("GEMINI_API_KEY")\n        self.vision_cloud_model = os.getenv("VISION_CLOUD_MODEL", "gemini-2.5-flash")'
)
vtext = re.sub(
    r'\s+self\.modelslab_model_id = os\.getenv\("VISION_MODELSLAB_MODEL_ID".*?\n',
    '\n',
    vtext
)

# Strip out trailing ModelsLab logger info
vtext = re.sub(
    r'\s+if self\.modelslab_key:.*?logger\.info\(f"\[VISION\] ModelsLab vision enabled.*?\n',
    '\n        if self.gemini_key:\n            logger.info(f"[VISION] Gemini vision enabled model_id={self.vision_cloud_model}")\n',
    vtext, flags=re.DOTALL
)

# New Gemini direct Vision block
gemini_vision_block = """        # Prefer Gemini vision when key exists (fallback to local models if it fails).
        if self.gemini_key:
            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = "image/jpeg"
                if image_bytes.startswith(b"\\x89PNG"):
                    mime_type = "image/png"
                elif image_bytes.startswith(b"RIFF"):
                    mime_type = "image/webp"

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": b64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": max_new_tokens,
                        "temperature": 0.1
                    }
                }
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.vision_cloud_model}:generateContent?key={self.gemini_key}",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                try:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info("[VISION] Gemini vision success; returning remote response.")
                    return content.strip()
                except Exception as parse_e:
                    logger.warning(f"[VISION] Gemini vision parse error: {parse_e} raw={result}")
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(f"Gemini vision failed: {e}")
                logger.warning(f"[VISION] Gemini vision failed, falling back to local: {e}")"""

# Reconstruct Vision file
parts = vtext.split('# Prefer ModelsLab vision when key exists')
if len(parts) > 1:
    after_part = parts[1].split('image = Image.open(io.BytesIO(image_bytes)).convert("RGB")')
    vtext = parts[0] + gemini_vision_block + '\n\n        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")' + after_part[1]

with open(vision_path, "w", encoding="utf-8") as f:
    f.write(vtext)


routes_path = r"d:\WorkSpace\Enterprise-RAG-Operations-Agent_POC\app\api\routes.py"
with open(routes_path, "r", encoding="utf-8") as f:
    rtext = f.read()

stt_fix = '''                        backoff = min(backoff * 1.5, 15)
                    if status != "success" and not stt_result.get("output") and not stt_result.get("text"):
                        raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT fetch failed: {last_err or 'Timeout'}"))

            transcript = None
            if isinstance(stt_result.get("output"), list) and stt_result["output"]:
                out_val = stt_result["output"][0]
                if isinstance(out_val, str) and out_val.startswith("http"):
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(out_val, timeout=30) as txt_resp:
                                txt_resp.raise_for_status()
                                transcript = await txt_resp.text()
                    except Exception as e:
                        logger.warning(f"Failed to download STT text from {out_val}: {e}")
                else:
                    transcript = out_val

            if not transcript and isinstance(stt_result.get("text"), str):
                transcript = stt_result["text"]

            if not transcript:
                raise HTTPException(status_code=502, detail=_mask_secret(f"Modelslab STT did not return transcript. Raw: {stt_result}"))'''

rtext = re.sub(
    r'                        backoff = min\(backoff \* 1\.5, 15\)\n                    if status != "success" and not stt_result\.get\("output"\) and not stt_result\.get\("text"\):\n                        raise HTTPException\(status_code=502, detail=_mask_secret\(f"Modelslab STT fetch failed: \{last_err or \'Timeout\'\}"\)\)\n                transcript = stt_result\["text"\]\n            if not transcript:\n                raise HTTPException\(status_code=502, detail=_mask_secret\(f"Modelslab STT did not return transcript\. Raw: \{stt_result\}"\)\)',
    stt_fix,
    rtext,
    flags=re.DOTALL
)

with open(routes_path, "w", encoding="utf-8") as f:
    f.write(rtext)

print("Applied Gemini Vision and STT URL payload parsing logic.")
