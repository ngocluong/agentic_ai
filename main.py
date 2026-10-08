import html
import io
import json
import time
from enum import Enum
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ai_businness_agent import agent
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from streaming_prompt import ask_ai, ask_ai_with_tools
import asyncio
import logging
import traceback
from fastapi import Query
from markitdown import MarkItDown

logging.basicConfig(
    filename="errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/chat")
async def chat():
  return FileResponse("main.html", media_type="text/html")

class ResponseFormat(str, Enum):
  json = "json"
  markdown = "markdown"
  plain = "plain"

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
request_log = {}  # session_id -> list of request timestamps

_STREAM_DONE = object()  # sentinel: stream ended with no chunks

def _render_html(value):
  if isinstance(value, dict):
    items = "".join(f"<li><strong>{html.escape(str(k))}:</strong> {_render_html(v)}</li>" for k, v in value.items())
    return f"<ul>{items}</ul>"
  if isinstance(value, list):
    items = "".join(f"<li>{_render_html(v)}</li>" for v in value)
    return f"<ul>{items}</ul>"
  return html.escape(str(value))

def _response_to_html(data):
  return f"<html><body><h1>Agent Response</h1>{_render_html(data)}</body></html>"

def check_rate_limit(session_id):
  now = time.time()
  timestamps = [t for t in request_log.get(session_id, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
  if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
    request_log[session_id] = timestamps
    return False
  timestamps.append(now)
  request_log[session_id] = timestamps
  return True

@app.get("/")
async def root():
  
    return {"message": "Hello World"}

@app.post("/query")
async def asking_agent(request: Request, format: ResponseFormat = Query(default=ResponseFormat.json)):
  try:
    body = await request.json()
  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Request body must be valid JSON")

  question = body.get("question", "").strip()
  if not question:
    raise HTTPException(status_code=400, detail="'question' field is required")

  session_id = body.get("session_id", "").strip()
  if not session_id:
    raise HTTPException(status_code=400, detail="'session_id' field is required")

  if not check_rate_limit(session_id):
    raise HTTPException(status_code=429, detail="Rate limit exceeded: max 10 requests per minute")

  try:
    raw = await asyncio.wait_for(asyncio.to_thread(agent, question), timeout=10.0)
  except asyncio.TimeoutError:
    logger.error(f"Timeout for question: {question}")
    raise HTTPException(
        status_code=504,
        detail="Gateway Timeout."
    )
  except Exception as e:
    logger.error(f"Unexpected error for question: {question}\n{traceback.format_exc()}")
    raise HTTPException(status_code=500, detail="Internal server error")

  if format == "json":
    try:
      json_response = json.loads(raw) if isinstance(raw, str) else raw
      # also parse the answer field if it's a string
      if isinstance(json_response.get("answer"), str):
        try:
          json_response["answer"] = json.loads(json_response["answer"])
        except (json.JSONDecodeError, TypeError):
          pass
    except (json.JSONDecodeError, TypeError):
      return raw
  elif format == "markdown":
    markitdown = MarkItDown()
    stream = io.BytesIO(_response_to_html(raw).encode("utf-8"))
    result = markitdown.convert_stream(stream, file_extension=".html")
    return PlainTextResponse(content=result.markdown, media_type="text/markdown")
  elif format == "plain":
    plain = raw if isinstance(raw, str) else json.dumps(raw)
    # strip any markdown symbols
    plain = re.sub(r'[#*`_~>\-]', '', plain)
    return PlainTextResponse(content=plain)
  if isinstance(json_response, dict) and json_response.get('error'):
    raise HTTPException(status_code=json_response['error_code'],
                        detail=json_response['error_message'])

  return json_response
  
@app.post("/stream_agent")
async def asking_stream_agent(request: Request):
  try:
    body = await request.json()
  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Request body must be valid JSON")
  question = body.get("question", "").strip()
  if not question:
    raise HTTPException(status_code=400, detail="'question' field is required")

  session_id = body.get("session_id", "").strip()
  if not session_id:
    raise HTTPException(status_code=400, detail="'session_id' field is required")

  if not check_rate_limit(session_id):
    raise HTTPException(status_code=429, detail="Rate limit exceeded: max 10 requests per minute")

  # return StreamingResponse(ask_ai(question), media_type="text/event-stream")
  
  stream = ask_ai_with_tools(question)
  try:
    first_chunk = await asyncio.wait_for(asyncio.to_thread(next, stream, _STREAM_DONE), timeout=10.0)
  except asyncio.TimeoutError:
    logger.error(f"Stream timeout for question: {question}")
    raise HTTPException(status_code=504, detail="Gateway Timeout.")
  except Exception:
    logger.error(f"Unexpected error for question: {question}\n{traceback.format_exc()}")
    raise HTTPException(status_code=500, detail="Internal server error")

  async def combined():
    if first_chunk is not _STREAM_DONE:
      yield first_chunk
      for chunk in stream:
        yield chunk

  return StreamingResponse(combined(), media_type="text/event-stream")
