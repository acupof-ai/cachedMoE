"""Text/tool adapters over the web bridge's single engine queue."""
import json
import time
import uuid
from pathlib import Path

MODEL = "deepseek-v4.1-flash"


def uid(prefix):
    return prefix + uuid.uuid4().hex


def text(value):
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return "\n".join(b.get("text", "") for b in value if b.get("type") in ("text", "input_text", "output_text"))


def normalize(path, body):
    if body.get("model", MODEL) != MODEL:
        raise ValueError("Unknown model; use " + MODEL)
    messages, tools = [], []
    if path == "/v1/responses":
        if body.get("previous_response_id"):
            raise ValueError("Send full input history; previous_response_id is unsupported")
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        inp = body.get("input", [])
        if isinstance(inp, str):
            inp = [{"role": "user", "content": inp}]
        for m in inp:
            kind = m.get("type", "message")
            if kind == "function_call":
                messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": m["call_id"], "type": "function", "function": {"name": m["name"], "arguments": m["arguments"]}}]})
            elif kind == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": m["call_id"], "content": text(m["output"])})
            elif kind == "message":
                messages.append({"role": "system" if m["role"] == "developer" else m["role"], "content": text(m.get("content"))})
            elif kind != "reasoning":
                raise ValueError("Unsupported Responses input: " + kind)
        def add_tool(t, namespace=None):
            if t["type"] == "namespace":
                for child in t["tools"]:
                    add_tool(child, t["name"])
                return
            if t["type"] != "function":
                raise ValueError("Only function tools are supported; disable custom/freeform tools")
            tool = {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
            if namespace:
                tool["namespace"] = namespace
            tools.append(tool)
        for t in body.get("tools", []):
            add_tool(t)
    elif path == "/v1/messages":
        if body.get("system"):
            messages.append({"role": "system", "content": text(body["system"])})
        for m in body.get("messages", []):
            msg = {"role": m["role"], "content": text(m.get("content"))}
            blocks = m.get("content", [])
            if isinstance(blocks, list):
                calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}} for b in blocks if b["type"] == "tool_use"]
                if calls:
                    msg["tool_calls"] = calls
                results = [b for b in blocks if b["type"] == "tool_result"]
                if results:
                    msg["content_blocks"] = [b for b in blocks if b["type"] in ("text", "tool_result")]
            messages.append(msg)
        tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}} for t in body.get("tools", [])]
    else:
        messages = [dict(m, content=text(m.get("content"))) for m in body.get("messages", [])]
        tools = body.get("tools", [])
    if not messages:
        raise ValueError("messages/input must not be empty")
    if tools:
        messages.insert(0, {"role": "system", "content": "", "tools": tools})
    limit = int(body.get("max_output_tokens", body.get("max_completion_tokens", body.get("max_tokens", 512))))
    if not 1 <= limit <= 32768:
        raise ValueError("Output limit must be 1..32768")
    return {"api_messages": messages, "text": "API request", "think": False, "max_tokens": limit,
            "temperature": body.get("temperature", 0), "seed": body.get("seed", 42)}, bool(tools)


def result(path, msg, usage, finish, ident):
    calls = msg.get("tool_calls", [])
    content = msg.get("content", "")
    for c in calls:
        c.setdefault("id", uid("call_"))
        if c.get("namespace"):
            c["function"]["name"] = c["namespace"] + "::" + c["function"]["name"]
    if path == "/v1/messages":
        blocks = ([{"type": "text", "text": content}] if content or not calls else [])
        blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"])} for c in calls]
        return {"id": ident, "type": "message", "role": "assistant", "model": MODEL, "content": blocks,
                "stop_reason": "tool_use" if calls else "max_tokens" if finish == "length" else "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}
    if path == "/v1/responses":
        output = [{"id": uid("fc_"), "type": "function_call", "status": "completed", "call_id": c["id"], **dict(c["function"], **({"name": c["function"]["name"].split("::", 1)[1], "namespace": c["namespace"]} if c.get("namespace") else {}))} for c in calls]
        if content or not calls:
            output.insert(0, {"id": uid("msg_"), "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": content, "annotations": []}]})
        return {"id": ident, "object": "response", "created_at": int(time.time()), "status": "completed", "model": MODEL, "output": output, "parallel_tool_calls": False,
                "usage": {"input_tokens": usage[0], "output_tokens": usage[1], "total_tokens": sum(usage)}, "error": None, "incomplete_details": None}
    return {"id": ident, "object": "chat.completion", "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}, "finish_reason": "tool_calls" if calls else "length" if finish == "length" else "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def handle(h, path, body):
    session = uid("api_")
    audit = getattr(h.bridge.args, "api_audit_dir", None)
    if audit:
        Path(audit).mkdir(parents=True, exist_ok=True)
        Path(audit, session + ".request.json").write_text(json.dumps({"path": path, "body": body}, ensure_ascii=False, indent=2))
    try:
        req, has_tools = normalize(path, body)
    except (ValueError, KeyError, TypeError) as e:
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}}
        if audit:
            Path(audit, session + ".response.json").write_text(json.dumps({"status": 400, "body": error}, indent=2))
        return h._send(400, error)
    job = h.bridge.submit(session, req)
    stream = bool(body.get("stream"))
    ident = uid("msg_" if path.endswith("messages") else "resp_" if path.endswith("responses") else "chatcmpl_")
    seq = 0
    if stream:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.send_header("Cache-Control", "no-cache")
        h.end_headers()
    def emit(obj, event=None):
        nonlocal seq
        if event and path.endswith("responses"):
            obj = dict(obj, type=event, sequence_number=seq)
            seq += 1
        data = (("event: " + event + "\n") if event else "") + "data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n"
        data = data.encode()
        h.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
        h.wfile.flush()
    prompt, generated, done = 0, 0, None
    chat = path.endswith("completions")
    anthropic = path.endswith("messages")
    def delta(value):
        if chat:
            emit({"id": ident, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": value, "finish_reason": None}]})
    try:
        if stream and chat:
            delta({"role": "assistant"})
        if stream and anthropic:
            emit({"type": "message_start", "message": {"id": ident, "type": "message", "role": "assistant", "model": MODEL, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}}, "message_start")
            if not has_tools:
                emit({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}, "content_block_start")
        if stream and not chat and not anthropic:
            emit({"response": {"id": ident, "object": "response", "status": "in_progress", "output": []}}, "response.created")
        while True:
            ev = job.out.get()
            if ev is None:
                break
            if ev["event"] == "start":
                prompt = ev["prompt_tokens"]
            elif ev["event"] == "error":
                raise RuntimeError(ev["message"])
            elif ev["event"] == "token":
                generated += 1
                if stream and not has_tools:
                    value = ev["text"].replace("<｜end▁of▁sentence｜>", "")
                    if chat:
                        delta({"content": value})
                    elif anthropic:
                        emit({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": value}}, "content_block_delta")
            elif ev["event"] == "done":
                done = ev
        if done is None:
            raise RuntimeError("Engine returned no completion")
        obj = result(path, done["api_message"], (prompt, generated), done.get("finish"), ident)
        if audit:
            Path(audit, session + ".response.json").write_text(json.dumps({"body": obj, "engine": done}, ensure_ascii=False, indent=2))
        if not stream:
            return h._send(200, obj)
        if chat:
            if has_tools:
                delta({k: v for k, v in obj["choices"][0]["message"].items() if k != "role" and k != "tool_calls"})
                for i, c in enumerate(obj["choices"][0]["message"].get("tool_calls", [])):
                    delta({"tool_calls": [dict(c, index=i)]})
            emit({"id": ident, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": obj["choices"][0]["finish_reason"]}], "usage": obj["usage"]})
            emit("[DONE]")
        elif anthropic:
            if has_tools:
                for i, b in enumerate(obj["content"]):
                    emit({"type": "content_block_start", "index": i, "content_block": dict(b, **({"input": {}} if b["type"] == "tool_use" else {"text": ""}))}, "content_block_start")
                    d = {"type": "input_json_delta", "partial_json": json.dumps(b["input"])} if b["type"] == "tool_use" else {"type": "text_delta", "text": b["text"]}
                    emit({"type": "content_block_delta", "index": i, "delta": d}, "content_block_delta")
                    emit({"type": "content_block_stop", "index": i}, "content_block_stop")
            else:
                emit({"type": "content_block_stop", "index": 0}, "content_block_stop")
            emit({"type": "message_delta", "delta": {"stop_reason": obj["stop_reason"], "stop_sequence": None}, "usage": obj["usage"]}, "message_delta")
            emit({"type": "message_stop"}, "message_stop")
        else:
            for i, item in enumerate(obj["output"]):
                emit({"output_index": i, "item": dict(item, status="in_progress")}, "response.output_item.added")
                if item["type"] == "function_call":
                    emit({"item_id": item["id"], "output_index": i, "delta": item["arguments"]}, "response.function_call_arguments.delta")
                    emit({"item_id": item["id"], "output_index": i, "arguments": item["arguments"]}, "response.function_call_arguments.done")
                else:
                    part = item["content"][0]
                    emit({"item_id": item["id"], "output_index": i, "content_index": 0, "part": dict(part, text="")}, "response.content_part.added")
                    emit({"item_id": item["id"], "output_index": i, "content_index": 0, "delta": part["text"]}, "response.output_text.delta")
                    emit({"item_id": item["id"], "output_index": i, "content_index": 0, "text": part["text"]}, "response.output_text.done")
                    emit({"item_id": item["id"], "output_index": i, "content_index": 0, "part": part}, "response.content_part.done")
                emit({"output_index": i, "item": item}, "response.output_item.done")
            emit({"response": obj}, "response.completed")
        h.wfile.write(b"0\r\n\r\n")
        h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        h.bridge.cancel(session)
    except Exception as e:
        if stream:
            emit({"error": {"message": str(e)}}, "error")
            h.wfile.write(b"0\r\n\r\n")
        else:
            h._send(500, {"error": {"message": str(e)}})
