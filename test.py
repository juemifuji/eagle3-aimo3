from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Optional

from jupyter_client import KernelManager
from openai import OpenAI

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolNamespaceConfig,
    load_harmony_encoding,
)

# -------------------------
# Env / logging
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def setup_env() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    # If you need custom tiktoken encodings path:
    os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", "/path/tiktoken_encodings")


setup_env()


# -------------------------
# Minimal Sandbox (Jupyter)
# -------------------------
class LocalJupyterSandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        self._owns_kernel = False
        self._km: Optional[KernelManager] = None
        self._client = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "0"
        env["JUPYTER_PLATFORM_DIRS"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        km = KernelManager()
        km.shell_port = ports[0]
        km.iopub_port = ports[1]
        km.stdin_port = ports[2]
        km.hb_port = ports[3]
        km.control_port = ports[4]

        km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        client = km.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=self._timeout)

        self._km = km
        self._client = client
        self._owns_kernel = True

        # common imports (optional)
        self.execute(
            "import math\n"
            "import sympy\n"
            "import numpy as np\n"
            "import itertools\n"
            "import collections\n"
        )

    def _format_error(self, traceback_list: list[str]) -> str:
        clean_lines = []
        for frame in traceback_list:
            frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in frame and "ipython-input" not in frame:
                continue
            clean_lines.append(frame)
        return "".join(clean_lines)

    @staticmethod
    def _ensure_last_print(code: str) -> str:
        lines = code.strip().split("\n")
        if not lines:
            return code
        last = lines[-1].strip()
        if not last or last.startswith("#"):
            return code
        if "print" in last or last.startswith(("import ", "from ", "%")):
            return code
        lines[-1] = f"print({last})"
        return "\n".join(lines)

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        if self._client is None or self._km is None:
            return "[ERROR] Jupyter kernel not initialized."

        client = self._client
        effective_timeout = timeout or self._timeout

        final_code = self._ensure_last_print(code)
        msg_id = client.execute(final_code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start = time.time()

        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout} seconds"

            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                stderr_parts.append(self._format_error(content.get("traceback", [])))

            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts).rstrip()
        stderr = "".join(stderr_parts).rstrip()

        if stderr:
            return (stdout + "\n" + stderr).strip() if stdout else stderr
        return stdout if stdout else "[WARN] No output. Use print() to see results."

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def __del__(self):
        self.close()


# -------------------------
# Harmony template
# -------------------------
def build_messages(system_prompt: str, user_prompt: str, tool_config: ToolNamespaceConfig) -> list[Message]:
    system_content = (
        SystemContent.new()
        .with_model_identity(system_prompt)
        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        .with_tools(tool_config)
    )
    system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)
    user_msg = Message.from_role_and_content(Role.USER, user_prompt)
    return [system_msg, user_msg]


def tool_namespace_for_python(tool_prompt: str) -> ToolNamespaceConfig:
    # NOTE: Tool definitions list can be empty; Harmony uses recipient routing.
    return ToolNamespaceConfig(name="python", description=tool_prompt, tools=[])


def make_tool_response(output: str, channel: Optional[str] = None) -> Message:
    content = TextContent(text=output)
    author = Author(role=Role.TOOL, name="python")
    msg = Message(author=author, content=[content]).with_recipient("assistant")
    if channel:
        msg = msg.with_channel(channel)
    return msg


# -------------------------
# vLLM server control
# -------------------------
def start_vllm_server(
    model_path: str,
    served_model_name: str,
    host: str,
    port: int,
    tp: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    dtype: str,
    kv_cache_dtype: str,
    max_model_len: int,
    stream_interval: int,
    speculative_config_json: Optional[str],
    log_path: str,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--max-model-len",
        str(max_model_len),
        "--async-scheduling",
        "--stream-interval",
        str(stream_interval),
    ]
    if speculative_config_json:
        cmd += ["--speculative-config", speculative_config_json]

    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._logf = logf  # type: ignore[attr-defined]
    return proc


def wait_server_ready(client: OpenAI, proc: subprocess.Popen, log_path: str, timeout_s: int = 180) -> None:
    log("Waiting for vLLM server...")
    start = time.time()
    while time.time() - start < timeout_s:
        rc = proc.poll()
        if rc is not None:
            with contextlib.suppress(Exception):
                proc._logf.flush()  # type: ignore[attr-defined]
            with open(log_path, "r") as f:
                logs = f.read()
            raise RuntimeError(f"Server exited early (code={rc}). Logs:\n{logs}")

        try:
            client.models.list()
            log("Server is ready.\n")
            return
        except Exception:
            time.sleep(1)

    raise RuntimeError(f"Server not ready after {timeout_s}s. Check {log_path}")


# -------------------------
# Single-query tool-loop inference
# -------------------------
def run_single_query(
    client: OpenAI,
    encoding_name: HarmonyEncodingName,
    served_model_name: str,
    system_prompt: str,
    tool_prompt: str,
    user_query: str,
    stop_token_ids: list[int],
    temperature: float,
    top_p: float,
    min_p: float,
    max_model_len: int,
) -> str:
    encoding = load_harmony_encoding(encoding_name)
    sandbox = LocalJupyterSandbox(timeout=10.0)

    tool_cfg = tool_namespace_for_python(tool_prompt)
    messages = build_messages(system_prompt, user_query, tool_cfg)
    conv = Conversation.from_messages(messages)

    full_final_text: Optional[str] = None

    try:
        while True:
            prompt_ids = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
            max_tokens = max_model_len - len(prompt_ids)
            if max_tokens <= 0:
                raise RuntimeError("No room for generation: prompt already exceeds max_model_len.")

            # Stream completion
            stream = client.completions.create(
                model=served_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                prompt=prompt_ids,
                stream=True,
                extra_body={
                    "min_p": min_p,
                    "stop_token_ids": stop_token_ids,
                    "return_token_ids": True,
                },
            )

            token_buffer: list[int] = []
            try:
                for chunk in stream:
                    # Print streamed text live
                    text_piece = chunk.choices[0].text
                    if text_piece:
                        print(text_piece, end="", flush=True)

                    # Collect tokens for Harmony parsing
                    new_tokens = chunk.choices[0].token_ids
                    if new_tokens:
                        token_buffer.extend(new_tokens)

            finally:
                with contextlib.suppress(Exception):
                    stream.close()

            # newline after one streaming block
            print("", flush=True)

            if not token_buffer:
                raise RuntimeError("Empty token stream (no tokens returned).")

            new_msgs = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
            conv.messages.extend(new_msgs)
            last = new_msgs[-1]

            # If model finished
            if last.channel == "final":
                full_final_text = last.content[0].text
                break

            # If tool call
            if last.recipient == "python":
                code = last.content[0].text
                log("\n[python tool] executing:\n" + code + "\n")
                output = sandbox.execute(code)
                log("[python tool] output:\n" + output + "\n")
                conv.messages.append(make_tool_response(output, channel=last.channel))
                # Continue loop to let model use tool output
                continue

            # Otherwise keep going (e.g., analysis message)
            continue

    finally:
        sandbox.close()

    return full_final_text or ""


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True, help="User query text")
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--served-model-name", type=str, default="gpt-oss")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--base-url-host", type=str, default="127.0.0.1", help="client base_url host (usually 127.0.0.1)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.96)
    ap.add_argument("--dtype", type=str, default="auto")
    ap.add_argument("--kv-cache-dtype", type=str, default="fp8_e4m3")
    ap.add_argument("--max-model-len", type=int, default=64 * 1024)
    ap.add_argument("--stream-interval", type=int, default=20)
    ap.add_argument("--speculative-config", type=str, default="", help="JSON string for speculative config")
    ap.add_argument("--log-path", type=str, default="vllm_server_simple.log")

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.02)

    args = ap.parse_args()

    system_prompt = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'You must use print() to output results.'
    )

    # Start server
    spec_cfg = args.speculative_config.strip()
    if spec_cfg:
        # Ensure it's valid JSON (optional sanity check)
        try:
            json.loads(spec_cfg)
        except Exception as e:
            raise SystemExit(f"--speculative-config is not valid JSON: {e}")

    proc = start_vllm_server(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        tp=args.tp,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        stream_interval=args.stream_interval,
        speculative_config_json=spec_cfg if spec_cfg else None,
        log_path=args.log_path,
    )

    base_url = f"http://{args.base_url_host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="sk-local", timeout=600)

    try:
        wait_server_ready(client, proc, args.log_path, timeout_s=180)

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()

        log("=== Streaming answer (live) ===\n")
        final_text = run_single_query(
            client=client,
            encoding_name=HarmonyEncodingName.HARMONY_GPT_OSS,
            served_model_name=args.served_model_name,
            system_prompt=system_prompt,
            tool_prompt=tool_prompt,
            user_query=args.query,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_model_len=args.max_model_len,
        )

        log("\n=== FINAL (full text) ===")
        print(final_text, flush=True)

    finally:
        # Shutdown server
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            proc._logf.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()