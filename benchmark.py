# -*- coding: utf-8 -*-
"""
Single-file script (NO %%writefile):
- env setup
- optional pip uninstall
- optional cache_model() to warm OS page cache
- embedded local python tool (LocalJupyterSession + PythonTool)
- start vLLM OpenAI-compatible server
- HarmonyTIRInferencer: stream sampling + python tool execution + early stop by majority
- evaluation loop over reference.csv (Kaggle AIMO3 reference format)
All logs use print(..., flush=True).
"""

from __future__ import annotations

import atexit
import warnings
import contextlib
import math
import os
import queue
import re
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from openai import OpenAI
from transformers import AutoTokenizer, set_seed

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    SystemContent,
    ReasoningEffort,
    RenderConversationConfig,
)

# =========================
# 0) logging / env / time
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


GLOBAL_START_TIME = time.time()
START_TIME = time.time()
FINAL_CUTOFF_TIME = START_TIME + (4 * 60 + 57) * 60  # 4h55m


def setup_env() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    # os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", "/path/tiktoken_encodings")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

setup_env()
warnings.simplefilter("ignore")

import os
import sys
import subprocess

# +
import gc
import re
import math
import time
import queue
import threading
import contextlib
from typing import Optional
from jupyter_client import KernelManager
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

import pandas as pd
import polars as pl

from openai import OpenAI

from openai_harmony import (
    HarmonyEncodingName, 
    load_harmony_encoding, 
    SystemContent, 
    ReasoningEffort, 
    ToolNamespaceConfig, 
    Author, 
    Message, 
    Role, 
    TextContent, 
    Conversation
)

from transformers import set_seed


# -

class CFG:

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

    preference_prompt = (
        'Use `math`, `numpy`,`sympy`,`itertools` and `collections` to solve the problem.'
    )

    served_model_name = 'gpt-oss'
    model_path = 'openai/gpt-oss-120b'
    
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'

    high_problem_timeout = 900
    base_problem_timeout = 300

    notebook_limit = 17520
    server_timeout = 180

    session_timeout = 960
    jupyter_timeout = 10
    sandbox_timeout = 5

    stream_interval = 20
    context_tokens = 64 * 1024
    search_tokens = 1024
    buffer_tokens = 512
    batch_size = 8
    early_stop = 4
    attempts = 8
    workers = 8
    turns = 128
    seed = 42

    gpu_memory_utilization = 0.96
    temperature = 0.5
    top_p = 0.95
    min_p = 0.02


set_seed(CFG.seed)


class AIMO3Template:

    def __init__(self):

        pass

    def get_system_content(self, system_prompt: str, tool_config: ToolNamespaceConfig) -> SystemContent:

        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        tool_config: ToolNamespaceConfig
    ) -> list[Message]:

        system_content = self.get_system_content(system_prompt, tool_config)        
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        user_message = Message.from_role_and_content(Role.USER, user_prompt)

        return [system_message, user_message]


class AIMO3Sandbox:

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:

        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count

            return ports

    def __init__(self, timeout: float):

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None
        
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])

        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import numpy as np\n'
        )

    def _format_error(self, traceback: list[str]) -> str:

        clean_lines = []

        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)

            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue

            clean_lines.append(clean_frame)

        return ''.join(clean_lines)

    def execute(self, code: str, timeout: float | None = None) -> str:

        client = self._client
        effective_timeout = timeout or self._default_timeout
        
        msg_id = client.execute(
            code, 
            store_history=True, 
            allow_stdin=False, 
            stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > effective_timeout:
                self._km.interrupt_kernel()

                return f'[ERROR] Execution timed out after {effective_timeout} seconds'

            try:
                msg = client.get_iopub_msg(timeout=1.0)

            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')

                if content.get('name') == 'stdout':
                    stdout_parts.append(text)

                else:
                    stderr_parts.append(text)

            elif msg_type == 'error':
                traceback_list = content.get('traceback', [])

                stderr_parts.append(self._format_error(traceback_list))

            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')

                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')

            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)

        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr

        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):

        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()

        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):

        self.execute('%reset -f')
        self.execute('import gc; gc.collect()')

        self.execute(
            'import math\n'
            'import sympy\n'
            'import itertools\n'
            'import collections\n'
            'import numpy as np\n'
        )

    def __del__(self):

        self.close()


class AIMO3Tool:

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str, sandbox=None):

        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        
        self._owns_session = sandbox is None
        
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):

        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code: str) -> str:

        lines = code.strip().split('\n')

        if not lines:
            return code

        last_line = lines[-1].strip()

        if 'print' in last_line or 'import' in last_line:
            return code

        if not last_line:
            return code

        if last_line.startswith('#'):
            return code

        lines[-1] = 'print(' + last_line + ')'

        return '\n'.join(lines)

    @property
    def instruction(self) -> str:

        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:

        return ToolNamespaceConfig(
            name='python', 
            description=self.instruction, 
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:

        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')

        if channel:
            message = message.with_channel(channel)

        return message

    def process_sync_plus(self, message: Message) -> list[Message]:

        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)

        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)

            except TimeoutError as exc:
                output = f'[ERROR] {exc}'

        return [self._make_response(output, channel=message.channel)]

    def close(self):

        if self._jupyter_session is not None:
            if self._owns_session:
                self._jupyter_session.close()

            self._jupyter_session = None

    def __del__(self):

        self.close()


class AIMO3Solver:

    def __init__(self, cfg, port: int = 8001):

        self.cfg = cfg
        self.port = port
        self.base_url = f'http://127.0.0.1:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()
        
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key, 
            timeout=self.cfg.session_timeout
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50

    def _preload_model_weights(self) -> None:

        print(f'Loading model weights from {self.cfg.model_path} into OS Page Cache...')
        start_time = time.time()
        
        files_to_load = []
        total_size = 0

        for root, _, files in os.walk(self.cfg.model_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if os.path.isfile(file_path):
                    files_to_load.append(file_path)
                    total_size += os.path.getsize(file_path)

        def _read_file(path: str) -> None:

            with open(path, 'rb') as file_object:
                while file_object.read(1024 * 1024 * 1024):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            list(executor.map(_read_file, files_to_load))

        elapsed = time.time() - start_time
        print(f'Processed {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

    def _start_server(self) -> subprocess.Popen:

        cmd = [
            sys.executable, 
            '-m', 
            'vllm.entrypoints.openai.api_server', 
            '--seed', 
            str(self.cfg.seed), 
            '--model', 
            self.cfg.model_path, 
            '--served-model-name', 
            self.cfg.served_model_name, 
            '--tensor-parallel-size', 
            '1', 
            '--max-num-seqs', 
            str(self.cfg.batch_size), 
            '--gpu-memory-utilization', 
            str(self.cfg.gpu_memory_utilization), 
            '--host', 
            '0.0.0.0', 
            '--port', 
            str(self.port), 
            '--dtype', 
            self.cfg.dtype, 
            '--kv-cache-dtype', 
            self.cfg.kv_cache_dtype, 
            '--max-model-len', 
            str(self.cfg.context_tokens), 
            '--stream-interval', 
            str(self.cfg.stream_interval), 
            '--async-scheduling', 
            "--speculative-config",
            '{"method": "eagle3","model": "gpt-oss-120b-eagle3-aimo3","num_speculative_tokens": 3,"draft_tensor_parallel_size": 1}'
        ]

        self.log_file = open('vllm_server_test.log', 'w')

        return subprocess.Popen(
            cmd, 
            stdout=self.log_file, 
            stderr=subprocess.STDOUT, 
            start_new_session=True
        )

    def _wait_for_server(self):

        print('Waiting for vLLM server...')
        start_time = time.time()

        for _ in range(self.cfg.server_timeout):
            return_code = self.server_process.poll()

            if return_code is not None:
                self.log_file.flush()

                with open('vllm_server_test.log', 'r') as log_file:
                    logs = log_file.read()

                raise RuntimeError(f'Server died with code {return_code}. Full logs:\n{logs}\n')

            try:
                self.client.models.list()
                elapsed = time.time() - start_time
                print(f'Server is ready (took {elapsed:.2f} seconds).\n')

                return

            except Exception:
                time.sleep(1)

        raise RuntimeError('Server failed to start (timeout).\n')

    def _initialize_kernels(self) -> None:

        print(f'Initializing {self.cfg.workers} persistent Jupyter kernels...')
        start_time = time.time()

        self.sandbox_pool = queue.Queue()

        def _create_sandbox():
            
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = [executor.submit(_create_sandbox) for _ in range(self.cfg.workers)]

            for future in as_completed(futures):
                self.sandbox_pool.put(future.result())

        elapsed = time.time() - start_time
        print(f'Kernels initialized in {elapsed:.2f} seconds.\n')

    def _scan_for_answer(self, text: str) -> int | None:

        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)

        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)

                if 0 <= value <= 99999:
                    return value

            except ValueError:
                pass

        return None

    def _process_attempt(
        self, 
        problem: str, 
        system_prompt: str, 
        attempt_index: int, 
        stop_event: threading.Event, 
        deadline: float
    ) -> dict:

        if stop_event.is_set() or time.time() > deadline:
            return {
                'Attempt': attempt_index + 1, 
                'Answer': None, 
                'Python Calls': 0, 
                'Python Errors': 0, 
                'Response Length': 0
            }

        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None

        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)

            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout, 
                tool_prompt=self.cfg.tool_prompt, 
                sandbox=sandbox
            )

            encoding = self.encoding
            messages = self.template.apply_chat_template(
                system_prompt, 
                problem, 
                local_tool.tool_config
            )

            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)

                if max_tokens < self.cfg.buffer_tokens:
                    break

                # Create stream request with error handling
                stream = None
                try:
                    stream = self.client.completions.create(
                        model=self.cfg.served_model_name, 
                        temperature=self.cfg.temperature, 
                        top_p=self.cfg.top_p,
                        max_tokens=max_tokens, 
                        prompt=prompt_ids, 
                        seed=attempt_seed, 
                        stream=True, 
                        extra_body={
                            'min_p': self.cfg.min_p, 
                            'stop_token_ids': self.stop_token_ids, 
                            'return_token_ids': True
                        },
                        timeout=max(0,deadline - time.time()),
                    )
                except Exception as e:
                    print(f"⚠️ Failed to create completion stream: {e}")
                    # Break this iteration and try next one
                    break
    
                if stream is None:
                    # Stream creation failed, skip this iteration
                    continue

                try:
                    token_buffer = []
                    text_chunks = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break

                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text

                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)

                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)

                            if answer is not None:
                                final_answer = answer
                                break

                finally:
                    stream.close()

                if final_answer is not None:
                    break

                if not token_buffer:
                    break

                new_messages = encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]

                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    break

                if last_message.recipient == 'python':
                    python_calls += 1
                    print("🐍 Executing Python code...")
                    tool_responses = local_tool.process_sync_plus(last_message)

                    response_text = tool_responses[0].content[0].text

                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1

                    conversation.messages.extend(tool_responses)

        except Exception as exc:
            python_errors += 1

        finally:
            if local_tool is not None:
                local_tool.close()

            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            'Attempt': attempt_index + 1, 
            'Response Length': total_tokens, 
            'Python Calls': python_calls, 
            'Python Errors': python_errors, 
            'Answer': final_answer
        }

    def _select_answer(self, detailed_results: list) -> int:

        stats = defaultdict(lambda: {'votes': 0, 'calls': 0})

        for result in detailed_results:
            answer = result['Answer']

            if answer is not None:
                stats[answer]['votes'] += 1
                stats[answer]['calls'] += result['Python Calls']

        sorted_stats = sorted(
            stats.items(), 
            key=lambda item: (item[1]['votes'], item[1]['calls']), 
            reverse=True
        )

        vote_data = []

        for answer, data in sorted_stats:
            vote_data.append((answer, data['votes'], data['calls']))

        vote_dataframe = pd.DataFrame(vote_data, columns=['Answer', 'Votes', 'Calls'])
        print(vote_dataframe)

        final_answer = sorted_stats[0][0]
        final_votes = sorted_stats[0][1]['votes']
        final_calls = sorted_stats[0][1]['calls']

        print(f'\nFinal Result: {final_answer} | Votes: {final_votes} | Calls: {final_calls}\n')

        return final_answer

    def solve_problem(self, problem: str) -> int:
        
        problem_start_time = time.time()
        print(f'\nProblem: {problem}\n')

        user_input = f'{problem} {self.cfg.preference_prompt}'
        elapsed_global = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed_global
        problems_left_others = max(0, self.problems_remaining - 1)
        reserved_time = problems_left_others * self.cfg.base_problem_timeout

        budget = time_left - reserved_time
        budget = min(budget, self.cfg.high_problem_timeout)
        budget = max(budget, self.cfg.base_problem_timeout)

        deadline = time.time() + budget

        print(f'Budget: {budget:.2f} seconds | Deadline: {deadline:.2f}\n')

        tasks = []

        for attempt_index in range(self.cfg.attempts):
            tasks.append((self.cfg.system_prompt, attempt_index))

        detailed_results = []
        valid_answers = []

        stop_event = threading.Event()

        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)

        try:
            futures = []

            for (system_prompt, attempt_index) in tasks:
                future = executor.submit(
                    self._process_attempt, 
                    user_input, 
                    system_prompt, 
                    attempt_index, 
                    stop_event, 
                    deadline
                )

                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)

                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])

                    counts = Counter(valid_answers).most_common(1)

                    if counts and counts[0][1] >= self.cfg.early_stop:
                        stop_event.set()

                        for f in futures:
                            f.cancel()

                        break

                except Exception as exc:
                    print(f'Future failed: {exc}')
                    continue

        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        # print the inference time and budget
        used_time = time.time() - problem_start_time
        saved_time = max(0.0, budget - used_time)
        print(f"[Budget]: {budget:.2f}s\n")
        print(f"[inference] Took {used_time:.2f}s\n")
        print(f"[Saved time]: {saved_time:.2f}s\n")

        if detailed_results:
            results_dataframe = pd.DataFrame(detailed_results)
            results_dataframe['Answer'] = results_dataframe['Answer'].astype('Int64')
            print(results_dataframe)

        if not valid_answers:
            print('\nResult: 0\n')

            return 0

        return self._select_answer(detailed_results)

    def __del__(self):

        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()

        if hasattr(self, 'log_file'):
            self.log_file.close()

        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    sb = self.sandbox_pool.get_nowait()
                    sb.close()

                except Exception:
                    pass


solver = AIMO3Solver(CFG)


def predict(id_: pl.DataFrame, question: pl.DataFrame, answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    global correct_count, total_count, predictions
    
    question_id = id_.item(0)
    question_text = question.item(0)
    
    print("------")
    print(f"ID: {question_id}")
    print(f"Question: {question_text[:200]}...")
    
    final_answer = solver.solve_problem(question_text)
    predictions[question_id] = final_answer

    # Check accuracy if ground truth available
    total_count += 1
    if question_id in ground_truth:
        gt = ground_truth[question_id]
        is_correct = (final_answer == gt)
        if is_correct:
            correct_count += 1
        status = "✅" if is_correct else "❌"
        print(f"Answer: {final_answer} | Ground Truth: {gt} | {status}")
        print(f"📊 Running Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    else:
        print(f"Answer: {final_answer}")
    
    print("------\n")
    
    return pl.DataFrame({'id': question_id, 'answer': final_answer})


# +
# Load reference data and keep ground truth for accuracy calculation
# data
REF_CSV = "data/imo-benchmark.csv"
SAMPLE_N = 50  # for cutoff times only
df = pl.read_csv(REF_CSV).sample(n=50, seed=42)

# Store ground truth answers for accuracy calculation (only in local mode)
ground_truth = dict(zip(df["id"], df["answer"])) if "answer" in df.columns else {}

# Track predictions for accuracy calculation
predictions = {}
correct_count = 0
total_count = 0

for i in range(len(df)):
    print(f"{'=' * 20} idx:{i} {'=' * 20}", flush=True)
    r = predict(df[i]["id"], df[i]["problem"], df[i]["answer"])
    print(f"time: {int(time.time() - GLOBAL_START_TIME)}s", flush=True)

print("##### TOTAL #####", flush=True)
print(f"time: {int(time.time() - GLOBAL_START_TIME)}s", flush=True)
