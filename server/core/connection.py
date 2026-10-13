import os
import sys
import copy
import json
import re
import uuid
import time
import queue
import asyncio
import threading
import traceback
import subprocess
import websockets
from core.utils.exit_handler import is_exit_command, handle_exit

import re
from datetime import datetime

from core.utils.util import (
    extract_json_from_string,
    check_vad_update,
    check_asr_update,
    filter_sensitive_info,
)
from typing import Dict, Any, Optional
from collections import deque
from core.utils.modules_initialize import (
    initialize_modules,
    initialize_tts,
    initialize_asr,
)
from core.handle.reportHandle import report, enqueue_tool_report
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.handle.textHandle import handleTextMessage
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.loadplugins import auto_import_modules
from plugins_func.register import Action, ActionResponse, all_function_registry, module_func_map
from core.auth import AuthenticationError
from config.config_loader import get_private_config_from_api
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger
from config.manage_api_client import DeviceNotFoundException, DeviceBindException, generate_and_save_chat_title
from core.utils.prompt_manager import PromptManager
from core.utils.voiceprint_provider import VoiceprintProvider
from core.utils import textUtils
from core.utils.history_rag import history_rag
from plugins_func.functions.get_weather import get_weather


TAG = __name__

auto_import_modules("plugins_func.functions")


class TTSException(RuntimeError):
    pass

# direct_answer 虚拟工具定义
# 不是真实工具，是路由机制：将"调不调工具"的二选一变为"调哪个"的多选，防止小模型误触发真实工具
DIRECT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "direct_answer",
        "description": "当用户的请求不匹配其他任何工具时，可用此选项直接回复。将回复内容写在response参数里。",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "你回复用户的完整内容",
                },
            },
            "required": ["response"],
        },
    },
}


class ConnectionHandler:
    def __init__(
        self,
        config: Dict[str, Any],
        _vad,
        _asr,
        _tts,
        _llm,
        _memory,
        _intent,
        server=None,
    ):
        self.common_config = config
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()
        self.server = server  # Save server instance reference

        self.need_bind = False  # Whether device binding is needed
        self.bind_completed_event = asyncio.Event()
        self.bind_code = None  # Verification code for device binding
        self.last_bind_prompt_time = 0  # Timestamp of last binding prompt (seconds)
        self.bind_prompt_interval = 60  # Binding prompt interval (seconds)

        self.read_config_from_api = self.config.get("read_config_from_api", False)

        self.websocket: websockets.ServerConnection | None = None
        self.headers = None
        self.device_id = None
        self.client_ip = None
        self.prompt = None
        self.pipeline_prompts = {}
        self.welcome_msg = None
        self.max_output_size = 0
        self.chat_history_conf = 0
        self.audio_format = "opus"
        self.sample_rate = 24000  # 默认采样率，从客户端 hello 消息中动态更新

        # Client status related
        self.client_abort = False
        self.client_is_speaking = False
        self.client_aec = False  # 客户端 hello 里 features.aec=true 时开启（检测到人声即打断）
        self.introduced_speakers = set()  # 已在对话里称呼过的说话人（声纹识别用）
        self.client_listen_mode = "auto"

        # Thread task related
        self.loop = None  # Get running event loop in handle_connection
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Add report thread pool
        self.report_queue = queue.Queue()
        self.report_thread = None
        # In the future, this can be modified to adjust asr report and tts report, currently both are enabled by default
        self.report_asr_enable = self.read_config_from_api
        self.report_tts_enable = self.read_config_from_api

        # Dependent components
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr
        self._tts = _tts
        self._vad = _vad
        self.llm = _llm
        self.memory = _memory
        self.intent = _intent

        # Manage voiceprint recognition separately for each connection
        self.voiceprint_provider = None

        # VAD related variables
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_window = deque(maxlen=5)
        self.first_activity_time = 0.0  # Record first activity time (ms)
        self.last_activity_time = 0.0  # Unified activity timestamp (ms)
        self.client_voice_stop = False
        self.last_is_voice = False

        # ASR related variables
        # Since public local ASR might be used during deployment, variables cannot be exposed to public ASR
        # So variables related to ASR need to be defined here, belonging to connection's private variables
        self.asr_audio = []
        self.asr_audio_queue = queue.Queue()
        self.current_speaker = None  # Store current speaker
        self.current_language_tag = None  # Store current ASR recognized language tag

        # LLM related variables
        self.llm_finish_task = True
        self.dialogue = Dialogue()

        # TTS related variables
        self.sentence_id = None
        # Handle TTS response with no text returned
        self.tts_MessageText = ""

        self.func_handler = None

        self.cmd_exit = self.config["exit_commands"]

        # Whether to close connection after chat ends
        self.close_after_chat = False
        self.load_function_plugin = False
        self.intent_type = "nointent"

        self.timeout_seconds = (
            int(self.config.get("close_connection_no_voice_time", 120)) + 60
        )  # Add 60 seconds to original first timeout for second timeout check
        self.timeout_task = None

        # {"mcp":true} indicates MCP function enabled
        self.features = None

        # Initialize prompt manager
        self.prompt_manager = PromptManager(self.config, self.logger)

    async def handle_connection(self, ws: websockets.ServerConnection):
        try:
            # Get running event loop (must be in async context)
            self.loop = asyncio.get_running_loop()

            # Get and verify headers
            self.headers = dict(ws.request.headers)
            real_ip = self.headers.get("x-real-ip") or self.headers.get(
                "x-forwarded-for"
            )
            if real_ip:
                self.client_ip = real_ip.split(",")[0].strip()
            else:
                self.client_ip = ws.remote_address[0]
            self.logger.bind(tag=TAG).info(
                f"{self.client_ip} conn - Headers: {self.headers}"
            )

            self.device_id = self.headers.get("device-id", None)
            # Prioritize client-id for session identification, fallback to device-id
            self.client_id = self.headers.get("client-id", self.device_id)

            # Authentication passed, continue processing
            self.websocket = ws

            # Initialize activity timestamp
            self.first_activity_time = time.time() * 1000
            self.last_activity_time = time.time() * 1000

            # Start timeout check task
            self.timeout_task = asyncio.create_task(self._check_timeout())

            self.welcome_msg = self.config["xiaozhi"]
            self.welcome_msg["session_id"] = self.session_id

            # Initialize config and components in background (completely non-blocking main loop)
            asyncio.create_task(self._background_initialize())

            try:
                async for message in self.websocket:
                    await self._route_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.bind(tag=TAG).info("Client disconnected")

        except AuthenticationError as e:
            self.logger.bind(tag=TAG).error(f"Authentication failed: {str(e)}")
            return
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.bind(tag=TAG).error(f"Connection error: {str(e)}-{stack_trace}")
            return
        finally:
            try:
                await self._save_and_close(ws)
            except Exception as final_error:
                self.logger.bind(tag=TAG).error(f"Error during final cleanup: {final_error}")
                # Ensure connection is closed even if saving memory fails
                try:
                    await self.close(ws)
                except Exception as close_error:
                    self.logger.bind(tag=TAG).error(
                        f"Error during forced connection close: {close_error}"
                    )

    async def _save_and_close(self, ws):
        """Save memory and close connection"""
        try:
            # 守护线程1：独立生成标题（不依赖记忆模型）
            if self.session_id:
                def generate_title_task():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            generate_and_save_chat_title(self.session_id)
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"生成标题失败: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                threading.Thread(target=generate_title_task, daemon=True).start()

            # 守护线程2：走老流程记忆保存（仅记忆，不含标题）
            if self.memory:
                # Use thread pool to save memory asynchronously
                def save_memory_task():
                    try:
                        # Create new event loop (avoid conflict with main loop)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.memory.save_memory(
                                self.dialogue.dialogue, self.session_id
                            )
                        )
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"Failed to save memory: {e}")
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                # Start thread to save memory, do not wait for completion
                threading.Thread(target=save_memory_task, daemon=True).start()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to save memory: {e}")
        finally:
            # Close connection immediately, do not wait for memory save completion
            try:
                await self.close(ws)
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(
                    f"Failed to close connection after saving memory: {close_error}"
                )

    async def _discard_message_with_bind_prompt(self):
        """Discard message and check if binding prompt is needed"""
        current_time = time.time()
        # Check if binding prompt is needed
        if current_time - self.last_bind_prompt_time >= self.bind_prompt_interval:
            self.last_bind_prompt_time = current_time
            # Reuse existing binding prompt logic
            from core.handle.receiveAudioHandle import check_bind_device

            asyncio.create_task(check_bind_device(self))

    async def _route_message(self, message):
        """Message routing"""
        # 1. Bypass bind check for protocol handshake
        is_hello_msg = False
        if isinstance(message, str):
            try:
                msg_json = json.loads(message)
                if isinstance(msg_json, dict) and msg_json.get("type") == "hello":
                    is_hello_msg = True
            except:
                pass

        # Check if real binding status has been obtained
        if not is_hello_msg and not self.bind_completed_event.is_set():
            # Real status not obtained yet, wait until real status obtained or timeout
            try:
                await asyncio.wait_for(self.bind_completed_event.wait(), timeout=1)
            except asyncio.TimeoutError:
                # Timeout still not obtained real status, discard message
                await self._discard_message_with_bind_prompt()
                return

        # Real status obtained, check if binding is needed
        if not is_hello_msg and self.need_bind:
            # Binding needed, discard message
            await self._discard_message_with_bind_prompt()
            return

        # Binding not needed, continue processing message

        if isinstance(message, str):
            await handleTextMessage(self, message)
        elif isinstance(message, bytes):
            if self.vad is None or self.asr is None:
                return

            self.asr_audio_queue.put(message)

    async def handle_restart(self, message):
        """Handle server restart request"""
        try:

            self.logger.bind(tag=TAG).info("Received server restart command, preparing to execute...")

            # Send confirmation response
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "success",
                        "message": "Server restarting...",
                        "content": {"action": "restart"},
                    }
                )
            )

            # Async execute restart operation
            def restart_server():
                """Actual restart execution method"""
                time.sleep(1)
                self.logger.bind(tag=TAG).info("Executing server restart...")
                subprocess.Popen(
                    [sys.executable, "app.py"],
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    start_new_session=True,
                )
                os._exit(0)

            # Use thread to execute restart to avoid blocking event loop
            threading.Thread(target=restart_server, daemon=True).start()

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Restart failed: {str(e)}")
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "error",
                        "message": f"Restart failed: {str(e)}",
                        "content": {"action": "restart"},
                    }
                )
            )

    def _initialize_components(self):
        try:
            if self.tts is None:
                # Check if a private voice override is set for this client
                selected_module = self.config.get("selected_module", {}).get("TTS")
                has_private_voice = False
                if selected_module and selected_module in self.config.get("TTS", {}):
                    has_private_voice = "private_voice" in self.config["TTS"][selected_module]

                # Try to reuse global TTS instance ONLY if no private voice override is present
                if self._tts is not None and not has_private_voice:
                    self.tts = self._tts
                else:
                    self.logger.bind(tag=TAG).info(f"Initializing private TTS instance (has_private_voice={has_private_voice})")
                    self.tts = self._initialize_tts(use_cache=not has_private_voice)
            # Open audio synthesis channel
            asyncio.run_coroutine_threadsafe(
                self.tts.open_audio_channels(self), self.loop
            )
            if self.need_bind:
                self.bind_completed_event.set()
                return
            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)

            """Initialize components"""
            prompt = None
            self._load_pipeline_prompts()

            persona_prompt = self.pipeline_prompts.get("persona") if self.pipeline_prompts else None
            if persona_prompt:
                prompt = persona_prompt
                self.logger.bind(tag=TAG).info(
                    f"Layered persona prompt loaded successfully {prompt[:50]}..."
                )
            elif self.config.get("prompt") is not None:
                user_prompt = self.config["prompt"]
                # Legacy fallback: use quick prompt for initialization
                client_id = self.headers.get("client-id")
                prompt = self.prompt_manager.get_quick_prompt(user_prompt, self.device_id, client_id)
                self.change_system_prompt(prompt)
                self.logger.bind(tag=TAG).info(
                    f"Quick component initialization: prompt success {prompt[:50]}..."
                )

            """Initialize local components"""
            if self.vad is None:
                self.vad = self._vad
            if self.asr is None:
                self.asr = self._initialize_asr()

            # Initialize voiceprint recognition
            self._initialize_voiceprint()
            # Open audio recognition channel
            asyncio.run_coroutine_threadsafe(
                self.asr.open_audio_channels(self), self.loop
            )

            # Protocol Enhancement: Notify client that initialization is complete and listening has started
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({"type": "stt", "state": "listening", "session_id": self.session_id})), 
                self.loop
            )
            self.logger.bind(tag=TAG).info("ASR components ready, notified client: state=listening")

            """Load memory"""
            self._initialize_memory()
            """Load intent recognition"""
            self._initialize_intent()
            """Initialize report thread"""
            self._init_report_threads()
            """Update system prompt"""
            self._init_prompt_enhancement(prompt if 'prompt' in locals() else None)

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to instantiate components: {e}")

    def _init_prompt_enhancement(self, current_prompt=None):

        # Update context info
        self.prompt_manager.update_context_info(self, self.client_ip)

        # Prefer persona prompt from the layered pipeline if available.
        persona_prompt = None
        if self.pipeline_prompts:
            persona_prompt = self.pipeline_prompts.get("persona")

        # Fallback order:
        # 1. explicit current_prompt passed in
        # 2. layered persona prompt
        # 3. config default prompt
        base_prompt = current_prompt if current_prompt else (persona_prompt or self.config["prompt"])

        client_id = self.headers.get("client-id") if self.headers else None
        enhanced_prompt = self.prompt_manager.build_enhanced_prompt(
            base_prompt, self.device_id, self.client_ip, client_id=client_id
        )
        if enhanced_prompt:
            self.change_system_prompt(enhanced_prompt)
            self.logger.bind(tag=TAG).debug("System prompt enhanced successfully")


    def _inject_tool_call_fewshot(self):
        """注入工具调用 few-shot 示例到对话历史。
        结构：正样本（工具调用示例）放在动态 system 之前，可命中前缀缓存；
        负样本（直接回答示例）放在动态 system 之后、紧挨真实用户消息，
        确保模型在处理用户消息前最后看到的是"不调工具"的行为模式。
        """
        if self.intent_type != "function_call":
            return
        if not hasattr(self, "func_handler") or self.func_handler is None:
            return

        tools = self.func_handler.get_functions()
        if not tools:
            return

        tool_names = {t.get("function", {}).get("name") for t in tools}

        # === few-shot 示例（is_temporary）===
        # 展示 direct_answer 携带 response 参数的用法，一次调用完成回复

        # 示例1：direct_answer（回复内容写在 response 参数里，无需递归）
        da_tc_id = "fewshot_da_001"
        self.dialogue.put(Message(role="user", content="给我讲个故事吧", is_temporary=True))
        self.dialogue.put(Message(
            role="assistant",
            tool_calls=[{
                "id": da_tc_id,
                "function": {"arguments": '{"response": "好呀，你想听什么类型的呀？童话、冒险还是搞笑的？选一个我给你开讲~"}', "name": "direct_answer"},
                "type": "function", "index": 0,
            }],
            is_temporary=True,
        ))
        self.dialogue.put(Message(
            role="tool", tool_call_id=da_tc_id,
            content="已直接回复", is_temporary=True,
        ))

        # 示例2：真实工具调用（handle_exit_intent）
        if "handle_exit_intent" in tool_names:
            tc_id = "fewshot_exit_001"
            self.dialogue.put(Message(role="user", content="拜拜", is_temporary=True))
            self.dialogue.put(Message(
                role="assistant",
                tool_calls=[{
                    "id": tc_id,
                    "function": {"arguments": '{"say_goodbye": "再见，下次再聊~"}', "name": "handle_exit_intent"},
                    "type": "function", "index": 0,
                }],
                is_temporary=True,
            ))
            self.dialogue.put(Message(
                role="tool", tool_call_id=tc_id,
                content="退出意图已处理", is_temporary=True,
            ))
            self.dialogue.put(Message(
                role="assistant", content="再见，下次再聊~", is_temporary=True,
            ))

        self.logger.bind(tag=TAG).debug("已注入工具调用 few-shot 示例")

    def _init_report_threads(self):
        """Initialize ASR and TTS report threads"""
        if not self.read_config_from_api or self.need_bind:
            return
        if self.chat_history_conf == 0:
            return
        if self.report_thread is None or not self.report_thread.is_alive():
            self.report_thread = threading.Thread(
                target=self._report_worker, daemon=True
            )
            self.report_thread.start()
            self.logger.bind(tag=TAG).info("TTS report thread started")

    def _initialize_tts(self, use_cache=True):
        """Initialize TTS"""
        tts = None
        if not self.need_bind:
            tts = initialize_tts(self.config, use_cache=use_cache)

        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)

        return tts

    def _initialize_asr(self):
        """Initialize ASR"""
        # Check if we can reuse the global ASR instance
        # Conditions:
        # 1. Global instance exists
        # 2. Global instance is Local type (reusable)
        # 3. Client config matches global config (no override)
        current_asr = self.config["selected_module"]["ASR"]
        global_asr = self.common_config["selected_module"]["ASR"]
        
        if (
            self._asr is not None
            and hasattr(self._asr, "interface_type")
            and self._asr.interface_type == InterfaceType.LOCAL
            and current_asr == global_asr
        ):
            # If public ASR is a local service AND matches client config, reuse it
            asr = self._asr
        else:
            # If public ASR is remote, OR client requested a different module, initialize new instance
            asr = initialize_asr(self.config)

        return asr

    def _initialize_voiceprint(self):
        """Initialize voiceprint recognition for current connection"""
        try:
            voiceprint_config = self.config.get("voiceprint", {})
            if voiceprint_config:
                voiceprint_provider = VoiceprintProvider(voiceprint_config)
                if voiceprint_provider is not None and voiceprint_provider.enabled:
                    self.voiceprint_provider = voiceprint_provider
                    self.logger.bind(tag=TAG).info("Voiceprint recognition enabled dynamically on connection")
                else:
                    self.logger.bind(tag=TAG).warning("Voiceprint recognition enabled but configuration incomplete")
            else:
                self.logger.bind(tag=TAG).info("Voiceprint recognition not enabled")
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"Voiceprint recognition initialization failed: {str(e)}")

    def _load_local_client_config(self):
        """Load local client config to override default settings"""
        if not self.headers:
            return

        client_id = self.headers.get("client-id", self.device_id)
        if not client_id:
            return

        try:
            config_path = os.path.join("data", client_id, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    client_config = json.load(f)
                    
                # Override TTS voice if present
                if "voice" in client_config:
                    voice = client_config["voice"]
                    selected_module = self.config.get("selected_module", {}).get("TTS")
                    if selected_module and selected_module in self.config.get("TTS", {}):
                        # Most TTS providers use 'private_voice' as an override
                        self.config["TTS"][selected_module]["private_voice"] = voice
                        self.logger.bind(tag=TAG).info(f"Overriding TTS voice for client {client_id}: {voice}")

                # Override ASR module if present
                if "asr_module" in client_config:
                    asr_module = client_config["asr_module"]
                    if asr_module in self.config.get("ASR", {}):
                        self.config["selected_module"]["ASR"] = asr_module
                        self.logger.bind(tag=TAG).info(f"Overriding ASR module for client {client_id}: {asr_module}")
                    else:
                         self.logger.bind(tag=TAG).warning(f"Client {client_id} requested invalid ASR module: {asr_module}")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to load local client config: {e}")

    async def _background_initialize(self):
        """Initialize config and components in background (completely non-blocking main loop)"""
        try:
            # Load local client config first to ensure it can override other settings
            self._load_local_client_config()

            # Async get diff config
            await self._initialize_private_config_async()
            # Initialize components in thread pool
            self.executor.submit(self._initialize_components)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Background initialization failed: {e}")

    async def _initialize_private_config_async(self):
        """Fetch private config from API async (async version, non-blocking main loop)"""
        if not self.read_config_from_api:
            self.need_bind = False
            self.bind_completed_event.set()
            return
        try:
            begin_time = time.time()
            private_config = await get_private_config_from_api(
                self.config,
                self.headers.get("device-id"),
                self.headers.get("client-id", self.headers.get("device-id")),
            )
            private_config["delete_audio"] = bool(self.config.get("delete_audio", True))
            private_config["tts_timeout"] = self.config.get("tts_timeout", 15)
            self.logger.bind(tag=TAG).info(
                f"{time.time() - begin_time} seconds, async fetch private config success: {json.dumps(filter_sensitive_info(private_config), ensure_ascii=False)}"
            )
            self.need_bind = False
            self.bind_completed_event.set()
        except DeviceNotFoundException as e:
            self.need_bind = True
            private_config = {}
        except DeviceBindException as e:
            self.need_bind = True
            self.bind_code = e.bind_code
            private_config = {}
        except Exception as e:
            self.need_bind = True
            self.logger.bind(tag=TAG).error(f"Async fetch private config failed: {e}")
            private_config = {}

        init_llm, init_tts, init_memory, init_intent = (
            False,
            False,
            False,
            False,
        )

        init_vad = check_vad_update(self.common_config, private_config)
        init_asr = check_asr_update(self.common_config, private_config)

        if init_vad:
            self.config["VAD"] = private_config["VAD"]
            self.config["selected_module"]["VAD"] = private_config["selected_module"][
                "VAD"
            ]
        if init_asr:
            self.config["ASR"] = private_config["ASR"]
            self.config["selected_module"]["ASR"] = private_config["selected_module"][
                "ASR"
            ]
        if private_config.get("TTS", None) is not None:
            init_tts = True
            self.config["TTS"] = private_config["TTS"]
            self.config["selected_module"]["TTS"] = private_config["selected_module"][
                "TTS"
            ]
        if private_config.get("LLM", None) is not None:
            init_llm = True
            self.config["LLM"] = private_config["LLM"]
            self.config["selected_module"]["LLM"] = private_config["selected_module"][
                "LLM"
            ]
        if private_config.get("VLLM", None) is not None:
            self.config["VLLM"] = private_config["VLLM"]
            self.config["selected_module"]["VLLM"] = private_config["selected_module"][
                "VLLM"
            ]
        if private_config.get("Memory", None) is not None:
            init_memory = True
            self.config["Memory"] = private_config["Memory"]
            self.config["selected_module"]["Memory"] = private_config[
                "selected_module"
            ]["Memory"]
        if private_config.get("Intent", None) is not None:
            init_intent = True
            self.config["Intent"] = private_config["Intent"]
            model_intent = private_config.get("selected_module", {}).get("Intent", {})
            self.config["selected_module"]["Intent"] = model_intent
            # Load plugin config
            if model_intent != "Intent_nointent":
                plugin_from_server = private_config.get("plugins", {})
                for plugin, config_str in plugin_from_server.items():
                    plugin_from_server[plugin] = json.loads(config_str)
                # 将模块级别的插件配置复制到各具体函数名，
                # 方便后续 per-function 查找描述、news_sources 等配置
                for module_name, func_names in module_func_map.items():
                    if module_name in plugin_from_server:
                        module_config = plugin_from_server[module_name]
                        for func_name in func_names:
                            if func_name not in plugin_from_server:
                                plugin_from_server[func_name] = module_config
                self.config["plugins"] = plugin_from_server
                # 将模块级别的插件名展开为具体函数名
                expanded_functions = []
                for plugin_key in plugin_from_server.keys():
                    if plugin_key in all_function_registry:
                        expanded_functions.append(plugin_key)
                    elif plugin_key in module_func_map:
                        expanded_functions.extend(module_func_map[plugin_key])
                    else:
                        expanded_functions.append(plugin_key)
                self.config["Intent"][self.config["selected_module"]["Intent"]][
                    "functions"
                ] = expanded_functions
        if private_config.get("prompt", None) is not None:
            self.config["prompt"] = private_config["prompt"]
        # Get voiceprint info
        if private_config.get("voiceprint", None) is not None:
            self.config["voiceprint"] = private_config["voiceprint"]
        if private_config.get("summaryMemory", None) is not None:
            self.config["summaryMemory"] = private_config["summaryMemory"]
        if private_config.get("device_max_output_size", None) is not None:
            self.max_output_size = int(private_config["device_max_output_size"])
        if private_config.get("chat_history_conf", None) is not None:
            self.chat_history_conf = int(private_config["chat_history_conf"])
        if private_config.get("mcp_endpoint", None) is not None:
            self.config["mcp_endpoint"] = private_config["mcp_endpoint"]
        if private_config.get("context_providers", None) is not None:
            self.config["context_providers"] = private_config["context_providers"]

        # Use run_in_executor to execute initialize_modules in thread pool, avoid blocking main loop
        try:
            modules = await self.loop.run_in_executor(
                None,  # Use default thread pool
                initialize_modules,
                self.logger,
                private_config,
                init_vad,
                init_asr,
                init_llm,
                init_tts,
                init_memory,
                init_intent,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to initialize components: {e}")
            modules = {}
        if modules.get("tts", None) is not None:
            self.tts = modules["tts"]
        if modules.get("vad", None) is not None:
            self.vad = modules["vad"]
        if modules.get("asr", None) is not None:
            self.asr = modules["asr"]
        if modules.get("llm", None) is not None:
            self.llm = modules["llm"]
        if modules.get("Intent", None) is not None:
            self.intent = modules["Intent"]
        if modules.get("memory", None) is not None:
            self.memory = modules["memory"]

    def _initialize_memory(self):
        if self.memory is None:
            return
        """Initialize memory module"""
        self.memory.init_memory(
            role_id=self.device_id,
            llm=self.llm,
            summary_memory=self.config.get("summaryMemory", None),
            save_to_file=not self.read_config_from_api,
        )

        # Get memory summary config
        memory_config = self.config["Memory"]
        memory_type = self.config["Memory"][self.config["selected_module"]["Memory"]][
            "type"
        ]
        # If nomen is used, return directly
        if memory_type == "nomem":
            return
        # Use mem_local_short mode
        elif memory_type == "mem_local_short":
            memory_llm_name = memory_config[self.config["selected_module"]["Memory"]][
                "llm"
            ]
            if memory_llm_name and memory_llm_name in self.config["LLM"]:
                # If dedicated LLM configured, create independent LLM instance
                from core.utils import llm as llm_utils

                memory_llm_config = self.config["LLM"][memory_llm_name]
                memory_llm_type = memory_llm_config.get("type", memory_llm_name)
                memory_llm = llm_utils.create_instance(
                    memory_llm_type, memory_llm_config
                )
                self.logger.bind(tag=TAG).info(
                    f"Created dedicated LLM for memory summary: {memory_llm_name}, type: {memory_llm_type}"
                )
                self.memory.set_llm(memory_llm)
            else:
                # Otherwise use main LLM
                self.memory.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("Using main LLM as intent recognition model")

    def _initialize_intent(self):
        """Standardized to Uppercase 'Intent' to match config.yaml"""
        if self.intent is None:
            return

        # Use Uppercase 'Intent' everywhere
        selected_intent = self.config["selected_module"].get("Intent")
        if not selected_intent:
            return

        self.intent_type = self.config["Intent"][selected_intent]["type"]
        
        if self.intent_type == "function_call" or self.intent_type == "intent_llm":
            self.load_function_plugin = True

        intent_config = self.config["Intent"]
        
        if self.intent_type == "nointent":
            return
        elif self.intent_type == "intent_llm":
            intent_llm_name = intent_config[selected_intent].get("llm")
            
            if intent_llm_name and intent_llm_name in self.config["LLM"]:
                from core.utils import llm as llm_utils
                intent_llm_config = self.config["LLM"][intent_llm_name]
                intent_llm_type = intent_llm_config.get("type", intent_llm_name)
                intent_llm = llm_utils.create_instance(intent_llm_type, intent_llm_config)
                
                self.intent.set_llm(intent_llm)
                self.logger.bind(tag=TAG).info(f"Intent Brain Linked: {intent_llm_name}")
            else:
                self.intent.set_llm(self.llm)
                self.logger.bind(tag=TAG).info("Using Main LLM for Intent.")

        """Load unified tool handler"""
        self.func_handler = UnifiedToolHandler(self)


    def change_system_prompt(self, prompt):
        self.prompt = prompt
        # Update system prompt to context
        self.dialogue.update_system_message(self.prompt)

    def _load_pipeline_prompts(self):
        """Load persona / decision / interpretation prompts for the current client."""
        try:
            client_id = self.headers.get("client-id") if self.headers else None
            self.pipeline_prompts = self.prompt_manager.get_pipeline_prompts(
                client_id=client_id,
                device_id=self.device_id,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to load pipeline prompts: {e}")
            self.pipeline_prompts = {}

    def _llm_single_shot(self, system_prompt: str, user_payload: Any) -> str:
        """Run a one-shot LLM call without mutating the main dialogue history."""
        if not self.llm:
            return ""

        try:
            if isinstance(user_payload, str):
                user_content = user_payload
            else:
                user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)

            temp_dialogue = Dialogue()
            temp_dialogue.update_system_message(system_prompt)
            temp_dialogue.put(Message(role="user", content=user_content))

            memory_str = None
            messages = temp_dialogue.get_llm_dialogue_with_memory(
                memory_str, self.config.get("voiceprint", {})
            )

            chunks = []
            temp_session_id = str(uuid.uuid4())
            for chunk in self.llm.response(temp_session_id, messages):
                if chunk:
                    chunks.append(chunk)
            return "".join(chunks).strip()
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Single-shot LLM call failed: {e}")
            return ""

    def _safe_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Parse JSON response safely, with fallback extraction from free text."""
        if not raw_text:
            return {}

        try:
            return json.loads(raw_text)
        except Exception:
            pass

        extracted = extract_json_from_string(raw_text)
        if extracted:
            try:
                return json.loads(extracted)
            except Exception:
                pass

        self.logger.bind(tag=TAG).warning(f"Failed to parse JSON response: {raw_text[:200]}")
        return {}
    def _filter_stale_realtime_summary_lines(self, summary_text: str) -> str:
        if not summary_text:
            return ""

        blocked_keywords = (
            "current cgm",
            "critical cgm",
            "last bolus",
            "temp basal",
            "latest pump event",
            "fresh pump status",
            "pump sync issue",
            "tir:",
            "time in range",
            "mean glucose",
            "gmi:",
            "mg/dl",
            "u/hr",
            "auto-bolus",
            "bolus recorded",
            "boluses recorded",
            "pump_events.csv",
            "cgm.csv",
        )

        kept_lines = []
        for line in summary_text.splitlines():
            lowered = line.lower()
            if any(keyword in lowered for keyword in blocked_keywords):
                continue
            kept_lines.append(line)

        return "\n".join(kept_lines).strip()

    def _build_pipeline_analysis_result(self, query: str, context_needs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build structured analysis input for the decision / interpretation pipeline."""
        result = {
            "user_query": query,
            "language_hint": getattr(self, "current_language_tag", None),
            "intent_context": context_needs or {},
            "cgm_status": "no_data",  # ok / stale / no_data
            "glucose_now": None,  # 只有 cgm_status == "ok" 时才有值
            "glucose_direction": None,
            "glucose_age_min": None,
            "last_glucose": None,  # stale 时的最后一次读数，不能当作当前值
            "cgm_note": None,
            "cgm_metrics": {},
            "events": {},
            "patterns": {
                "daily": [],
                "weekly": [],
                "monthly": [],
            },
            "pump_data_status": {"status": "no_data"},  # ok / stale / no_data，见 PumpManager.get_data_status
            "pump_status": {},
            "pump_patterns": [],
            "pump_glucose_joint": [],
            "anomalies": [],
        }

        client_id = self.headers.get("client-id") if self.headers else None
        if not client_id:
            return result

        try:
            config_path = os.path.join("data", client_id, "config.json")
            client_cfg = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    client_cfg = json.load(f)

            cgm_enabled = bool(client_cfg.get("cgm"))
            pump_enabled = bool(client_cfg.get("pump"))

            from core.utils.cgm_manager import CGMManager, format_cgm_status, NO_DATA_NOTE
            result["cgm_note"] = NO_DATA_NOTE

            if cgm_enabled:
                cgm_manager = CGMManager()

                cgm_state = cgm_manager.get_current_status(
                    client_id, int(self.config.get("cgm_stale_after_minutes", 15))
                )
                result["cgm_status"] = cgm_state["status"]
                result["glucose_direction"] = cgm_state["direction"]
                result["glucose_age_min"] = cgm_state["age_min"]
                result["cgm_note"] = format_cgm_status(cgm_state)
                if cgm_state["status"] == "ok":
                    result["glucose_now"] = cgm_state["glucose"]
                elif cgm_state["status"] == "stale":
                    result["last_glucose"] = cgm_state["glucose"]

                # 这里尽量兼容你现有 manager，如果某个方法不存在就跳过
                if hasattr(cgm_manager, "get_metrics_summary"):
                    metrics = cgm_manager.get_metrics_summary(client_id)
                    if isinstance(metrics, dict):
                        result["cgm_metrics"] = metrics.get("metrics", metrics)
                        if "events" in metrics and isinstance(metrics["events"], dict):
                            result["events"] = metrics["events"]

                context_summary = None
                if hasattr(cgm_manager, "get_context_summary"):
                    context_summary = cgm_manager.get_context_summary(client_id)
                if context_summary:
                    result["patterns"]["daily"].append(context_summary)

            if pump_enabled:
                from core.utils.pump_manager import PumpManager
                pump_manager = PumpManager()

                pump_context = pump_manager.get_authoritative_summary(client_id)
                if pump_context:
                    result["pump_status"]["summary"] = pump_context

                result["pump_data_status"] = pump_manager.get_data_status(
                    client_id, float(self.config.get("pump_stale_after_hours", 6))
                )
                result["pump_patterns"] = pump_manager.analyze_recent_patterns(client_id)
                result["pump_glucose_joint"] = pump_manager.analyze_insulin_effectiveness(client_id)
                result["anomalies"] = pump_manager.detect_pump_anomalies(client_id)

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to build pipeline analysis result: {e}")

        return result

    def run_coach_pipeline(self, query: str, context_needs: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Run decision -> interpretation -> persona pipeline for diabetes coaching."""
        prompts = self.pipeline_prompts or {}
        persona_prompt = self.prompt or prompts.get("persona", "")
        decision_prompt = prompts.get("decision", "")
        interpretation_prompt = prompts.get("interpretation", "")

        if not persona_prompt or not decision_prompt or not interpretation_prompt:
            return None

        analysis_result = self._build_pipeline_analysis_result(query, context_needs=context_needs)

        decision_raw = self._llm_single_shot(decision_prompt, analysis_result)
        decision_result = self._safe_parse_json(decision_raw)

        interpretation_payload = {
            "analysis_result": analysis_result,
            "decision_result": decision_result,
        }
        insight_text = self._llm_single_shot(interpretation_prompt, interpretation_payload)
        if not insight_text:
            return None

        persona_payload = {
            "user_message": query,
            "insight_text": insight_text,
            "decision_result": decision_result,
            "analysis_result": analysis_result,
        }
        final_reply = self._llm_single_shot(persona_prompt, persona_payload)
        return final_reply or insight_text
    
    def chat(self, query, depth=0, audio_data=None):
        search_text = query or ""
        if query is not None:
            self.logger.bind(tag=TAG).info(f"LLM received user message: {query}")

            # Parse Query (could be JSON or Text)
            is_json = False
            json_data = {}

            if depth == 0:
                try:
                    if query.strip().startswith("{") and query.strip().endswith("}"):
                        json_data = json.loads(query)
                        if "content" in json_data:
                            search_text = json_data["content"]
                            is_json = True
                except json.JSONDecodeError:
                    pass

                # 1. Intent Classification (Unified call for shortcuts, CGM context, pump context, and search injection)
                intent_result = {
                    "needs_cgm": False,
                    "needs_pump": False,
                    "needs_search": False,
                    "fast_answer": None,
                    "reply": None,
                }
                try:
                    client_id = self.headers.get("client-id") if self.headers else None
                    if client_id:
                        from core.utils.cgm_intent import classify_context_needs
                        intent_result = classify_context_needs(
                            search_text, 
                            client_id, 
                            self.config, 
                            language_hint=getattr(self, "current_language_tag", None)
                        )
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"Intent classification failed: {e}")

                # 2. Fast Intent Check (time, weather, location, exit - skip main LLM)
                try:
                    client_id = self.headers.get("client-id") if self.headers else None
                    if client_id:
                        from core.utils.cgm_intent import get_fast_answer
                        fast_answer_type = intent_result.get("fast_answer")
                        if fast_answer_type:
                            # a. Special Case: Exit
                            if fast_answer_type == "exit":
                                self.logger.bind(tag=TAG).info(f"Exit intent recognized by LLM: {search_text}")
                                handle_exit(self)
                                return
                            
                            # b. Get Content to Speak
                            fast_response = None
                            
                            # Give priority to programmatic answers for data-driven intents
                            if fast_answer_type in ["time", "weather", "location"]:
                                fast_response = get_fast_answer(fast_answer_type, client_id, self.config)
                            
                            # If no programmatic answer or it's a pure conversation intent, use LLM reply
                            if not fast_response:
                                fast_response = intent_result.get("reply")
                            elif intent_result.get("reply"):
                                # If we have both, we can optionally prefix (choosing to prefer the accurate data for now)
                                self.logger.bind(tag=TAG).debug(f"Combined response: LLM says '{intent_result.get('reply')}', using data: '{fast_response}'")

                            if fast_response:
                                self.logger.bind(tag=TAG).info(f"Fast response ({fast_answer_type}): {fast_response}")
                                self.sentence_id = str(uuid.uuid4().hex)
                                # Wake up TTS with FIRST message
                                self.tts.tts_text_queue.put(
                                    TTSMessageDTO(
                                        sentence_id=self.sentence_id,
                                        sentence_type=SentenceType.FIRST,
                                        content_type=ContentType.TEXT,
                                        content_detail="",
                                    )
                                )
                                self.tts.tts_text_queue.put(
                                    TTSMessageDTO(
                                        sentence_id=self.sentence_id,
                                        sentence_type=SentenceType.LAST,
                                        content_type=ContentType.TEXT,
                                        content_detail=fast_response,
                                    )
                                )
                                self.dialogue.put(Message(role="user", content=search_text))
                                self.dialogue.put(Message(role="assistant", content=fast_response))
                                return
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"Fast answer check failed: {e}")

                # Deterministic pump-bolus shortcut.
                # ASR sometimes hears "bolus" as "bullet/bullets"; do not let this fall back to history/RAG.
                # This answer must come directly from pump_events.csv via PumpManager.
                try:
                    lower_search_text = (search_text or "").lower()
                    pump_bolus_keywords = [
                        "bolus",
                        "bolo",
                        "bullet",
                        "bullets",
                        "pump bolus",
                        "last pump bolus",
                        "last recorded bolus",
                    ]
                    asks_last = any(k in lower_search_text for k in ["last", "latest", "recent", "when", "上次", "最近", "上一"])
                    asks_pump_bolus = any(k in lower_search_text for k in pump_bolus_keywords)

                    if asks_pump_bolus and asks_last:
                        client_id = self.client_id if getattr(self, "client_id", None) else (self.headers.get("client-id") if self.headers else None)
                        if client_id:
                            from core.utils.pump_manager import PumpManager
                            pm = PumpManager()
                            self.logger.bind(tag=TAG).info(f"Direct pump bolus lookup for {client_id}: {search_text}")
                            pm.fetch_and_update(client_id)
                            direct_response = pm.format_latest_bolus_summary(client_id)
                            if direct_response:
                                direct_response = direct_response.replace("Last Bolus", "Last recorded bolus")
                                direct_response += " This is the last recorded bolus only, not the last insulin delivery."

                                self.sentence_id = str(uuid.uuid4().hex)
                                self.tts.tts_text_queue.put(
                                    TTSMessageDTO(
                                        sentence_id=self.sentence_id,
                                        sentence_type=SentenceType.FIRST,
                                        content_type=ContentType.TEXT,
                                        content_detail="",
                                    )
                                )
                                self.tts.tts_text_queue.put(
                                    TTSMessageDTO(
                                        sentence_id=self.sentence_id,
                                        sentence_type=SentenceType.LAST,
                                        content_type=ContentType.TEXT,
                                        content_detail=direct_response,
                                    )
                                )
                                self.dialogue.put(Message(role="user", content=search_text))
                                self.dialogue.put(Message(role="assistant", content=direct_response))
                                return None
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"Direct pump bolus lookup failed: {e}")

            try:
                # Experimental: RAG Injection (using actual content)
                if depth == 0 and self.headers:
                    client_id = self.headers.get("client-id")
                    if client_id:
                        context_list = []
                        
                        # History RAG
                        hist_context = history_rag.search(search_text, client_id)
                        if hist_context:
                            self.logger.bind(tag=TAG).info(f"Injecting history RAG context for {client_id}")
                            context_list.append(hist_context)

                        # CGM RAG from Config
                        cgm_status_injected = False
                        rag_config_path = os.path.join("data", client_id, "config.json")
                        if os.path.exists(rag_config_path):
                            with open(rag_config_path, "r") as f:
                                c = json.load(f)

                                # Use captured intent result (cached from start of chat)
                                context_needs = intent_result

                                # CGM - Only for clients with "cgm" config object
                                # Config format: {"cgm": {"api_secret": "...", "user_tz": "..."}}
                                cgm_config = c.get("cgm")
                                if cgm_config:
                                    # Always include the current reading (like time/date); the prompt rules decide when to MENTION it.
                                    # Refresh from Nightscout first if the local copy is older than 5 min.
                                    from core.utils.cgm_manager import CGMManager, format_cgm_status
                                    cgm_manager = CGMManager()
                                    stale_after = int(self.config.get("cgm_stale_after_minutes", 15))

                                    cgm_state = cgm_manager.get_current_status(client_id, stale_after)
                                    if cgm_state["status"] == "no_data" or cgm_state["age_min"] > 5:
                                        self.logger.bind(tag=TAG).debug(f"CGM data missing or older than 5 min ({cgm_state['age_min']}), refreshing...")
                                        cgm_manager.fetch_and_update(client_id)
                                        cgm_state = cgm_manager.get_current_status(client_id, stale_after)

                                    if cgm_state["status"] == "ok" and (cgm_state["glucose"] < 70 or cgm_state["glucose"] > 250):
                                        self.logger.bind(tag=TAG).warning(f"CGM critical: {cgm_state['glucose']} for {client_id}")
                                    elif cgm_state["status"] != "ok":
                                        self.logger.bind(tag=TAG).info(f"CGM {cgm_state['status']} for {client_id} (age_min={cgm_state['age_min']})")
                                    context_list.append(format_cgm_status(cgm_state))
                                    cgm_status_injected = True

                                    # Full CGM context (trends, patterns) only if user asks
                                    if context_needs.get("needs_cgm"):
                                        cgm_summary = cgm_manager.get_context_summary(client_id)
                                        if cgm_summary:
                                            self.logger.bind(tag=TAG).info(f"Injecting full CGM context for {client_id}")
                                            context_list.append(cgm_summary)
                                
                                # Pump/CGM Sync moved to before coach pipeline above
                                pass

                                if context_needs.get("needs_cgm") and context_needs.get("needs_pump"):
                                    self.logger.bind(tag=TAG).info(
                                        f"Joint diabetes reasoning requested for {client_id}: injecting both CGM and pump context"
                                    )
                                        
                                # Web Search - One-shot factual grounding
                                if context_needs.get("needs_search") and context_needs.get("search_query"):
                                    search_query = context_needs.get("search_query")
                                    self.logger.bind(tag=TAG).info(f"Triggering web search: {search_query}")
                                    from core.utils.web_search import perform_web_search
                                    search_results = perform_web_search(search_query, self.config)
                                    if search_results:
                                        # Truncate and wrap as a primary source
                                        if len(search_results) > 2000:
                                            search_results = search_results[:2000] + "... [truncated]"
                                        
                                        context_list.append(
                                            f"--- WEB SEARCH DATA: {search_query} ---\n"
                                            f"{search_results}\n"
                                            f"--- END DATA ---\n"
                                            f"IMPORTANT: Use the data above to answer the user now. Do not mention searching again."
                                        )

                                if not any([
                                    context_needs.get("needs_cgm"),
                                    context_needs.get("needs_pump"),
                                    context_needs.get("needs_search"),
                                ]):
                                    self.logger.bind(tag=TAG).debug(f"Context classifier: no extra injection needed")

                        # 用户问到血糖、但这个用户没有配置 CGM：明确告诉 LLM 暂无数据，避免它编一个读数
                        if not cgm_status_injected and intent_result.get("needs_cgm"):
                            from core.utils.cgm_manager import NO_DATA_NOTE
                            self.logger.bind(tag=TAG).info(f"CGM requested but not configured for {client_id}: injecting no-data note")
                            context_list.append(NO_DATA_NOTE)

                        # Inject Context into Query
                        if context_list:
                            full_context = "\n".join(context_list)
                            if is_json:
                                json_data["content"] = f"{search_text}\n{full_context}"
                                query = json.dumps(json_data, ensure_ascii=False)
                            else:
                                query = f"{search_text}\n{full_context}"

            except Exception as e:
                self.logger.bind(tag=TAG).error(f"RAG Injection failed: {e}")

        # Create session ID and send FIRST request when at top level
        if depth == 0:
            self.llm_finish_task = False
            self.sentence_id = str(uuid.uuid4().hex)
            current_sentence_id = self.sentence_id
            self.suppress_tts = False


            self.dialogue.put(Message(role="user", content=query))
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
        else:
            # 递归调用时，使用当前的sentence_id
            current_sentence_id = self.sentence_id

        # Set max recursion depth to avoid infinite loop, adjust as needed
        MAX_DEPTH = 5
        force_final_answer = False  # Mark whether to force final answer

        if depth >= MAX_DEPTH:
            self.logger.bind(tag=TAG).debug(
                f"Max tool call depth {MAX_DEPTH} reached, forcing answer based on existing information"
            )
            force_final_answer = True
            # Add system instruction, requesting LLM to answer based on existing information
            self.dialogue.put(
                Message(
                    role="user",
                    content="[System Prompt] Max tool call limit reached, please provide a final answer based on all information obtained so far. Do not attempt to call any more tools.",
                )
            )

        # Define intent functions
        functions = None
        # When max depth reached, disable tool calls, force LLM to answer directly
        if (
                self.intent_type == "function_call"
                and hasattr(self, "func_handler")
                and not force_final_answer
        ):
            functions = list(self.func_handler.get_functions())
            # 仅在第一层调用时注入 direct_answer 虚拟工具
            # 递归调用（depth>0）不注入，避免模型在生成文本回复时再次调 direct_answer 导致循环
            if functions is not None and depth == 0:
                functions.append(DIRECT_ANSWER_TOOL)

        response_message = []

        try:
            # Layered diabetes-coach pipeline.
            # Only run at top level for diabetes-related queries when layered prompts are available.
            should_run_coach_pipeline = False
            context_for_pipeline = intent_result if 'intent_result' in locals() else {}
            if depth == 0 and self.pipeline_prompts and isinstance(context_for_pipeline, dict):
                should_run_coach_pipeline = any([
                    context_for_pipeline.get("needs_cgm"),
                    context_for_pipeline.get("needs_pump"),
                ])

            # ON-DEMAND SYNC BEFORE COACH PIPELINE OR REGULAR CHAT
            pump_summary = None
            if isinstance(context_for_pipeline, dict):
                client_id = self.client_id
                if context_for_pipeline.get("needs_pump"):
                    from core.utils.pump_manager import PumpManager
                    pm = PumpManager()
                    self.logger.bind(tag=TAG).info(f"Syncing pump data for {client_id} before response...")
                    pm.fetch_and_update(client_id)
                    pump_summary = pm.get_authoritative_summary(client_id)

                if context_for_pipeline.get("needs_cgm"):
                    from core.utils.cgm_manager import CGMManager
                    cm = CGMManager()
                    self.logger.bind(tag=TAG).info(f"Syncing CGM data for {client_id} before response...")
                    cm.fetch_and_update(client_id)
                    # CGM usually doesn't need an authoritative summary block in memory_str as it's less prone to "memory" confusion,
                    # but we sync it here anyway for the coach pipeline to be fresh.

            if should_run_coach_pipeline:
                try:
                    coach_reply = self.run_coach_pipeline(
                        search_text,
                        context_needs=context_for_pipeline,
                    )
                    if coach_reply:
                        self.logger.bind(tag=TAG).info(f"Coach pipeline reply: {coach_reply}")

                        self.dialogue.put(Message(role="assistant", content=coach_reply))

                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=self.sentence_id,
                                sentence_type=SentenceType.LAST,
                                content_type=ContentType.TEXT,
                                content_detail=coach_reply,
                            )
                        )
                        self.llm_finish_task = True
                        return None
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"Coach pipeline failed, falling back to legacy chat path: {e}")

            # Use dialogue with memory
            # Use dialogue with memory
            memory_str = None

            if self.memory is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self.memory.query_memory(query), self.loop
                )
                memory_str = future.result()
                
            # CRITICAL: Overwrite memory with authoritative pump data if available
            if pump_summary:
                authoritative_block = (
                    "[CRITICAL: THE DATA BELOW IS REAL-TIME. DISREGARD ANY CONTRADICTING PUMP INFO IN THE MEMORY TAG BELOW.]\n"
                    f"{pump_summary}\n"
                    "[END REAL-TIME DATA]"
                )
                if memory_str:
                    memory_str = f"{authoritative_block}\n\n{memory_str}"
                else:
                    memory_str = authoritative_block

            else:
                # Fallback: Try to load local summary.txt
                try:
                     c_id = self.headers.get("client-id") if self.headers else getattr(self, "device_id", None)
                     if c_id:
                         sum_path = os.path.join("data", c_id, "summary.txt")
                         if os.path.exists(sum_path):
                             with open(sum_path, "r", encoding="utf-8") as f:
                                 memory_str = f.read().strip()
                                 memory_str = self._filter_stale_realtime_summary_lines(memory_str)
                except Exception as e:
                     self.logger.bind(tag=TAG).warning(f"Failed to load local summary: {e}")


            # 仅在该说话人首次出现时把身份注入 system，之后靠对话历史首轮保留，
            # 避免每轮在 system 重复出现名字诱导模型反复称呼
            speaker_for_system = None
            cs = (self.current_speaker or "").strip()
            if cs and cs != "未知说话人" and cs not in self.system_introduced_speakers:
                self.system_introduced_speakers.add(cs)
                speaker_for_system = cs

            if self.intent_type == "function_call" and functions is not None:
                # Use streaming interface with function support
                llm_responses = self.llm.response_with_functions(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {}), speaker_for_system
                    ),
                    functions=functions,
                    audio_data=audio_data
                )
            else:
                llm_responses = self.llm.response(
                    self.session_id,
                    self.dialogue.get_llm_dialogue_with_memory(
                        memory_str, self.config.get("voiceprint", {}), speaker_for_system
                    ),
                    audio_data=audio_data
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"LLM processing error {query}: {e}")
            self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail="I'm sorry, I'm not thinking clearly right now.")
            return None

        # Handle streaming response
        tool_call_flag = False
        # Support multiple parallel tool calls - store in list
        tool_calls_list = []  # Format: [{"id": "", "name": "", "arguments": ""}]
        content_arguments = ""
        emotion_flag = True
        
        # Buffer for intent trigger detection
        stream_buffer = ""
        trigger_executed = False
        in_think_block = False # State for filtering <think> tags

        # Streaming watchdog (60s limit for total chat response)
        chat_start_time = time.time()
        CHAT_TIMEOUT = 60

        for response in llm_responses:
            # Check for total duration timeout
            if time.time() - chat_start_time > CHAT_TIMEOUT:
                self.logger.bind(tag=TAG).warning(f"Main chat stream exceeded {CHAT_TIMEOUT}s limit, aborting.")
                break

            if self.client_abort:
                break
            if self.intent_type == "function_call" and functions is not None:
                content, tools_call = response
                if "content" in response:
                    content = response["content"]
                    tools_call = None
                if content is not None and len(content) > 0:
                    content_arguments += content

                if not tool_call_flag and content_arguments.startswith("<tool_call>"):
                    tool_call_flag = True

                if tools_call is not None and len(tools_call) > 0:
                    tool_call_flag = True
                    self._merge_tool_calls(tool_calls_list, tools_call)
            else:
                content = response

            # --- Thinking Filter Logic ---
            if content:
                # Handle start of thinking block
                if "<think>" in content:
                    self.logger.bind(tag=TAG).info(f"Detected <think> start in stream...")
                    pre, post = content.split("<think>", 1)
                    # Process 'pre' (keep it)
                    # Discard 'post' (start of think) unless closed immediately
                    content = pre 
                    in_think_block = True
                    # Check if closed in same chunk (edge case)
                    if "</think>" in post:
                         _, remainder = post.split("</think>", 1)
                         content += remainder
                         in_think_block = False
                         self.logger.bind(tag=TAG).info(f"Thinking block finished (same chunk).")

                # Handle end of thinking block (if already in block)
                elif in_think_block:
                    if "</think>" in content:
                        _, post = content.split("</think>", 1)
                        content = post
                        in_think_block = False
                        self.logger.bind(tag=TAG).info(f"Thinking block finished.")
                    else:
                        content = "" # Consume entire chunk as thinking
                        # Optional: self.logger.debug("Filtering thought chunk...")
            # --- End Thinking Filter Logic ---

            # Get emotion in LLM response, only once at the beginning of a turn
            if emotion_flag and content is not None and content.strip():
                asyncio.run_coroutine_threadsafe(
                    textUtils.get_emotion(self, content),
                    self.loop,
                )
                emotion_flag = False


            # STREAM TRIGGER HANDLING with BUFFERING
            if content is not None and len(content) > 0 and not tool_call_flag:
                # If we already executed a trigger, we suppress all subsequent output (assuming LLM complies with "ONLY output trigger")
                if trigger_executed:
                    continue
                
                stream_buffer += content
                
                # Check if buffer *starts* with potential trigger marker "@@"
                # We strip leading whitespace from buffer to be safe
                buffer_stripped = stream_buffer.lstrip()
                
                if buffer_stripped.startswith("@@"):
                    # We are potentially in a trigger. Hold output.
                    # Check if we have a full trigger
                    if "@@" in buffer_stripped[2:]: # Look for closing @@
                        # Full trigger found. Process it.
                        closing_index = buffer_stripped.find("@@", 2)
                        trigger_content = buffer_stripped[:closing_index+2]
                        
                        # @@WEATHER@@
                        if "@@WEATHER@@" in trigger_content:
                            self.logger.bind(tag=TAG).info("Detected Stream Trigger: @@WEATHER@@")
                            trigger_executed = True
                            try:
                                w_res = get_weather(self, location=None)
                                w_text = w_res.response if w_res.response else w_res.result
                                self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=w_text)
                                response_message.append(w_text)
                            except Exception as e:
                                self.logger.bind(tag=TAG).error(f"Trigger execution failed: {e}")
                        
                        # @@EXIT@@
                        elif "@@EXIT@@" in trigger_content:
                            self.logger.bind(tag=TAG).info("Detected Stream Trigger: @@EXIT@@")
                            trigger_executed = True
                            self.close_after_chat = True
                            if not response_message:
                                self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail="Goodbye!")
                                response_message.append("Goodbye!")

                        # [TOOL_CALL] - Web Search
                        elif "[TOOL_CALL]" in trigger_content:
                            self.logger.bind(tag=TAG).info("Detected Stream Trigger: [TOOL_CALL]")
                            trigger_executed = True
                            try:
                                # Extract JSON payload
                                # Format: [TOOL_CALL] { "tool": "google_search", "args": { "query": "..." } }
                                match = re.search(r"\[TOOL_CALL\]\s*(\{.*?\})", trigger_content)
                                if match:
                                    json_str = match.group(1)
                                    tool_data = json.loads(json_str)
                                    tool_name = tool_data.get("tool")
                                    args = tool_data.get("args", {})
                                    
                                    if tool_name == "google_search":
                                        query = args.get("query")
                                        self.logger.bind(tag=TAG).info(f"Executing Web Search Tool: {query}")
                                        
                                        # Speak loading message
                                        self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=f"Let me search specifically for {query}...")
                                        
                                        # Execute Search
                                        from core.utils.web_search import perform_web_search
                                        results = perform_web_search(query, self.config)
                                        
                                        if not results:
                                            results = "No results found."
                                        
                                        # Truncate results
                                        if len(results) > 2000:
                                            results = results[:2000] + "... [truncated]"
                                            
                                        # RECURSIVE CALL: Feed results back to LLM
                                        # We append the results as a "Function" or "System" message to history
                                        # and call chat() again to get the final answer.
                                        
                                        search_context = (
                                            f"--- WEB SEARCH RESULTS for '{query}' ---\n"
                                            f"{results}\n"
                                            f"--- END RESULTS ---\n"
                                            f"Instructions: Use the search results above to answer the user's question directly. "
                                            f"Do not mention the tool call or the search process again. Just give the answer."
                                        )
                                        
                                        # Add to history
                                        self.dialogue.put(Message(role="system", content=search_context))
                                        
                                        # Recurse!
                                        # Important: Increment depth to prevent infinite loops logic (though we handle it)
                                        self.chat(query="[System: results injected, please answer]", depth=depth+1)
                                        return # End this level of recursion, the child call handles the rest
                                        
                            except Exception as e:
                                self.logger.bind(tag=TAG).error(f"Tool execution failed: {e}")
                                self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail="I tried to search but something went wrong.")

                        else:
                            # Unknown tag
                            self.logger.bind(tag=TAG).warning(f"Unknown trigger detected and ignored: {trigger_content}")
                            self.tts.tts_text_queue.put(
                                TTSMessageDTO(
                                    sentence_id=self.sentence_id,
                                    sentence_type=SentenceType.MIDDLE,
                                    content_type=ContentType.TEXT,
                                    content_detail=stream_buffer,
                                )
                            )
                            response_message.append(stream_buffer)
                            stream_buffer = ""
                        
                        if trigger_executed:
                            stream_buffer = "" 
                        else:
                            stream_buffer = "" 
                            
                    else:
                        # Starts with @@ but no closing @@ yet. Wait.
                        pass
                
                elif "@" in buffer_stripped and len(buffer_stripped) < 5:
                    # Could be start of @@. Hold briefly.
                    # e.g. "@" -> hold. "@@" -> move to startswith block.
                    # "Email@" -> "@" in... but "Email@" doesn't start with @
                    # We check `buffer_stripped` length. If it's short and contains @, we hold just in case.
                    # If user says "My email is foo@bar", buffer "My email is foo@bar" > 5. 
                    # But at the start: "My" -> flush.
                    # So this logic only holds if valid Text starts with @.
                    pass
                
                else:
                    # --- Garbage Filter for TTS ---
                    # If we detect tool call markup or start of JSON, we stop TTS for this response
                    # Also blocking <invoke> web search tags
                    block_tags = ["[TOOL_CALL]", "<tool_call>", "{ \"tool\"", "```json", "<invoke>", "<web_search>", "<query>"]
                    if self.suppress_tts:
                        stream_buffer = ""
                        continue

                    # --- Safety Buffer for Fragments ---
                    # Check if buffer ends with a potential start of a tag or markdown
                    # Capture unclosed tags: [ start, then non-] chars until end
                    # Capture unclosed markdown: ** start, then non-* chars until end
                    import re
                    partial_match = re.search(r'([<\[\{][^>\]\}]*|(?:\*\*|__)[^*_]*)$', stream_buffer)
                    
                    safe_content = stream_buffer
                    kept_suffix = ""
                    
                    if partial_match:
                        # If we have a potential start, hold it back
                        kept_suffix = partial_match.group(1)
                        safe_content = stream_buffer[:-len(kept_suffix)]
                        # If we end up holding everything, update buffer and continue
                        if not safe_content:
                            stream_buffer = kept_suffix
                            continue

                    # Check for complete tags inside the safe content
                    block_tags = ["[TOOL_CALL]", "<tool_call>", "{ \"tool\"", "```json", "<invoke>", "<web_search>", "<query>"]
                    if any(tag in safe_content for tag in block_tags):
                        self.logger.bind(tag=TAG).info(f"Purging tool-call garbage: {safe_content[:50]}...")
                        self.suppress_tts = True
                        stream_buffer = "" 
                        continue

                    # --- Markdown Cleaner ---
                    # 1. Links [text](url) -> text
                    safe_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', safe_content)
                    # 2. Bold/Italic (** or __) -> text
                    safe_content = re.sub(r'(\*\*|__)(.*?)\1', r'\2', safe_content)
                    # 3. Headers (## ) -> empty (or text)
                    safe_content = re.sub(r'^#+\s+', '', safe_content)
                    
                    if safe_content:
                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=self.sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.TEXT,
                                content_detail=safe_content,
                            )
                        )
                        response_message.append(safe_content)
                    
                    # Keep the dangerous suffix for the next chunk
                    stream_buffer = kept_suffix


            # End of loop body

        # Handle function call
        if tool_call_flag:
            bHasError = False
            # Handle text-based tool call format
            if len(tool_calls_list) == 0 and content_arguments:
                a = extract_json_from_string(content_arguments)
                if a is not None:
                    try:
                        content_arguments_json = json.loads(a)
                        tool_calls_list.append(
                            {
                                "id": str(uuid.uuid4().hex),
                                "name": content_arguments_json["name"],
                                "arguments": json.dumps(
                                    content_arguments_json["arguments"],
                                    ensure_ascii=False,
                                ),
                            }
                        )
                    except Exception as e:
                        bHasError = True
                        response_message.append(a)
                else:
                    bHasError = True
                    response_message.append(content_arguments)
                if bHasError:
                    self.logger.bind(tag=TAG).error(
                        f"function call error: {content_arguments}"
                    )

            if not bHasError and len(tool_calls_list) > 0:
                # If LLM needs to process a round first, add related processing logs
                # 本轮已流式播报的文本，供 _handle_function_result 去重
                streamed_text = "".join(response_message)
                if len(response_message) > 0:
                    text_buff = "".join(response_message)
                    self.tts_MessageText = text_buff
                    self.dialogue.put(Message(role="assistant", content=text_buff))
                response_message.clear()

                # 分离 direct_answer 虚拟工具和真实工具调用
                direct_answer_calls = [
                    tc for tc in tool_calls_list if tc.get("name") == "direct_answer"
                ]
                real_tool_calls = [
                    tc for tc in tool_calls_list if tc.get("name") != "direct_answer"
                ]

                if direct_answer_calls:
                    self.logger.bind(tag=TAG).debug(
                        f"模型选择 direct_answer，流式已播报，写入对话历史"
                    )
                    for tc in direct_answer_calls:
                        da_response = self._extract_direct_answer_response(tc.get("arguments", "{}"))
                        if da_response:
                            # 刷新流式缓冲区中未发送的部分
                            sent_len = tc.get("_da_sent", 0)
                            remaining = da_response[sent_len:]
                            if remaining:
                                remaining = self._clean_response_garbage(remaining)
                                if remaining:
                                    self.tts.tts_text_queue.put(
                                        TTSMessageDTO(
                                            sentence_id=current_sentence_id,
                                            sentence_type=SentenceType.MIDDLE,
                                            content_type=ContentType.TEXT,
                                            content_detail=remaining,
                                        )
                                    )
                            # 写入对话历史
                            da_response = self._clean_response_garbage(da_response)
                            self.tts.store_tts_text(current_sentence_id, da_response)
                            self.dialogue.put(Message(role="assistant", content=da_response))

                    if not real_tool_calls:
                        if depth == 0:
                            self.tts.tts_text_queue.put(
                                TTSMessageDTO(
                                    sentence_id=current_sentence_id,
                                    sentence_type=SentenceType.LAST,
                                    content_type=ContentType.ACTION,
                                )
                            )
                        return

                    tool_calls_list = real_tool_calls

            if not bHasError and len(tool_calls_list) > 0:
                self.logger.bind(tag=TAG).debug(
                    f"Detected {len(tool_calls_list)} tool calls"
                )

                # Collect all tool call Futures
                futures_with_data = []
                for tool_call_data in tool_calls_list:
                    self.logger.bind(tag=TAG).debug(
                        f"function_name={tool_call_data['name']}, function_id={tool_call_data['id']}, function_arguments={tool_call_data['arguments']}"
                    )

                    # 使用公共方法上报工具调用
                    tool_input = json.loads(tool_call_data.get("arguments") or "{}")
                    enqueue_tool_report(self, tool_call_data['name'], tool_input)

                    future = asyncio.run_coroutine_threadsafe(
                        self.func_handler.handle_llm_function_call(
                            self, tool_call_data
                        ),
                        self.loop,
                    )
                    futures_with_data.append((future, tool_call_data, tool_input))

                # Wait for coroutines to finish (actual wait time is the slowest one)
                tool_results = []
                for future, tool_call_data, tool_input in futures_with_data:
                    result = future.result()
                    tool_results.append((result, tool_call_data))

                # Unified handling of all tool call results
                if tool_results:
                    self._handle_function_result(tool_results, depth=depth, streamed_text=streamed_text)

        # Store dialogue content
        if len(response_message) > 0:
            text_buff = "".join(response_message)
            self.tts.store_tts_text(current_sentence_id, text_buff)
            self.dialogue.put(Message(role="assistant", content=text_buff))

            # Send full LLM text to client for on-screen display.
            # This arrives while TTS audio is still being synthesized/played,
            # allowing the device to show the complete response text immediately.
            if depth == 0:
                try:
                    display_text = textUtils.check_emoji(text_buff)
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(json.dumps({
                            "type": "tts",
                            "state": "sentence_start",
                            "text": display_text,
                            "session_id": self.session_id,
                        })),
                        self.loop,
                    )
                except Exception as e:
                    self.logger.bind(tag=TAG).warning(f"Failed to send LLM text to client: {e}")

        if depth == 0:
            self.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=current_sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
            self.llm_finish_task = True
            # Use lambda for lazy evaluation, execute get_llm_dialogue() only at DEBUG level
            self.logger.bind(tag=TAG).debug(
                lambda: json.dumps(
                    self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False
                )
            )

        return True

    def _handle_function_result(self, tool_results, depth, streamed_text=""):
        need_llm_tools = []
        record_tools = []

        for result, tool_call_data in tool_results:
            if result.action in [
                Action.RESPONSE,
                Action.NOTFOUND,
                Action.ERROR,
            ]:  # Direct reply to frontend
                text = result.response if result.response else result.result
                if streamed_text and text in streamed_text:
                    self.logger.bind(tag=TAG).debug(
                        f"Skipping duplicate TTS for tool {tool_call_data['name']}, already streamed"
                    )
                else:
                    self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=text)
                    self.tts.store_tts_text(self.sentence_id, text)
                self.dialogue.put(Message(role="assistant", content=text))
            elif result.action == Action.REQLLM:
                # Collect tools needing LLM processing
                need_llm_tools.append((result, tool_call_data))
            elif result.action == Action.RECORD:
                record_tools.append((result, tool_call_data))
            else:
                pass

        # Action.RECORD：写入完整工具调用链（assistant(tool_calls) → tool(result) → assistant(response)）
        # 模型从历史中学到工具调用模式，不额外调用LLM
        if record_tools:
            # 构造 assistant 消息（含 tool_calls），记录"模型调用了哪些工具"
            all_tool_calls = [
                {
                    "id": tool_call_data["id"],
                    "function": {
                        "arguments": (
                            "{}"
                            if tool_call_data["arguments"] == ""
                            else tool_call_data["arguments"]
                        ),
                        "name": tool_call_data["name"],
                    },
                    "type": "function",
                    "index": idx,
                }
                for idx, (_, tool_call_data) in enumerate(record_tools)
            ]
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            # 写入每条工具的执行结果，记录"工具返回了什么"
            for result, tool_call_data in record_tools:
                text = result.result or ""
                self.dialogue.put(
                    Message(
                        role="tool",
                        tool_call_id=(
                            str(uuid.uuid4())
                            if tool_call_data["id"] is None
                            else tool_call_data["id"]
                        ),
                        content=text,
                    )
                )

            # 用固定文本作为最终回复，补全标准三段式，保证下一条消息是 user 而非接 tool
            response_parts = []
            for result, _ in record_tools:
                resp = result.response or result.result
                if resp:
                    response_parts.append(resp)
            if response_parts:
                self.dialogue.put(Message(role="assistant", content="，".join(response_parts)))

        if need_llm_tools:
            all_tool_calls = [
                {
                    "id": tool_call_data["id"],
                    "function": {
                        "arguments": (
                            "{}"
                            if tool_call_data["arguments"] == ""
                            else tool_call_data["arguments"]
                        ),
                        "name": tool_call_data["name"],
                    },
                    "type": "function",
                    "index": idx,
                }
                for idx, (_, tool_call_data) in enumerate(need_llm_tools)
            ]
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tool_call_data in need_llm_tools:
                text = result.result
                if text is not None and len(text) > 0:
                    self.dialogue.put(
                        Message(
                            role="tool",
                            tool_call_id=(
                                str(uuid.uuid4())
                                if tool_call_data["id"] is None
                                else tool_call_data["id"]
                            ),
                            content=text,
                        )
                    )

            self.chat(None, depth=depth + 1)

    def _report_worker(self):
        """Chat history report worker thread"""
        while not self.stop_event.is_set():
            try:
                # Get data from queue, set timeout to check stop event regularly
                item = self.report_queue.get(timeout=1)
                if item is None:  # Check for poison pill
                    break
                try:
                    # Check thread pool status
                    if self.executor is None:
                        continue
                    # Submit task to thread pool
                    self.executor.submit(self._process_report, *item)
                except Exception as e:
                    self.logger.bind(tag=TAG).error(f"Chat history report thread exception: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"Chat history report worker thread exception: {e}")

        self.logger.bind(tag=TAG).info("Chat history report thread exited")

    def _process_report(self, type, text, audio_data, report_time):
        """Handle report task"""
        try:
            # Execute async report (run in event loop)
            asyncio.run(report(self, type, text, audio_data, report_time))
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Report handling exception: {e}")
        finally:
            # Mark task as done
            self.report_queue.task_done()

    def clearSpeakStatus(self):
        self.client_is_speaking = False
        self.logger.bind(tag=TAG).debug(f"Clear server speaking status")

    def save_session_memory(self):
        """Save current session dialogue to local JSON memory"""
        try:
            # Determine Client ID
            client_id = None
            if self.headers:
                 client_id = self.headers.get("client-id")
            if not client_id and hasattr(self, "device_id"):
                 client_id = self.device_id
            
            if not client_id:
                return

            memory_dir = os.path.join("data", client_id)
            if not os.path.exists(memory_dir):
                os.makedirs(memory_dir, exist_ok=True)
                
            memory_file = os.path.join(memory_dir, "memory.json")
            
            # Serialize dialogue
            messages = []
            if hasattr(self.dialogue, "dialogue"):
                for msg in self.dialogue.dialogue:
                    # Basic serialization
                    m_dict = {
                        "role": msg.role,
                        "content": msg.content
                    }
                    # Handle tool calls if present (simple serialization to avoid JSON errors)
                    if msg.tool_calls:
                         try:
                             # Attempt to convert to dict/list if possible, else str
                             if hasattr(msg.tool_calls, "to_dict"):
                                 m_dict["tool_calls"] = msg.tool_calls.to_dict()
                             elif isinstance(msg.tool_calls, (list, dict)):
                                 m_dict["tool_calls"] = msg.tool_calls
                             else:
                                 m_dict["tool_calls"] = str(msg.tool_calls)
                         except:
                             m_dict["tool_calls"] = str(msg.tool_calls)
                             
                    if msg.tool_call_id:
                        m_dict["tool_call_id"] = msg.tool_call_id
                        
                    messages.append(m_dict)
            
            if not messages:
                return

            session_data = {
                "session_id": getattr(self, "session_id", str(uuid.uuid4().hex)),
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }

            # Load existing history
            history = []
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            history = []
                except Exception as e:
                    self.logger.bind(tag=TAG).warning(f"Failed to load existing memory json: {e}")
                    history = []
            
            # Append new session
            history.append(session_data)
            
            # Save back
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
                
            self.logger.bind(tag=TAG).info(f"Session memory saved to {memory_file}")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to save session memory: {e}")

    def summarize_session(self):
        """Summarize current session and update local summary file"""
        try:
            # 1. Check prerequisites
            if not getattr(self, "client_id", None):
                 if self.headers:
                     self.client_id = self.headers.get("client-id")
                 if not getattr(self, "client_id", None) and hasattr(self, "device_id"):
                     self.client_id = self.device_id
            
            if not self.client_id:
                return

            if not hasattr(self, "llm") or not self.llm:
                return

            # Avoid summarizing short conversations
            if not hasattr(self.dialogue, "dialogue") or len(self.dialogue.dialogue) < 4:
                return

            memory_dir = os.path.join("data", self.client_id)
            if not os.path.exists(memory_dir):
                os.makedirs(memory_dir, exist_ok=True)
            summary_file = os.path.join(memory_dir, "summary.txt")

            # 2. Get existing summary
            existing_summary = ""
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        existing_summary = f.read().strip()
                        existing_summary = self._filter_stale_realtime_summary_lines(existing_summary)
                except Exception as e:
                    self.logger.bind(tag=TAG).warning(f"Failed to read summary file: {e}")

            # 3. Format current dialogue
            new_dialogue_text = ""
            for msg in self.dialogue.dialogue:
                if msg.role in ["user", "assistant"] and msg.content:
                     new_dialogue_text += f"{msg.role}: {msg.content}\n"
            
            if not new_dialogue_text:
                return

            # 4. Prompt LLM
            prompt = (
                "You are a memory manager maintaining long-term memory about a user.\n"
                "Only store durable, stable information such as user preferences, habits, identity, and long-term context.\n"
                "Do NOT store real-time or frequently changing data.\n"
                "Specifically, NEVER store medical/device state such as: CGM values, glucose trends, Time in Range (TIR), GMI, mean glucose, pump events, boluses, basal rates, or sync-status statements.\n"
                "All real-time health/device data must come from live system context (CSV/API), NOT memory.\n"
                "Keep the summary concise and factual. Do not include timestamps, temporary observations, or conversational filler.\n\n"
            )
            
            if existing_summary:
                prompt += f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            
            prompt += (
                f"NEW CONVERSATION:\n{new_dialogue_text}\n\n"
                "TASK: Update the summary using ONLY durable information from this conversation. "
                "Do NOT include any real-time glucose, insulin, CGM, or pump-related values or interpretations. "
                "If the new conversation only contains temporary or real-time data, do NOT modify the summary. "
                "Return the existing summary unchanged in that case."
            )

            self.logger.bind(tag=TAG).info("Generating session summary...")
            
            # Use response_no_stream for single-turn generation
            # Ensure timeout is handled by LLM provider config
            # Use response_no_stream for single-turn generation
            # Ensure timeout is handled by LLM provider config
            new_summary = self.llm.response_no_stream(
                system_prompt="You are a memory manager.", 
                user_prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )

            # 5. Save updated summary
            if new_summary and "[LLM Service Response Error]" not in new_summary and len(new_summary.strip()) > 0:
                 with open(summary_file, "w", encoding="utf-8") as f:
                     f.write(new_summary.strip())
                 self.logger.bind(tag=TAG).info(f"Updated summary saved to {summary_file}")
            else:
                 self.logger.bind(tag=TAG).warning(f"Summary generation failed or empty. Result: {new_summary}")

        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Failed to summarize session: {e}")

    async def close(self, ws=None):
        """Resource cleanup method"""
        try:
            # Save session memory before closing
            self.save_session_memory()
            
            # Generate summary before closing (if not terminating due to error)
            # Run in executor to avoid blocking main loop validation if possible, 
            # but close() is async, so we'll just run it. 
            # Note: response_no_stream might block if not truly async, but we are closing anyway.
            # Ideally wrap in run_in_executor
            try:
                await self.loop.run_in_executor(None, self.summarize_session)
            except Exception as e:
                 self.logger.bind(tag=TAG).error(f"Async summary generation failed: {e}")

            # Clear audio buffer
            if hasattr(self, "audio_buffer"):
                self.audio_buffer.clear()

            # Cancel timeout task
            if self.timeout_task and not self.timeout_task.done():
                self.timeout_task.cancel()
                try:
                    await self.timeout_task
                except asyncio.CancelledError:
                    pass
                self.timeout_task = None

            # Clear tool handler resources
            if hasattr(self, "func_handler") and self.func_handler:
                try:
                    await self.func_handler.cleanup()
                except Exception as cleanup_error:
                    self.logger.bind(tag=TAG).error(
                        f"Error cleaning up tool handler: {cleanup_error}"
                    )

            # Trigger stop event
            if self.stop_event:
                self.stop_event.set()

            # Clear task queues
            self.clear_queues()

            # Close WebSocket connection
            try:
                if ws:
                    # Safely check WebSocket state and close
                    try:
                        if hasattr(ws, "closed") and not ws.closed:
                            await ws.close()
                        elif hasattr(ws, "state") and ws.state.name != "CLOSED":
                            await ws.close()
                        else:
                            # If no closed attribute, try to close directly
                            await ws.close()
                    except Exception:
                        # Ignore error if close fails
                        pass
                elif self.websocket:
                    try:
                        if (
                                hasattr(self.websocket, "closed")
                                and not self.websocket.closed
                        ):
                            await self.websocket.close()
                        elif (
                                hasattr(self.websocket, "state")
                                and self.websocket.state.name != "CLOSED"
                        ):
                            await self.websocket.close()
                        else:
                            # If no closed attribute, try to close directly
                            await self.websocket.close()
                    except Exception:
                        # Ignore error if close fails
                        pass
            except Exception as ws_error:
                self.logger.bind(tag=TAG).error(f"Error closing WebSocket connection: {ws_error}")

            if self.tts:
                await self.tts.close()
            if self.asr:
                await self.asr.close()

            # Finally close thread pool (avoid blocking)
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(
                        f"Error closing thread pool: {executor_error}"
                    )
                self.executor = None
            self.logger.bind(tag=TAG).info("Connection resources released")
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Error closing connection: {e}")
        finally:
            # Ensure stop event is set
            if self.stop_event:
                self.stop_event.set()

    def clear_queues(self):
        """Clear all task queues"""
        if self.tts:
            self.logger.bind(tag=TAG).debug(
                f"Start cleanup: TTS queue size={self.tts.tts_text_queue.qsize()}, audio queue size={self.tts.tts_audio_queue.qsize()}"
            )

            # Clear queue in non-blocking way
            for q in [
                self.tts.tts_text_queue,
                self.tts.tts_audio_queue,
                self.report_queue,
            ]:
                if not q:
                    continue
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            # Reset audio rate controller (cancel background tasks and clear queue)
            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                self.audio_rate_controller.reset()
                self.logger.bind(tag=TAG).debug("Audio rate controller reset")

            self.logger.bind(tag=TAG).debug(
                f"Cleanup finished: TTS queue size={self.tts.tts_text_queue.qsize()}, audio queue size={self.tts.tts_audio_queue.qsize()}"
            )

    def reset_vad_states(self):
        """只重置 VAD 状态（auto 模式下 ASR 取走音频后调用）"""
        self.client_audio_buffer.clear()
        self.client_have_voice = False
        self.client_voice_stop = False
        self.client_voice_window.clear()
        self.last_is_voice = False
        self.vad_last_voice_time = 0.0

    def reset_audio_states(self):
        """
        重置所有音频相关状态(VAD + ASR)
        """
        self.reset_vad_states()

        # Clear ASR buffers
        self.asr_audio.clear()

        self.logger.bind(tag=TAG).debug("All audio states reset.")

    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            # Use the existing chat method
            self.chat(text)

            # After chat is complete, close the connection
            self.close_after_chat = True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")

    async def _check_timeout(self):
        """Check connection timeout"""
        try:
            while not self.stop_event.is_set():
                last_activity_time = self.last_activity_time
                if self.need_bind:
                    last_activity_time = self.first_activity_time

                # Check for timeout (only if timestamp is initialized)
                if last_activity_time > 0.0:
                    current_time = time.time() * 1000
                    if current_time - last_activity_time > self.timeout_seconds * 1000:
                        if not self.stop_event.is_set():
                            self.logger.bind(tag=TAG).info("Connection timed out, preparing to close")
                            # Set stop event to prevent duplicate processing
                            self.stop_event.set()
                            # Wrap close operation with try-except to ensure it doesn't block due to exception
                            try:
                                await self.close(self.websocket)
                            except Exception as close_error:
                                self.logger.bind(tag=TAG).error(
                                    f"Error closing connection due to timeout: {close_error}"
                                )
                        break
                # Check every 10 seconds to avoid being too frequent
                await asyncio.sleep(10)
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"Timeout check task error: {e}")
        finally:
            self.logger.bind(tag=TAG).info("Timeout check task exited")

    @staticmethod
    def _extract_direct_answer_response(arguments_str):
        """从 direct_answer 的参数中提取 response 值。
        优先使用 json.loads 标准解析，流式阶段 fallback 到字符串提取。
        """
        if not arguments_str:
            return ""
        # 优先尝试标准 JSON 解析（适用于完整且格式正确的 JSON）
        try:
            data = json.loads(arguments_str)
            if isinstance(data, dict) and "response" in data:
                return data["response"]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback：流式阶段 JSON 可能不完整，使用字符串提取
        marker = '"response": "'
        idx = arguments_str.find(marker)
        if idx < 0:
            marker = '"response":"'
            idx = arguments_str.find(marker)
        if idx < 0:
            return ""
        start = idx + len(marker)
        raw = arguments_str[start:]
        # 去掉末尾的 JSON 闭合符号（如果已完整）
        if raw.endswith('"}'):
            raw = raw[:-2]
        elif raw.endswith('"'):
            raw = raw[:-1]
        # 处理 JSON 转义
        raw = raw.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        return raw

    @staticmethod
    def _clean_response_garbage(text):
        """清理 response 中可能泄漏的 JSON 闭合符号。
        模型有时会在 response 内容中生成 JSON 闭合字符（如 ）"}} 或 '})，
        这些不是故事内容的一部分，需要去除。
        """
        if not text:
            return text
        # 清理独立一行的 JSON 闭合垃圾（如 ）"}}  '}}  "}}  }}  } ）
        _garbage_chars = frozenset('")\'}）')
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) <= 8 and all(c in _garbage_chars for c in stripped):
                continue
            cleaned.append(line)
        result = '\n'.join(cleaned)
        # 清理末尾残留的 JSON 闭合符号
        result = re.sub(r'["\'}\]]+$', '', result.rstrip()).rstrip()
        return result

    def _merge_tool_calls(self, tool_calls_list, tools_call):
        """Merge tool call list

        Args:
            tool_calls_list: Collected tool call list
            tools_call: New tool call
        """
        for tool_call in tools_call:
            tool_index = getattr(tool_call, "index", None)
            if tool_index is None:
                if tool_call.function.name:
                    # Has function_name, indicating new tool call
                    tool_index = len(tool_calls_list)
                else:
                    tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0

            # Ensure list has enough space
            if tool_index >= len(tool_calls_list):
                tool_calls_list.append({"id": "", "name": "", "arguments": ""})

            # Update tool call info
            if tool_call.id:
                tool_calls_list[tool_index]["id"] = tool_call.id
            if tool_call.function.name:
                tool_calls_list[tool_index]["name"] = tool_call.function.name
            if tool_call.function.arguments:
                tool_calls_list[tool_index]["arguments"] += tool_call.function.arguments
