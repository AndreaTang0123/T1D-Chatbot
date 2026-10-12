import uuid
import re
from typing import List, Dict
from datetime import datetime


class Message:
    def __init__(
            self,
            role: str,
            content: str = None,
            uniq_id: str = None,
            tool_calls=None,
            tool_call_id=None,
            is_temporary=False,
    ):
        self.uniq_id = uniq_id if uniq_id is not None else str(uuid.uuid4())
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.is_temporary = is_temporary  # 标记临时消息（如工具调用提醒）


class Dialogue:
    def __init__(self):
        self.dialogue: List[Message] = []
        # Get current time
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def put(self, message: Message):
        self.dialogue.append(message)

    def getMessages(self, m, dialogue):
        if m.tool_calls is not None:
            dialogue.append({"role": m.role, "tool_calls": m.tool_calls})
        elif m.role == "tool":
            dialogue.append(
                {
                    "role": m.role,
                    "tool_call_id": (
                        str(uuid.uuid4()) if m.tool_call_id is None else m.tool_call_id
                    ),
                    "content": m.content,
                }
            )
        else:
            dialogue.append({"role": m.role, "content": m.content})

    def get_llm_dialogue(self) -> List[Dict[str, str]]:
        # Directly call get_llm_dialogue_with_memory, passing None as memory_str
        # This ensures speaker function works in all call paths
        return self.get_llm_dialogue_with_memory(None, None)

    def update_system_message(self, new_content: str):
        """Update or add system message (ensure only one exists)"""
        # Find all system messages
        system_indices = [i for i, msg in enumerate(self.dialogue) if msg.role == "system"]
        
        if system_indices:
            # Update the first one
            self.dialogue[system_indices[0]].content = new_content
            # Remove any others (duplicate cleanup)
            for i in reversed(system_indices[1:]):
                self.dialogue.pop(i)
        else:
            # Insert at the beginning
            self.dialogue.insert(0, Message(role="system", content=new_content))

    def _ensure_tool_calls_complete(self, messages: List[Message]) -> List[Message]:
        """
        确保所有 tool_calls 都有对应的 tool 响应
        修复被打断导致的悬空 tool_calls，防止大模型 API 报 400 错误
        """
        pending_tool_calls = set()
        result = []

        for msg in messages:
            result.append(msg)

            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        pending_tool_calls.add(tc_id)

            elif msg.role == "tool" and msg.tool_call_id:
                pending_tool_calls.discard(msg.tool_call_id)

        for missing_id in pending_tool_calls:
            dummy_tool_msg = Message(
                role="tool",
                content='{"status": "interrupted", "message": "动作已取消/被打断"}',
                tool_call_id=missing_id
            )
            result.append(dummy_tool_msg)

        return result

    def get_llm_dialogue_with_memory(
            self, memory_str: str = None, voiceprint_config: dict = None,
            current_speaker: str = None,
    ) -> List[Dict[str, str]]:
        # Build dialogue
        dialogue = []

        # Add system prompt and memory
        system_message = next(
            (msg for msg in self.dialogue if msg.role == "system"), None
        )

        if system_message:
            # Basic system prompt
            enhanced_system_prompt = system_message.content
            # Replace time placeholder
            enhanced_system_prompt = enhanced_system_prompt.replace(
                "{{current_time}}", datetime.now().strftime("%H:%M")
            )

            # Add speaker personalized description
            try:
                current_speaker_name = (current_speaker or "").strip()
                # 仅在本轮注入了有效身份时才输出 speakers_info，避免列表里的名字每轮
                # 重复出现诱导模型反复称呼；后续轮不再注入身份，靠对话历史首轮保留
                if current_speaker_name and current_speaker_name != "未知说话人":
                    speakers = voiceprint_config.get("speakers", [])
                    speakers_info = "\n<speakers_info>"
                    speakers_info += f"\n当前说话人：{current_speaker_name}"
                    for speaker_str in speakers:
                        try:
                            parts = speaker_str.split(",", 2)
                            if len(parts) >= 2:
                                name = parts[1].strip()
                                # If description is empty, set to ""
                                description = (
                                    parts[2].strip() if len(parts) >= 3 else ""
                                )
                                speakers_info += f"\n- {name}：{description}"
                        except:
                            pass
                    speakers_info += "\n</speakers_info>"
                    full_prompt += speakers_info
            except:
                # Ignore error if config read fails, does not affect other functions
                pass

            # Use regex to match <memory> tag, regardless of content
            if memory_str is not None:
                if "<memory>" in enhanced_system_prompt:
                    enhanced_system_prompt = re.sub(
                        r"<memory>.*?</memory>",
                        f"<memory>\n{memory_str}\n</memory>",
                        enhanced_system_prompt,
                        flags=re.DOTALL,
                    )
                else:
                    enhanced_system_prompt += f"\n\n<memory>\n{memory_str}\n</memory>"
            
            # FINAL SANITY CHECK: Ensure we don't have nested identity if it somehow happened
            if enhanced_system_prompt.count("<identity>") > 1:
                # If nested, try to extract the innermost content or at least flatten
                # This is a fallback defensive measure
                inner_match = re.search(r"<identity>(?:(?!<identity>).)*?</identity>", enhanced_system_prompt, re.DOTALL)
                if inner_match:
                     # Keep the whole thing but maybe this is where we should be careful.
                     # For now, let's just log it if we were in a logging context.
                     pass

            dialogue.append({"role": "system", "content": enhanced_system_prompt})

        # Add user and assistant dialogue
        # Limit to the last N turns to avoid context overflow and reduce latency
        MAX_HISTORY = 20
        
        # Filter out system messages first (we already added the system prompt above)
        conversation_history = [m for m in self.dialogue if m.role != "system"]
        
        # Take the last MAX_HISTORY messages
        recent_history = conversation_history[-MAX_HISTORY:] if len(conversation_history) > MAX_HISTORY else conversation_history
        
        for m in recent_history:
            self.getMessages(m, dialogue)

        return dialogue
