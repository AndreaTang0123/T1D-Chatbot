import asyncio
import json
from typing import Dict, Any

from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType
from core.utils.status_snapshot import build_status_snapshot

TAG = __name__


class StatusMessageHandler(TextMessageHandler):
    """浏览器客户端的血糖胶囊 / Today 面板：{"type":"status"} → 回复同名的状态快照"""

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.STATUS

    async def handle(self, conn, msg_json: Dict[str, Any]) -> None:
        # 连接上的消息是逐条 await 处理的；补拉 Nightscout 最长 20s，不能挡住随后的 listen 和音频帧
        asyncio.create_task(self._send_snapshot(conn))

    async def _send_snapshot(self, conn) -> None:
        try:
            # 可能要读几 MB 的 CSV、补拉 Nightscout，放到线程里
            snapshot = await asyncio.to_thread(build_status_snapshot, conn.client_id, conn.config)
            await conn.websocket.send(json.dumps(snapshot, ensure_ascii=False))
        except Exception as e:
            conn.logger.bind(tag=TAG).error(f"Failed to send status snapshot: {e}")
