import json
from core.handle.textMessageHandlerRegistry import TextMessageHandlerRegistry

TAG = __name__

class TextMessageProcessor:
    """Main Message Processor Class"""

    def __init__(self, registry: TextMessageHandlerRegistry):
        self.registry = registry

    async def process_message(self, conn, message: str) -> None:
        """Main entry point for message processing"""
        try:
            # Parse JSON message
            msg_json = json.loads(message)

            # Handle JSON message
            if isinstance(msg_json, dict):
                message_type = msg_json.get("type")

                conn.logger.bind(tag=TAG).info(f"Received {message_type} message: {message}")

                # Get and execute the handler
                handler = self.registry.get_handler(message_type)
                if handler:
                    await handler.handle(conn, msg_json)
                else:
                    conn.logger.bind(tag=TAG).error(f"Received unknown message type: {message}")
            
            # Handle numeric messages
            elif isinstance(msg_json, int):
                conn.logger.bind(tag=TAG).info(f"Received numeric message: {message}")
                await conn.websocket.send(message)

        except json.JSONDecodeError:
            # Forward non-JSON messages directly
            conn.logger.bind(tag=TAG).error(f"Failed to parse message: {message}")
            await conn.websocket.send(message)

