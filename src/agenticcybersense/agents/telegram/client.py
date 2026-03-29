"""Telethon client wrapper for Telegram agent."""

from telethon import TelegramClient, errors
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


class TelegramClientWrapper:
    def __init__(self, api_id: int, api_hash: str, session_name: str = "agentic_telegram_session"):
        self.api_id = int(api_id)
        self.api_hash = api_hash
        self.session_name = session_name
        self._client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        self._started = False
    
    async def __aenter__(self) -> "TelegramClientWrapper":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        if not self._started:
            await self._client.start()
            self._started = True
            logger.debug("Telethon client started (session=%s)", self.session_name)

    async def stop(self) -> None:
        if self._started:
            await self._client.disconnect()
            self._started = False
            logger.debug("Telethon client disconnected")

    async def fetch_channel_messages(self, channel_username: str, limit: int = 50) -> List[Any]:
        """Return list of Telethon Message objects for channel_username (public).
        If the channel is not found, returns empty list.
        """
        await self.start()
        try:
            entity = await self._client.get_entity(channel_username)
            messages = await self._client.get_messages(entity, limit=limit)
            return messages
        except errors.rpcerrorlist.UsernameNotOccupiedError:
            logger.warning("Channel not found: %s", channel_username)
            return []
        except Exception as e:
            logger.exception("Error fetching messages for %s: %s", channel_username, e)
            raise