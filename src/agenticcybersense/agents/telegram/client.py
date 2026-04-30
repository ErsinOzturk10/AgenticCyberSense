"""Telethon client wrapper for Telegram agent.

Provides a thin async context-manager wrapper around Telethon's TelegramClient
to simplify start/stop lifecycle handling and common helper methods used by
the Telegram agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self, cast

if TYPE_CHECKING:
    import types  # moved into TYPE_CHECKING to satisfy TC003 (used only for annotations)

try:
    from telethon import TelegramClient as _TelegramClient  # type: ignore[import-untyped]
    from telethon import errors as _telethon_errors
except ImportError:  # telethon is optional
    _TelegramClient = None
    _telethon_errors = None

UsernameNotOccupiedError = _telethon_errors.rpcerrorlist.UsernameNotOccupiedError if _telethon_errors is not None else RuntimeError

logger = logging.getLogger(__name__)


class TelegramClientWrapper:
    """A small wrapper around Telethon's TelegramClient.

    Usage:
        async with TelegramClientWrapper(api_id, api_hash) as client:
            messages = await client.fetch_channel_messages("@somechannel")
    """

    def __init__(self, api_id: int, api_hash: str, session_name: str = "agentic_telegram_session") -> None:
        """Initialize the wrapper with Telethon credentials and session name.

        Args:
            api_id: Numeric API ID from Telegram.
            api_hash: API hash from Telegram.
            session_name: Session name used by Telethon to store session data.

        """
        self.api_id = int(api_id)
        self.api_hash = api_hash
        self.session_name = session_name
        if _TelegramClient is None:
            msg = "telethon is not installed"
            raise RuntimeError(msg)
        self._client: Any = _TelegramClient(self.session_name, self.api_id, self.api_hash)
        self._started = False

    async def __aenter__(self) -> Self:
        """Async context manager enter: ensure the underlying client is started."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: types.TracebackType | None) -> None:
        """Async context manager exit: stop/cleanup the underlying client.

        Args:
            exc_type: Exception type if an exception was raised in the context, else None.
            exc: Exception instance if one was raised, else None.
            tb: Traceback object if an exception was raised, else None.

        """
        await self.stop()

    async def start(self) -> None:
        """Start the Telethon client if not already started.

        This method initializes the connection to Telegram by starting the underlying
        Telethon client. It ensures the client is only started once by tracking the
        started state. Once successfully started, a debug log message is recorded
        with the session name.

        Raises:
            Any exception raised by the underlying Telethon client's start method.

        """
        if not self._started:
            await self._client.start()
            self._started = True
            logger.debug("Telethon client started (session=%s)", self.session_name)

    async def stop(self) -> None:
        """Stop the Telethon client connection.

        Disconnects the client if it is currently running and updates the internal
        state to reflect that the client is no longer active.
        """
        if self._started:
            await self._client.disconnect()
            self._started = False
            logger.debug("Telethon client disconnected")

    async def fetch_channel_messages(self, channel_username: str, limit: int = 50) -> list[Any]:
        """Return list of Telethon Message objects for a public channel.

        If the channel is not found, returns an empty list.

        Args:
            channel_username: Public channel username (e.g. "@channel" or "t.me/username").
            limit: Maximum number of messages to fetch.

        Returns:
            A list of Telethon Message objects (or an empty list).

        """
        await self.start()
        try:
            entity = await self._client.get_entity(channel_username)
            messages = await self._client.get_messages(entity, limit=limit)
            return cast("list[Any]", messages)
        except UsernameNotOccupiedError:
            logger.warning("Channel not found: %s", channel_username)
            return []
        except Exception:
            # Log full exception and re-raise to let caller handle it as appropriate.
            # Do not pass the exception object to logger.exception (redundant); it logs exc_info automatically.
            logger.exception("Error fetching messages for %s", channel_username)
            raise
