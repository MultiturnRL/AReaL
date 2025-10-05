from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class MCPClient:
    def __init__(self, sandbox_gateway: str):
        self.stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.sandbox_gateway = sandbox_gateway

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def attach_session(self, sandbox_uuid: str) -> ClientSession:
        try:
            self.stack = AsyncExitStack()

            read, write, _ = await self.stack.enter_async_context(
                streamablehttp_client(
                    f"{self.sandbox_gateway}/mcp",
                    headers={"X-MCP-Session-ID": sandbox_uuid},
                )
            )

            sess = await self.stack.enter_async_context(ClientSession(read, write))

            await sess.initialize()

            self.session = sess
            return sess
        except Exception:
            await self.close_session()
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def close_session(self) -> None:
        if self.stack is not None:
            await self.stack.aclose()
            self.stack = None
        self.session = None