from asyncio.log import logger

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


async def on_request_start(session, trace_config_ctx, params):
    logger.info("HTTP Request: %s %s", params.method, params.url)


async def on_request_end(session, trace_config_ctx, params):
    logger.info("HTTP Response: %s %s", params.response.status, params.response.url)


trace_config = aiohttp.TraceConfig()
trace_config.on_request_start.append(on_request_start)
trace_config.on_request_end.append(on_request_end)


class Sandbox:
    def __init__(self, base_url: str):
        self.session = aiohttp.ClientSession(
            base_url=base_url, trace_configs=[trace_config]
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def spawn(self, image: str) -> str:
        async with self.session.post(
            "/spawn",
            json={
                "image": image,
                "ports": [{"container_port": 3000}],
                "env": {"MCP_HUB_ADDR": "proxy-mcp.apps.svc.cluster.local:3000/mcp"},
                "expose": "ClusterIP",
                "replicas": 1,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            return (await resp.json())["uuid"]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def deprovision(self, sandbox_uuid: str):
        async with self.session.delete(f"/deprovision/{sandbox_uuid}") as _:
            return

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def deprovision_all(self):
        async with self.session.delete("/deprovision-all") as _:
            return

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=0, max=60),
    )
    async def close(self):
        await self.session.close()


if __name__ == "__main__":
    import asyncio

    CLUSTER_IP = "35.87.202.180"

    async def rollout():
        try:
            sandbox = Sandbox(f"http://{CLUSTER_IP}:30693")
            uuid = await sandbox.spawn("timemagic/rl-mcp:general")
            print(f"Spawned sandbox with UUID: {uuid}")
            await asyncio.sleep(10)
            await sandbox.deprovision(uuid)
        except Exception as e:
            print(f"Error occurred: {e}")
            await sandbox.deprovision_all()
        await sandbox.close()

    async def run():
        await asyncio.gather(*[rollout() for _ in range(10)])

    asyncio.run(run())