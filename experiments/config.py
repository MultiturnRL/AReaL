from dataclasses import dataclass
from areal.api.cli_args import GRPOConfig

@dataclass
class Config(GRPOConfig):
    sandbox_control_plane: str = ""
    sandbox_gateway: str = ""
    sandbox_fusion_url: str = ""
    max_turns: int = 0
    sandbox_image_override: str | None = None