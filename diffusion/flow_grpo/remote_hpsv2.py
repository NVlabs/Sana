from diffusion.flow_grpo.remote_reward_base import RemoteImageRewardScorer


class RemoteHPSv2Scorer(RemoteImageRewardScorer):
    """Remote HPSv2 wrapper with strict request/response contract."""

    def __init__(self):
        super().__init__(reward_name="hpsv2", response_key="hpsv2")
