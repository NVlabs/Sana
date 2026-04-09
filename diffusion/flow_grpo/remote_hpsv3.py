from diffusion.flow_grpo.remote_reward_base import RemoteImageRewardScorer


class RemoteHPSv3Scorer(RemoteImageRewardScorer):
    """Remote HPSv3 wrapper with strict request/response contract."""

    def __init__(self):
        super().__init__(reward_name="hpsv3", response_key="hpsv3")
