from diffusion.flow_grpo.remote_reward_base import RemoteImageRewardScorer


class RemoteImageRewardServerScorer(RemoteImageRewardScorer):
    """Remote ImageReward wrapper with strict request/response contract."""

    def __init__(self):
        super().__init__(reward_name="image_reward", response_key="image_reward")
