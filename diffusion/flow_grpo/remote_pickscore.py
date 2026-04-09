from diffusion.flow_grpo.remote_reward_base import RemoteImageRewardScorer


class RemotePickScoreScorer(RemoteImageRewardScorer):
    """Remote PickScore wrapper with strict request/response contract."""

    def __init__(self):
        super().__init__(reward_name="pickscore", response_key="pickscore")
