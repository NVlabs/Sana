from diffusion.flow_grpo.remote_reward_base import RemoteImageRewardScorer


class RemoteOCRScorer(RemoteImageRewardScorer):
    """Remote OCR wrapper with strict request/response contract."""

    def __init__(self):
        # OCR response key from remote service is "ocr_reward".
        super().__init__(
            reward_name="ocr",
            response_key="ocr_reward",
            extra_data={"ocr_use_gpu": True},
        )

    def build_request_prompts(self, prompts):
        # Align to OCR examples that use a double-quoted target phrase.
        normalized = []
        for p in prompts:
            if isinstance(p, str) and '"' not in p and p.count("'") >= 2:
                first = p.find("'")
                last = p.rfind("'")
                if 0 <= first < last:
                    p = p[:first] + '"' + p[first + 1 : last] + '"' + p[last + 1 :]
            normalized.append(p)
        return normalized
