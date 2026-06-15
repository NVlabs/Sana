import numpy as np
import torch


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def filter_top_bottom_k_gpu(data, unique_num, num_in_group, k=2):
    prompts = data["prompt_ids"]  # [N, L]
    advs = data["advantages"][:, 0]  # [N]

    # 1. Pure GPU grouping.
    # unique identifies unique rows in [N, 300].
    # inverse_indices has shape [N] and values 0..num_groups-1, indicating each row's unique group.
    unique_prompts, inverse_indices = torch.unique(prompts, dim=0, return_inverse=True)

    num_groups = unique_prompts.shape[0]
    keep_indices = []

    print(f"unique_num: {unique_num}, num_groups: {num_groups}")
    assert unique_num == num_groups, f"unique_num: {unique_num}, num_groups: {num_groups}"

    # 2. Iterate over each group; the group count is small, so a Python loop is acceptable.
    for group_idx in range(num_groups):
        # Create a mask for the current group.
        group_mask = inverse_indices == group_idx

        # Get the global indices for the current group.
        # nonzero returns [m, 1]; squeeze converts it to [m].
        global_indices = torch.nonzero(group_mask).squeeze(1)

        current_count = global_indices.numel()

        assert current_count == num_in_group, f"current_count: {current_count}, num_in_group: {num_in_group}"

        # Get the corresponding advantages.
        group_advs = advs[global_indices]

        # Check whether there are enough samples.
        if group_advs.shape[0] < 2 * k:
            # If there are not enough samples, the policy could skip or error; this implementation errors.
            assert False, f"group_advs.shape[0]: {group_advs.shape[0]}, k: {k}"

        # 3. Sort and select Top/Bottom.
        # argsort returns indices relative to the group.
        sorted_relative_indices = torch.argsort(group_advs)

        # Map back to global indices.
        sorted_global_indices = global_indices[sorted_relative_indices]

        # Select the smallest k entries (Bottom).
        keep_indices.append(sorted_global_indices[:k])
        # Select the largest k entries (Top).
        keep_indices.append(sorted_global_indices[-k:])

    # 4. Concatenate indices and slice.
    if len(keep_indices) > 0:
        final_indices = torch.cat(keep_indices)
        # Optional: preserve the original batch order.
        final_indices, _ = torch.sort(final_indices)
    else:
        # Edge case: no group satisfies the condition.
        final_indices = torch.tensor([], device=prompts.device, dtype=torch.long)

    # 5. Rebuild the dictionary.
    new_dict = {}
    n_total = prompts.shape[0]
    for key, val in data.items():
        if torch.is_tensor(val) and val.shape[0] == n_total:
            new_dict[key] = val[final_indices]
        else:
            new_dict[key] = val

    return new_dict


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
