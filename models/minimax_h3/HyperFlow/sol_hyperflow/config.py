"""The validated resident HyperFlow request profile, independent of CUDA imports."""
from dataclasses import dataclass

WIDTH, HEIGHT, FPS, STEPS = 1344, 768, 24, 8
DURATION_FRAMES = {5: 124, 10: 243, 15: 362}
TASKS = {"t2v", "i2v", "ref2va"}


@dataclass(frozen=True)
class Request:
    prompt: str
    task: str = "t2v"
    duration: int = 5
    seed: int = 0
    attention_backend: str = "sol_bsa"
    image: object = None
    references: tuple = ()

    def payload(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("A nonempty prompt is required.")
        if self.task not in TASKS or self.duration not in DURATION_FRAMES:
            raise ValueError("Select t2v/i2v/ref2va and a 5/10/15-second duration.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if self.attention_backend not in {"dense", "sol_bsa"}:
            raise ValueError("attention_backend must be dense or sol_bsa.")
        if self.task == "i2v" and self.image is None:
            raise ValueError("I2V requires a first-frame image.")
        if self.task != "i2v" and self.image is not None:
            raise ValueError("A first-frame image is only valid for I2V.")
        if self.task == "ref2va":
            if not 1 <= len(self.references) <= 9:
                raise ValueError("The accelerated Ref2VA profile requires 1–9 image references.")
        elif self.references:
            raise ValueError("Reference images are only valid for Ref2VA.")
        value = dict(prompt=self.prompt, task="ref2v" if self.task == "ref2va" else self.task,
                     height=HEIGHT, width=WIDTH, num_frames=DURATION_FRAMES[self.duration],
                     steps=STEPS, seed=self.seed,
                     performance_mode="quality" if self.attention_backend == "dense" else "speed")
        if self.image is not None:
            value["image"] = self.image
        if self.references:
            value["reference_images"] = list(self.references)
        return value


def validate_world_size(world_size):
    if world_size != 8:
        raise ValueError("This resident acceleration profile requires torchrun --nproc_per_node=8; other GPU counts have not been validated here.")
