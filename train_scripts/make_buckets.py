import json
import math
import os
import os.path as osp
from itertools import chain

import pyrallis
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from diffusion.model.builder import get_vae, vae_encode
from diffusion.utils.config import SanaConfig


@pyrallis.wrap()
def main(config: SanaConfig) -> None:
    preferred_pixel_count = config.model.image_size * config.model.image_size

    min_size = config.model.image_size // 2
    max_size = config.model.image_size * 2
    step = 32

    ratios_array = []
    while min_size != max_size:
        width = int(preferred_pixel_count / min_size)
        if width % step != 0:
            mod = width % step
            if mod < step // 2:
                width -= mod
            else:
                width += step - mod

        ratio = min_size / width

        ratios_array.append((ratio, (int(min_size), width)))
        min_size += step

    def get_closest_ratio(height: float, width: float):
        aspect_ratio = height / width
        closest_ratio = min(ratios_array, key=lambda ratio: abs(ratio[0] - aspect_ratio))
        return closest_ratio

    def get_preffered_size(height: float, width: float):
        pixel_count = height * width

        scale = math.sqrt(pixel_count / preferred_pixel_count)
        return height / scale, width / scale

    class BucketsDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, skip_files):
            valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
            self.files = [
                osp.join(data_dir, f)
                for f in os.listdir(data_dir)
                if osp.isfile(osp.join(data_dir, f))
                and osp.splitext(f)[1].lower() in valid_extensions
                and osp.join(data_dir, f) not in skip_files
            ]

            self.transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = self.files[idx]
            img = Image.open(path).convert("RGB")
            ratio = get_closest_ratio(img.height, img.width)
            prefsize = get_preffered_size(img.height, img.width)

            crop = T.Resize(ratio[1], interpolation=InterpolationMode.BICUBIC)
            return {
                "img": self.transform(crop(img)),
                "size": torch.tensor([ratio[1][0], ratio[1][1]]),
                "prefsize": torch.tensor([prefsize[0], prefsize[1]]),
                "ratio": ratio[0],
                "path": path,
            }

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, "cuda").to(torch.float16)

    def encode_images(batch, vae):
        with torch.no_grad():
            z = vae_encode(
                config.vae.vae_type,
                vae,
                batch,
                sample_posterior=config.vae.sample_posterior,  # Adjust as necessary
                device="cuda",
            )
        return z

    if os.path.exists(config.data.buckets_file):
        with open(config.data.buckets_file) as json_file:
            buckets = json.load(json_file)
            existings_images = set(chain.from_iterable(buckets.values()))
    else:
        buckets = {}
        existings_images = set()

    def add_to_list(key, item):
        if key in buckets:
            buckets[key].append(item)
        else:
            buckets[key] = [item]

    for path in config.data.data_dir:
        print(f"Processing {path}")
        dataset = BucketsDataset(path, existings_images)
        dataloader = DataLoader(dataset, batch_size=1)
        for batch in tqdm(dataloader):
            img = batch["img"]
            size = batch["size"]
            ratio = batch["ratio"]
            image_path = batch["path"]
            prefsize = batch["prefsize"]

            encoded = encode_images(img.to(torch.half), vae)

            for i in range(0, len(encoded)):
                filename_wo_ext = os.path.splitext(os.path.basename(image_path[i]))[0]
                add_to_list(str(ratio[i].item()), image_path[i])

                torch.save(
                    {"img": encoded[i].detach().clone(), "size": size[i], "prefsize": prefsize[i], "ratio": ratio[i]},
                    f"{path}/{filename_wo_ext}_img.npz",
                )

    with open(config.data.buckets_file, "w") as json_file:
        json.dump(buckets, json_file, indent=4)

    for ratio in buckets.keys():
        print(f"{float(ratio):.2f}: {len(buckets[ratio])}")


if __name__ == "__main__":
    main()
