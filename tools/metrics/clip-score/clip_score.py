import io
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import clip
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from diffusion.data.transforms import get_transform
from tools.metrics.utils import tracker

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


import json

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}
TEXT_EXTENSIONS = {"txt"}


class DummyDataset(Dataset):
    FLAGS = ["img", "txt", "json"]

    def __init__(
        self,
        real_path,
        fake_path,
        real_flag: str = "img",
        fake_flag: str = "img",
        gen_img_path="",
        transform=None,
        tokenizer=None,
    ) -> None:
        super().__init__()
        assert (
            real_flag in self.FLAGS and fake_flag in self.FLAGS
        ), f"CLIP Score only support modality of {self.FLAGS}. However, get {real_flag} and {fake_flag}"
        self.gen_img_path = gen_img_path
        print(f"images are from {gen_img_path}")
        self.real_folder = self._load_img_from_path(real_path)
        self.real_flag = real_flag
        self.fake_data = self._load_txt_from_path(fake_path)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_dict = {}

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_txt(self.fake_data[index])
        sample = dict(real=real_data, fake=fake_data, prompt=self.fake_data[index])
        return sample

    def _load_modality(self, path, modality):
        if modality == "img":
            data = self._load_img(path)
        else:
            raise TypeError(f"Got unexpected modality: {modality}")
        return data

    def _load_txt(self, data):
        if self.tokenizer is not None:
            data = self.tokenizer(data, context_length=77, truncate=True).squeeze()
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load_img_from_path(self, path):
        image_list = []
        if path.endswith(".json"):
            with open(path) as file:
                data_dict = json.load(file)
            all_lines = list(data_dict.keys())[:sample_nums]
            if isinstance(all_lines, list):
                for k in all_lines:
                    img_path = os.path.join(self.gen_img_path, f"{k}.jpg")
                    image_list.append(img_path)
            elif isinstance(all_lines, dict):
                assert sample_nums >= 30_000, ValueError(f"{sample_nums} is not supported for json files")
                for k, v in all_lines.items():
                    img_path = os.path.join(self.gen_img_path, f"{k}.jpg")
                    image_list.append(img_path)

        else:
            raise ValueError(f"Only JSON file type is supported now. Wrong with: {path}")

        return image_list

    def _load_txt_from_path(self, path):
        txt_list = []
        if path.endswith(".json"):
            with open(path) as file:
                data_dict = json.load(file)
            all_lines = list(data_dict.keys())[:sample_nums]
            if isinstance(all_lines, list):
                for k in all_lines:
                    v = data_dict[k]
                    txt_list.append(v["prompt"])
            elif isinstance(all_lines, dict):
                assert sample_nums >= 30_000, ValueError(f"{sample_nums} is not supported for json files")
                for k, v in all_lines.items():
                    txt_list.append(v["prompt"])
        else:
            raise ValueError(f"Only JSON file type is supported now. Wrong with: {path}")

        return txt_list


class DummyTarDataset(IterableDataset):
    def __init__(
        self, tar_path, transform=None, external_json_path=None, prompt_key="prompt", tokenizer=None, **kwargs
    ):
        assert ".tar" in tar_path
        self.sample_nums = args.sample_nums
        self.dataset = (
            wds.WebDataset(tar_path)
            .map(self.safe_decode)
            .to_tuple("png;jpg", "json", "__key__")
            .map(self.process_sample)
            .slice(0, self.sample_nums)
        )
        if external_json_path is not None and os.path.exists(external_json_path):
            print(f"Loading {external_json_path}, wait...")
            self.json_file = json.load(open(external_json_path))
        else:
            self.json_file = {}
            assert prompt_key == "prompt"
        self.prompt_key = prompt_key
        self.transform = transform
        self.tokenizer = tokenizer

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for i, (ori_img, info, key) in enumerate(self.dataset):
            if self.transform is not None:
                img = self.transform(ori_img)

            if key in self.json_file:
                info.update(self.json_file[key])

            prompt = info.get(self.prompt_key, "")
            if not prompt:
                prompt = ""
                print(f"{self.prompt_key} not exist in {key}.json")
            txt_feat = self._load_txt(prompt)

            yield dict(
                real=img, fake=txt_feat, prompt=prompt, ori_img=np.array(img), key=key, prompt_key=self.prompt_key
            )

    def __len__(self):
        return self.sample_nums

    def _load_txt(self, data):
        if self.tokenizer is not None:
            data = self.tokenizer(data, context_length=77, truncate=True).squeeze()
        return data

    @staticmethod
    def process_sample(sample):
        try:
            image_bytes, json_bytes, key = sample
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            json_dict = json.loads(json_bytes)
            return image, json_dict, key
        except (ValueError, TypeError, OSError) as e:
            print(f"Skipping sample due to error: {e}")
            return None

    @staticmethod
    def safe_decode(sample):
        def custom_decode(sample):
            result = {}
            for k, v in sample.items():
                result[k] = v
            return result

        try:
            return custom_decode(sample)
        except Exception as e:
            print(f"skipping sample due to decode error: {e}")
            return None


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag, save_json_path=None):
    score_acc = 0.0
    sample_num = 0.0
    json_dict = {} if save_json_path is not None else None
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader, desc=f"CLIP-Score: {args.exp_name}", position=args.gpu_id, leave=True):
        real_features = forward_modality(model, batch_data["real"], real_flag)
        fake_features = forward_modality(model, batch_data["fake"], fake_flag)

        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)

        score = logit_scale * (fake_features * real_features).sum()
        if save_json_path is not None:
            json_dict[batch_data["key"][0]] = {f"{batch_data['prompt_key'][0]}": f"{score:.04f}"}

        score_acc += score
        sample_num += batch_data["real"].shape[0]

    if save_json_path is not None:
        json.dump(json_dict, open(save_json_path, "w"))
    return score_acc / sample_num


@torch.no_grad()
def calculate_clip_score_official(dataloader):
    import numpy as np
    from torchmetrics.multimodal.clip_score import CLIPScore

    clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    # clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    all_clip_scores = []

    for batch_data in tqdm(dataloader, desc=args.exp_name, position=args.gpu_id, leave=True):
        imgs = batch_data["real"].add_(1.0).mul_(0.5)
        imgs = (imgs * 255).to(dtype=torch.uint8, device=device)

        prompts = batch_data["prompt"]
        clip_scores = clip_score_fn(imgs, prompts).detach().cpu()
        all_clip_scores.append(float(clip_scores))

    clip_scores = float(np.mean(all_clip_scores))
    return clip_scores


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == "img":
        features = model.encode_image(data.to(device))
    elif flag == "txt":
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


def main():
    txt_path = args.txt_path if args.txt_path is not None else args.img_path
    gen_img_path = str(os.path.join(args.img_path, args.exp_name))
    if ".tar" in gen_img_path:
        save_txt_path = os.path.join(txt_path, f"{args.exp_name}_{args.tar_prompt_key}_clip_score.txt").replace(
            ".tar", ""
        )
        save_json_path = save_txt_path.replace(".tar", "").replace(".txt", ".json")
        if os.path.exists(save_json_path):
            print(f"{save_json_path} exists. Finished.")
            return None
    else:
        save_txt_path = os.path.join(txt_path, f"{args.exp_name}_sample{sample_nums}_clip_score.txt")
        save_json_path = None
    if os.path.exists(save_txt_path):
        with open(save_txt_path) as f:
            clip_score = f.readlines()[0].strip()
        print(f"CLIP Score:  {clip_score}: {args.exp_name}")
        return {args.exp_name: float(clip_score)}

    print(f"Loading CLIP model: {args.clip_model}")
    if args.clipscore_type == "diffusers":
        preprocess = get_transform("default_train", 512)
    else:
        model, preprocess = clip.load(args.clip_model, device=device)

    if ".tar" in gen_img_path:
        dataset = DummyTarDataset(
            gen_img_path,
            transform=preprocess,
            external_json_path=args.external_json_file,
            prompt_key=args.tar_prompt_key,
            tokenizer=clip.tokenize,
        )
    else:
        dataset = DummyDataset(
            args.real_path,
            args.fake_path,
            args.real_flag,
            args.fake_flag,
            transform=preprocess,
            tokenizer=clip.tokenize,
            gen_img_path=gen_img_path,
        )
    dataloader = DataLoader(dataset, args.batch_size, num_workers=num_workers, pin_memory=True)

    print("Calculating CLIP Score:")
    if args.clipscore_type == "diffusers":
        clip_score = calculate_clip_score_official(dataloader)
    else:
        clip_score = calculate_clip_score(
            dataloader, model, args.real_flag, args.fake_flag, save_json_path=save_json_path
        )
        clip_score = clip_score.cpu().item()
    print("CLIP Score: ", clip_score)
    with open(save_txt_path, "w") as file:
        file.write(str(clip_score))
    print(f"Result saved at: {save_txt_path}")

    return {args.exp_name: clip_score}


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14", help="CLIP model to use")
    # parser.add_argument('--clip-model', type=str, default='ViT-B/16', help='CLIP model to use')
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--sample_nums", type=int, default=30_000)
    parser.add_argument("--exp_name", type=str, default="Sana")
    parser.add_argument(
        "--num-workers", type=int, help="Number of processes to use for data loading.  Defaults to `min(8, num_cpus)`"
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument("--real_flag", type=str, default="img", help="The modality of real path. Default to img")
    parser.add_argument("--fake_flag", type=str, default="txt", help="The modality of real path. Default to txt")
    parser.add_argument("--real_path", type=str, help="Paths to the generated images")
    parser.add_argument("--fake_path", type=str, help="Paths to the generated images")
    parser.add_argument("--external_json_file", type=str, default=None, help="external meta json file for tar_file")
    parser.add_argument("--tar_prompt_key", type=str, default="prompt", help="key name of prompt in json")

    # online logging setting
    parser.add_argument("--clipscore_type", type=str, default="self", choices=["diffusers", "self"])
    parser.add_argument("--log_metric", type=str, default="metric")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_clip_score", action="store_true")
    parser.add_argument("--suffix_label", type=str, default="", help="used for clip_score online log")
    parser.add_argument("--tracker_pattern", type=str, default="epoch_step", help="used for fid online log")
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="t2i-evit-baseline",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help=("Wandb Project Name"),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sample_nums = args.sample_nums
    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    args.exp_name = os.path.basename(args.exp_name) or os.path.dirname(args.exp_name)
    clip_score_result = main()
    if args.log_clip_score:
        tracker(args, clip_score_result, args.suffix_label, pattern=args.tracker_pattern, metric="CLIP-Score")
