import json
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_fid.inception import InceptionV3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from tools.metrics.utils import tracker

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path)
            assert img.mode == "RGB"
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception as e:
            raise FileNotFoundError(path, "\n", e)

        return img


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    model.eval()

    if batch_size > len(files):
        print("Warning: batch size is bigger than the data size. " "Setting batch size to data size")
        batch_size = len(files)
    transform = T.Compose(
        [
            T.Resize(args.img_size),  # Image.BICUBIC
            T.CenterCrop(args.img_size),
            T.ToTensor(),
        ]
    )
    dataset = ImagePathDataset(files, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader, desc=f"FID: {args.exp_name}", position=args.gpu_id, leave=True):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers=1, flag="ref"):
    if path.endswith(".npz"):
        print("loaded from npz files")
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    elif path.endswith(".json"):
        with open(path) as file:
            data_dict = json.load(file)
        all_lines = list(data_dict.keys())[:sample_nums]

        files = []
        if isinstance(all_lines, list):
            for k in all_lines:
                v = data_dict[k]
                if "PG-eval-data" in args.img_path:
                    img_path = os.path.join(args.img_path, v["category"], f"{k}.jpg")
                else:
                    img_path = os.path.join(args.img_path, args.exp_name, f"{k}.jpg")
                files.append(img_path)
        elif isinstance(all_lines, dict):
            assert sample_nums >= 30_000, ValueError(f"{sample_nums} is not supported for json files")
            for k, v in all_lines.items():
                if "PG-eval-data" in args.img_path:
                    img_path = os.path.join(args.img_path, v["category"], f"{k}.jpg")
                else:
                    img_path = os.path.join(args.img_path, args.exp_name, f"{k}.jpg")
                files.append(img_path)

        files = sorted(files)
        m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers)
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob(f"*.{ext}")])

        m, s = calculate_activation_statistics(files, model, batch_size, dims, device, num_workers)
    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers, flag="ref")
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size, dims, device, num_workers, flag="gen")
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def save_fid_stats(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    if not os.path.exists(paths[0]):
        raise RuntimeError("Invalid path: %s" % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError("Existing output file: %s" % paths[1])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, dims, device, num_workers, flag="ref")
    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def main():
    txt_path = args.txt_path if args.txt_path is not None else args.img_path
    save_txt_path = os.path.join(txt_path, f"{args.exp_name}_sample{sample_nums}.txt")
    if os.path.exists(save_txt_path):
        with open(save_txt_path) as f:
            fid_value = f.readlines()[0].strip()
        print(f"FID {fid_value}: {args.exp_name}")
        return {args.exp_name: float(fid_value)}

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

    if args.save_stats:
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(args.path, args.batch_size, device, args.dims, num_workers)
    print(f"FID {fid_value}: {args.exp_name}")
    with open(save_txt_path, "w") as file:
        file.write(str(fid_value))

    return {args.exp_name: fid_value}


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument(
        "--num-workers", type=int, help="Number of processes to use for data loading.  Defaults to `min(8, num_cpus)`"
    )
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use. Like cuda, cuda:0 or cpu")

    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="Sana")
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--sample_nums", type=int, default=30_000)

    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help="Dimensionality of Inception features to use.  By default, uses pool3 features",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Generate an npz archive from a directory of samples.  The first path is used as input and the second as output.",
    )
    parser.add_argument("--stat", action="store_true")
    parser.add_argument(
        "--path", type=str, nargs=2, default=["", ""], help="Paths to the generated images or  to .npz statistic files"
    )

    # online logging setting
    parser.add_argument("--log_metric", type=str, default="metric")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_fid", action="store_true")
    parser.add_argument("--suffix_label", type=str, default="", help="used for fid online log")
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
    if args.stat:
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
        save_fid_stats(args.path, args.batch_size, device, args.dims, num_workers)
    else:
        print(args.path, args.exp_name)
        args.exp_name = os.path.basename(args.exp_name) or os.path.dirname(args.exp_name)
        fid_result = main()
        if args.log_fid:
            tracker(args, fid_result, args.suffix_label, pattern=args.tracker_pattern, metric="FID")
