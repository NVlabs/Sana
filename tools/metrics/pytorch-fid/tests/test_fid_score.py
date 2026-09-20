import numpy as np
import pytest
import torch
from PIL import Image
from pytorch_fid import fid_score, inception


@pytest.fixture
def device():
    return torch.device("cpu")


def test_calculate_fid_given_statistics(mocker, tmp_path, device):
    dim = 2048
    m1, m2 = np.zeros((dim,)), np.ones((dim,))
    sigma = np.eye(dim)

    def dummy_statistics(path, model, batch_size, dims, device, num_workers):
        if path.endswith("1"):
            return m1, sigma
        elif path.endswith("2"):
            return m2, sigma
        else:
            raise ValueError

    mocker.patch("pytorch_fid.fid_score.compute_statistics_of_path", side_effect=dummy_statistics)

    dir_names = ["1", "2"]
    paths = []
    for name in dir_names:
        path = tmp_path / name
        path.mkdir()
        paths.append(str(path))

    fid_value = fid_score.calculate_fid_given_paths(paths, batch_size=dim, device=device, dims=dim, num_workers=0)

    # Given equal covariance, FID is just the squared norm of difference
    assert fid_value == np.sum((m1 - m2) ** 2)


def test_compute_statistics_of_path(mocker, tmp_path, device):
    model = mocker.MagicMock(inception.InceptionV3)()
    model.side_effect = lambda inp: [inp.mean(dim=(2, 3), keepdim=True)]

    size = (4, 4, 3)
    arrays = [np.zeros(size), np.ones(size) * 0.5, np.ones(size)]
    images = [(arr * 255).astype(np.uint8) for arr in arrays]

    paths = []
    for idx, image in enumerate(images):
        paths.append(str(tmp_path / f"{idx}.png"))
        Image.fromarray(image, mode="RGB").save(paths[-1])

    stats = fid_score.compute_statistics_of_path(
        str(tmp_path), model, batch_size=len(images), dims=3, device=device, num_workers=0
    )

    assert np.allclose(stats[0], np.ones((3,)) * 0.5, atol=1e-3)
    assert np.allclose(stats[1], np.ones((3, 3)) * 0.25)


def test_compute_statistics_of_path_from_file(mocker, tmp_path, device):
    model = mocker.MagicMock(inception.InceptionV3)()

    mu = np.random.randn(5)
    sigma = np.random.randn(5, 5)

    path = tmp_path / "stats.npz"
    with path.open("wb") as f:
        np.savez(f, mu=mu, sigma=sigma)

    stats = fid_score.compute_statistics_of_path(str(path), model, batch_size=1, dims=5, device=device, num_workers=0)

    assert np.allclose(stats[0], mu)
    assert np.allclose(stats[1], sigma)


def test_image_types(tmp_path):
    in_arr = np.ones((24, 24, 3), dtype=np.uint8) * 255
    in_image = Image.fromarray(in_arr, mode="RGB")

    paths = []
    for ext in fid_score.IMAGE_EXTENSIONS:
        paths.append(str(tmp_path / f"img.{ext}"))
        in_image.save(paths[-1])

    dataset = fid_score.ImagePathDataset(paths)

    for img in dataset:
        assert np.allclose(np.array(img), in_arr)
