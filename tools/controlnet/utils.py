import random

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from tools.controlnet.annotator.hed import HEDdetector
from tools.controlnet.annotator.util import HWC3, nms, resize_image

preprocessor = None


def transform_control_signal(control_signal, hw):
    if isinstance(control_signal, str):
        control_signal = Image.open(control_signal)
    elif isinstance(control_signal, Image.Image):
        control_signal = control_signal
    elif isinstance(control_signal, np.ndarray):
        control_signal = Image.fromarray(control_signal)
    else:
        raise ValueError("control_signal must be a path or a PIL.Image.Image or a numpy array")

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((int(hw[0, 0]), int(hw[0, 1])), interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
            T.CenterCrop((int(hw[0, 0]), int(hw[0, 1]))),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    return transform(control_signal).unsqueeze(0)


def get_scribble_map(input_image, det, detect_resolution=512, thickness=None):
    """
    Generate scribble map from input image

    Args:
        input_image: Input image (numpy array, HWC format)
        det: Detector type ('Scribble_HED', 'Scribble_PIDI', 'None')
        detect_resolution: Processing resolution
        thickness: Line thickness (between 0-24, None for random)

    Returns:
        Processed scribble map
    """
    global preprocessor

    # Initialize detector
    if "HED" in det and not isinstance(preprocessor, HEDdetector):
        preprocessor = HEDdetector()

    input_image = HWC3(input_image)

    if det == "None":
        detected_map = input_image.copy()
    else:
        # Generate scribble map
        detected_map = preprocessor(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)

        # Post-processing
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        # Control line thickness
        if thickness is None:
            thickness = random.randint(0, 24)  # Random thickness, including 0
        if thickness == 0:
            # Use erosion operation to get thinner lines
            kernel = np.ones((4, 4), np.uint8)
            detected_map = cv2.erode(detected_map, kernel, iterations=1)
        elif thickness > 1:
            kernel_size = thickness // 2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            detected_map = cv2.dilate(detected_map, kernel, iterations=1)

    return detected_map
