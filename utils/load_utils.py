import numpy as np
import os
from PIL import Image
from utils.augment import  grayscale_variation


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to between 0 and 1.

    Args:
        image (array): Image array.

    Returns:
        (array): Array with values between 0 and 1
    """
    i_min, i_max = image.min(), image.max()
    return (255 * ((image - i_min) / ((i_max - i_min) if i_max > i_min else 1))).astype(np.uint8)


def get_samples(N: int, T: int) -> np.ndarray:
    """Samples T frames from N frames.

    Args:
        N (int): Number of frames in video.
        T (int): Sequence length.

    Returns:
        np.ndarray: Array containing min(T, N) indices.
    """

    samples = np.arange(N) if N < T else np.linspace(0, N - 1, T)
    return samples.astype(np.int32)


def load_joints(joint_path: str, frame_idxs: np.ndarray, T: int) -> np.ndarray:
    """Loads joint points.

    Args:
        joint_path (str): Path to joint points.
        frame_idxs (np.ndarray): Selected frames.
        T (int): Sequence length.

    Returns:
        np.ndarray: Sequence of joint points of shape (T, 22 * D).
    """

    joint_points = np.loadtxt(joint_path, dtype=np.float32)[frame_idxs]
    
    palm_idx = 1
    num_frames = joint_points.shape[0]
    
    joint_points = joint_points.reshape(num_frames, 22, -1)
    joint_points = (joint_points - joint_points[0, palm_idx, :]).reshape(num_frames, -1)

    return joint_points


def load_image_sequence(image_dir: str, frame_idxs: np.ndarray, T: int, transform_dict: dict, mode: str = "shrec") -> np.ndarray:
    """Loads image sequence and applies necessary processing.

    Args:
        image_dir (str): Path to image folder.
        frame_idxs (np.ndarray): Selected frames.
        T (int): Sequence length.
        transform_dict (dict): Dict containing transform specifications.

    Returns:
        np.ndarray: Sequences of images of shape (T, 1, H, W)
    """
    
    H_new, W_new = transform_dict["resize"]["H_new"], transform_dict["resize"]["W_new"]
    image_blocks = np.zeros((len(frame_idxs), 1, H_new, W_new), dtype=np.uint8)
    file_name = "{}_depth.png" if mode == "shrec" else "depth_{}.png"
    
    for i, idx in enumerate(frame_idxs):
        path = os.path.join(image_dir, file_name.format(idx))
        image = Image.open(path).resize((W_new, H_new), Image.LANCZOS)

        image = np.array(image)

        if transform_dict["gvar"]:
            image = grayscale_variation(image, **transform_dict["gvar"])

        else:
            image = normalize(image)

        image_blocks[i, 0, :, :] = image

    return image_blocks