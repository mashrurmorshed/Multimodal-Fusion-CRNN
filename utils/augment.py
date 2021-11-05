import albumentations as A
import numpy as np
from PIL import Image
import cv2
from typing import Tuple

def grayscale_variation(image: np.ndarray, eta: int = 10, g_min: int = 155, g_max: int = 255, depth_range: Tuple[float, float] = (200, 1500)) -> np.ndarray:
    """Quantizes depth levels into discrete grayscale image levels.

    Args:
        image (np.ndarray): Input image of shape (H, W).
        eta (int, optional): Number of gray levels. Defaults to 10.
        g_min (int, optional): Lowest gray level. Defaults to 155.
        g_max (int, optional): Highest gray level. Defaults to 255.

    Returns:
        np.ndarray: Depth quantized image.
    """
    
    if depth_range != None:
        near, far = depth_range
        mask = np.logical_and(image > near, image < far)
    else:
        mask = image > 0

    d_th = image.max()
    
    if np.any(mask):
        d_min = image[mask].min()
    else:
        d_min, d_th = 0, max(1, d_th)
     
    g_stride = int((g_max - g_min) / eta)
    return np.where(mask, g_min + np.round(eta * (image - d_min) / (d_th - d_min)) * g_stride, 0).astype(np.uint8)


def joint_shift_scale_rotate(joint_points: np.ndarray, shift_limit: float, scale_limit: float, rotate_limit: int, p: float = 0.5) -> np.ndarray:
    """Shift, scale, and rotate joint points within a specified range.

    Args:
        joint_points (np.ndarray): Joint points, of shape (num_frames, 44).
        shift_limit (float): How much the points can be shifted.
        scale_limit (float): Scale factor range.
        rotate_limit (int): Rotation range.
        p (float, optional): Probability of applying transform. Defaults to 0.5.

    Returns:
        np.ndarray: [description]
    """

    if np.random.random() >= p:
        return joint_points

    palm_idx = 1
    num_frames = joint_points.shape[0]
    

    if shift_limit:
        shift = np.random.uniform(-shift_limit, shift_limit, 2)    # shift = (shift_x, shift_y)
        joint_points = joint_points.reshape(num_frames, 22, -1)    # (num_frames, 44) -> (num_frames, 22, 2)
        joint_points = joint_points + shift                        # shift is broadcasted and added
        joint_points[0, palm_idx] = 0                              # joint points are supposed to be normalized relative to the palm point
        joint_points = joint_points.reshape(num_frames, -1)        # if palm point is non-zero, cannot apply scaling

    if scale_limit:
        scale_factor = 1 + np.random.uniform(-scale_limit, scale_limit)
        joint_points *= scale_factor

    if rotate_limit:
        rot_angle = np.random.randint(-rotate_limit, rotate_limit)
        joint_points = joint_points.reshape(num_frames, 22, -1)

        for i in range(num_frames):
            center = joint_points[i, palm_idx, :2]
            rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1)
            joint_points[i] = np.hstack([joint_points[i], np.ones((22, 1))]) @ rot_mat.T
        
        joint_points = joint_points.reshape(num_frames, -1)

    return joint_points


def image_shift_scale_rotate(image_sequence: np.ndarray, shift_limit: float, scale_limit:float, rotate_limit: int, p : float = 0.5):
    """Shift scale and rotate image within a certain limit."""

    transform = A.ReplayCompose([
        A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=p)
    ])

    data = transform(image=image_sequence[0])
    image_sequence[0] = data["image"]

    # Use same params for all frames
    for i in range(1, image_sequence.shape[0]):
        image_sequence[i] = A.ReplayCompose.replay(data['replay'], image=image_sequence[i])["image"]

    return image_sequence


def shift_scale_rotate(image_sequence: np.ndarray, joint_points: np.ndarray, shift_limit: float, scale_limit:float, rotate_limit: int, p : float = 0.5):
    transform = A.ReplayCompose([
        A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=p)
    ])

    data = transform(image=image_sequence[0])

    if not data['replay']['transforms'][0]['applied']:
        return image_sequence, joint_points

    image_sequence[0] = data["image"]
    for i in range(1, image_sequence.shape[0]):
        image_sequence[i] = A.ReplayCompose.replay(data['replay'], image=image_sequence[i])["image"]

    params = data['replay']['transforms'][0]['params']
    rot_angle = params['angle']
    scale_factor = params['scale']
    shift = np.array([params['dx'], params['dy']])
    
    palm_idx = 1
    num_frames = joint_points.shape[0]
    

    if np.any(shift):
        joint_points = joint_points.reshape(num_frames, 22, -1)    # (num_frames, 44) -> (num_frames, 22, 2)
        joint_points = joint_points + shift                        # shift is broadcasted and added
        joint_points[0, palm_idx] = 0                              # joint points are supposed to be normalized relative to the palm point
        joint_points = joint_points.reshape(num_frames, -1)        # if palm point is non-zero, cannot apply scaling

    if scale_factor:
        joint_points *= scale_factor

    if rot_angle:
        joint_points = joint_points.reshape(num_frames, 22, -1)
        for i in range(num_frames):
            center = joint_points[i, palm_idx, :2]
            rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1)
            joint_points[i] = np.hstack([joint_points[i], np.ones((22, 1))]) @ rot_mat.T
        
        joint_points = joint_points.reshape(num_frames, -1)

    return image_sequence, joint_points


def time_shift(image_sequence, joint_points, frame_limit, p):
    """shift frames by random frames."""

    if np.random.random() >= p:
        return image_sequence, joint_points
        
    shift = np.random.randint(-frame_limit, frame_limit)

    if shift < 0: # cut off some start frames
        image_sequence = image_sequence[-shift:]
        joint_points = joint_points[-shift:]

    elif shift > 0: # cut off some end frames
        image_sequence = image_sequence[:-shift]
        joint_points = joint_points[:-shift]
    
    return image_sequence, joint_points


def apply_augs(joint_points, image_sequence, augs):

    if "shift_scale_rotate" in augs:
        image_sequence, joint_points = shift_scale_rotate(image_sequence, joint_points, **augs["shift_scale_rotate"])
    else:
        if "joint_shift_scale_rotate" in augs:
            joint_points = joint_shift_scale_rotate(joint_points, **augs["joint_shift_scale_rotate"])

        if "image_shift_scale_rotate" in augs:
            image_sequence = image_shift_scale_rotate(image_sequence, **augs["image_shift_scale_rotate"])
    
    if "time_shift" in augs:
        image_sequence, joint_points = time_shift(image_sequence, joint_points, **augs["time_shift"])
    return joint_points, image_sequence