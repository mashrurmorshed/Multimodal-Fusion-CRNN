"""Crops hand region of interest (ROI) from video frames."""

from argparse import ArgumentParser
import numpy as np
from PIL import Image
import multiprocessing as mp
import os
import glob
import shutil
import time


def crop_and_save(i: int, data_dir: str, out_dir: str, roi: np.ndarray, mode: str) -> None:
    """Crops ROI from frame and saves it to some specified location.

    Args:
        i (int): Frame id.
        data_dir (str): Path to directory containing frames of a particular data item.
        out_dir (str): Path to output directory which will contain cropped frames.
        roi (np.ndarray): Region of interest, array of shape (4,) containing x, y, w, h.
        mode (str): One of 'shrec' or 'dhg'.
    """

    x, y, w, h = roi

    if mode == "shrec":
        file_name = f"{i}_depth.png"
    elif mode == "dhg":
        file_name = f"depth_{i+1}.png"

    image_path = os.path.join(data_dir, file_name)
    out_path = os.path.join(out_dir, file_name)

    image = np.array(Image.open(image_path))
    image = image[y: y + h, x: x + w]
    Image.fromarray(image).save(out_path)


def data_loc_generator(data_root: str, out_root: str, mode: str):
    """Creates subdirectories, transfers metadata and generates roi.

    Args:
        data_root (str): Base path containing full dataset.
        out_root (str): Output path for cropped dataset.
        mode (str): One of 'shrec' or 'dhg'.

    Yields:
        i (int): Frame id.
        data_dir (str): Path to directory containing frames of a particular data item.
        out_dir (str): Path to output directory which will contain cropped frames.
        (np.ndarray): Region of interest, array of shape (4,) containing x, y, w, h.
    """

    if mode == "shrec":
        gen_file = "general_informations.txt" 
        joint_2d_file = "skeletons_image.txt"
        joint_3d_file = "skeletons_world.txt"
    elif mode == "dhg":
        gen_file = "general_information.txt" 
        joint_2d_file = "skeleton_image.txt"
        joint_3d_file = "skeleton_world.txt"


    for data_dir in glob.glob(f"{data_root}/gesture_*/finger_*/subject_*/essai_*/"):
        gen_info_path = os.path.join(data_dir, gen_file)
        gen_info = np.loadtxt(gen_info_path, dtype=np.int32)[:, 1:]
        
        # create output directory
        out_dir = os.path.join(out_root, "/".join(data_dir.split("/")[-5:]))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        # copy meta information
        shutil.copy2(gen_info_path, out_dir)
        shutil.copy2(os.path.join(data_dir, joint_2d_file), out_dir)
        shutil.copy2(os.path.join(data_dir, joint_3d_file), out_dir)
        
        for i in range(gen_info.shape[0]):
            yield i, data_dir, out_dir, gen_info[i], mode


def main(args):
    if args.p:
        pool = mp.Pool(args.p)


    for func_args in data_loc_generator(args.i, args.o, args.mode):
        if args.p:
            pool.apply_async(crop_and_save, func_args)
        else:
            crop_and_save(*func_args)


    if args.p:
        pool.close()
        pool.join()

    # Copy .txt metadata (SHREC: train_gestures.txt, test_gestures.txt | DHG: informations_troncage_sequences.txt)
    for meta_file in glob.glob(os.path.join(args.i, "*.txt")):
        shutil.copy2(meta_file, args.o)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--i", type=str, required=True, help="Data input dir.")
    parser.add_argument("--o", type=str, required=True, help="Data output dir.")
    parser.add_argument("--p", type=int, default=0, help="Number of worker processes.")
    parser.add_argument("--mode", type=str, required=True, help="shrec or dhg.")
    args = parser.parse_args()


    start = time.time()
    main(args)
    print(f"Completed in {(time.time() - start):.2f} s.")