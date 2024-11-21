"""
This script performs image matching using a specified matcher model. It processes pairs of input images,
detects keypoints, matches them, and performs RANSAC to find inliers. The results, including visualizations
and metadata, are saved to the specified output directory.
"""

import sys
import torch
import argparse
import matplotlib
from pathlib import Path

from matching.utils import get_image_pairs_paths
from matching import get_matcher, available_models
from matching.viz import plot_matches

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")


def main(args):
    image_size = [args.im_size, args.im_size]
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Choose a matcher
    matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.n_kpts)

    if args.matcher == "mickey":
        print("Problem with loading intrinsics file: {args.path_intrinsics}")
        exit()
        matcher.resize = image_size
        matcher.path_intrinsics = args.path_intrinsics

    pairs_of_paths = get_image_pairs_paths(args.input)
    for i, (img0_path, img1_path) in enumerate(pairs_of_paths):
        image0 = matcher.load_image(img0_path, resize=image_size)
        image1 = matcher.load_image(img1_path, resize=image_size)

        result = matcher(image0, image1)

        out_str = f"Paths: {str(img0_path), str(img1_path)}. Found {result['num_inliers']} inliers after RANSAC. "

        if not args.no_viz:
            viz_path = args.out_dir / f"output_{i}_matches.jpg"
            plot_matches(image0, image1, result, save_path=viz_path)
            out_str += f"Viz saved in {viz_path}. "

        result["img0_path"] = img0_path
        result["img1_path"] = img1_path
        result["matcher"] = args.matcher
        result["n_kpts"] = args.n_kpts
        result["im_size"] = args.im_size

        dict_path = args.out_dir / f"output_{i}_result.torch"
        torch.save(result, dict_path)
        out_str += f"Output saved in {dict_path}"

        print(out_str)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Matching Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )

    # Hyperparameters shared by all methods:
    parser.add_argument("--im_size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--n_kpts", type=int, default=2048, help="max num keypoints")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--no_viz", action="store_true", help="avoid saving visualizations")

    parser.add_argument(
        "--input",
        type=str,
        default="assets/example_pairs",
        help="path to either (1) dir with dirs with image pairs or (2) txt file with two image paths per line",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="path where outputs are saved")

    # Path intrinsics for Mickey matcher (if not provided, we use defaults)
    parser.add_argument(
        "--path_intrinsics", type=str, default=None, help="path to intrinsics"
    )

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"outputs_{args.matcher}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
