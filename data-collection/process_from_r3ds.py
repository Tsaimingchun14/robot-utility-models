"""
This task runner assumes that we have the r3d files as zip files in the folder, with the
general structure of task_name/home_id/env_id/timestamp.zip

We will unzip the files, and then process them one by one.
"""

import argparse
import json
import logging
import os
import pickle as pkl
import shutil
import traceback
import zipfile
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple, cast

import cv2
import liblzfse
import numpy as np
import PIL
import torch
from quaternion import as_rotation_matrix, quaternion
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension
from tqdm import tqdm

from utils.action_transforms import apply_permutation_transform
from utils.error_handlers import CustomFormatter, send_slack_message
from utils.models import GripperNet
from utils.new_gripper_model import (
    NewGripperModel,
    extract_gripper_value_from_pretrained_model_and_frames,
    IMAGE_SIZE,
    NORMALIZER,
)
from utils.aruco_gripper import detect_gripper_width_aruco

torch.set_num_threads(1)
COMPLETION_FILENAME = "completed.txt"
ABANDONED_FILENAME = "abandoned.txt"
MIN_CALIB_FRAMES = 300  # ~10 seconds at 30 Hz
MAX_UNDETECTED_RUN = 10
logger = logging.getLogger("R3D Processing")
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class LabelFreeImageFolder(ImageFolder):
    def find_classes(self, directory: str):
        return [], {}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int] = dict(),
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []

        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, 0  # The images have no classes.
                    instances.append(item)

        return instances


def label_image_folder(
    image_folder,
    labelling_model_path="./gripper_model.pth",
    device="cpu",
    batch_size=64,
):
    # We will use the labelling model to label the images in the image folder.
    # We will then save the labels in the same folder as the images.
    model = GripperNet()
    model.to(device)
    model.load_state_dict(torch.load(labelling_model_path, map_location=device))
    model.eval()

    dataset = LabelFreeImageFolder(
        image_folder,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    with torch.no_grad():
        for X, _ in dataloader:
            batch = X.to(device)
            output = model(batch)
            labels.append(output.cpu().numpy())

    labels = np.concatenate(labels, axis=0)
    return labels


def label_image_folder_new(
    image_folder,
    labelling_model_path="./gripper_model_new.pth",
    device="cpu",
    batch_size=64,
):
    model = NewGripperModel()
    model.to(device)
    model.load_state_dict(torch.load(labelling_model_path, map_location=device))
    model.eval()

    dataset = LabelFreeImageFolder(
        image_folder,
        transform=transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                NORMALIZER,
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    with torch.no_grad():
        for X, _ in dataloader:
            batch = X.to(device)
            output = extract_gripper_value_from_pretrained_model_and_frames(
                model, batch
            )
            labels.append(output.cpu().numpy())
    labels = np.concatenate(labels, axis=0)
    return labels


def detect_aruco_distances(image_folder, report_path=None):
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
    dists = []
    detections = []

    for f in files:
        img_path = os.path.join(image_folder, f)
        img = cv2.imread(img_path)
        if img is None:
            dists.append(None)
            detections.append(False)
            continue

        dist = detect_gripper_width_aruco(img)
        dists.append(dist)
        detections.append(dist is not None)

    if report_path is not None:
        import csv

        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "filename", "detected", "dist_px"])
            for idx, (fname, detected, dist) in enumerate(
                zip(files, detections, dists)
            ):
                writer.writerow(
                    [idx, fname, int(detected), "" if dist is None else f"{dist:.6f}"]
                )

    return files, dists


def label_image_folder_aruco(image_folder, min_dist, max_dist, report_path=None):
    """
    Detects gripper width via ArUco markers and interpolates missing frames.
    """
    files, dists = detect_aruco_distances(image_folder, report_path=report_path)

    # Fail fast on long undetected runs
    run = 0
    max_run = 0
    for d in dists:
        if d is None:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    if max_run >= MAX_UNDETECTED_RUN:
        raise ValueError(
            f"Found {max_run} consecutive undetected frames (>= {MAX_UNDETECTED_RUN}) in {image_folder}"
        )

    # Convert to numeric array, replacing None with NaN for interpolation
    dists = np.array([d if d is not None else np.nan for d in dists])
    if np.all(np.isnan(dists)):
        raise ValueError(f"No ArUco markers found in entire folder: {image_folder}")

    # Interpolate missing values
    nans = np.isnan(dists)
    if np.any(nans):
        indices = np.arange(len(dists))
        dists[nans] = np.interp(indices[nans], indices[~nans], dists[~nans])

    # Normalize
    normalized = (dists - min_dist) / (max_dist - min_dist)
    return np.clip(normalized, 0, 1)


class R3DZipFileProcessor:
    # TODO: rotate the images and label them with the model.
    def __init__(
        self,
        path,
        model_path,
        device="cpu",
        use_aruco=False,
        aruco_min=None,
        aruco_max=None,
        aruco_report=False,
    ):
        self.path = path
        assert self.path.endswith(".zip")
        self._extracted_path = self.path[:-4]
        self.model_path = model_path
        self.device = device
        self.use_aruco = use_aruco
        self.aruco_min = aruco_min
        self.aruco_max = aruco_max
        self.aruco_report = aruco_report

        # Keep a cache of the last rotated images because sometimes the RGB file coming out
        # of the iphone/R3D app is corrputed.
        self._last_rotated_image = None

    def process(self):
        try:
            return self._process()
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"Error processing {self.path}: {e}")
            logger.error(error_message)
            # send_slack_message(f"Error processing {self.path}: {e}")
            # send_slack_message(error_message)
            return self._process(redo_everything=True)

    def _process(self, redo_everything=False):
        try:
            self.process_metadata()
        except zipfile.BadZipFile as e:
            logger.error(f"Error extracting metadata from {self.path}: {e}")
            with open(os.path.join(self._extracted_path, ABANDONED_FILENAME), "w") as f:
                f.write("Abandoned\n")
                f.write(str(e))
            return False
        self.extract_images(redo_everything=redo_everything)
        transforms = self.process_poses()
        gripper_labels = self.process_gripper_positions(
            os.path.join(self._extracted_path, "images")
        )
        assert self.validate()
        self.save_transforms(transforms, gripper_labels)
        with open(os.path.join(self._extracted_path, COMPLETION_FILENAME), "w") as f:
            f.write("Completed")
        return True

    def process_metadata(self):
        # Extract and process the "metadata" file from the r3d export.
        # This file contains the camera intrinsics, the camera extrinsics, and the
        # camera transforms.
        logger.info("Processing metadata")
        os.makedirs(self._extracted_path, exist_ok=True)
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extract("metadata", self._extracted_path)

        with open(os.path.join(self._extracted_path, "metadata"), "r") as f:
            self._metadata_dict = json.load(f)
        metadata_dict = self._metadata_dict

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.aspect_ratio = self.rgb_width / self.rgb_height
        self.image_size = (self.rgb_width, self.rgb_height)

        self.poses = np.array(metadata_dict["poses"])
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.fps = metadata_dict["fps"]

        self.total_images = len(self.poses)
        self.init_pose = np.array(metadata_dict["initPose"])

    @staticmethod
    def _process_filename(filename):
        if filename == "rgbd":
            return filename
        name, extension = filename.split(".")
        assert extension in ["jpg", "depth", "conf"]
        return f"rgbd/{int(name)}.{extension}"

    def extract_images(self, redo_everything=False):
        # First, we create the proper directory structure.
        # We assume that the zip files are in the format of task/home/env/timestamp.zip
        # We will unzip them to task/home/env/timestamp/
        logger.info("Extracting images")

        with zipfile.ZipFile(self.path, "r") as zip_ref:
            all_files = zip_ref.namelist()

        rgb_files = {f for f in all_files if f.endswith(".jpg")}
        depth_files = {f for f in all_files if f.endswith(".depth")}
        conf_files = {f for f in all_files if f.endswith(".conf")}

        image_folder_name = "images" if self.aspect_ratio > 1 else "unrotated_images"
        rgbfolder = os.path.join(self._extracted_path, image_folder_name)
        # Process the RGB images.
        os.makedirs(rgbfolder, exist_ok=True)
        # Now, remove the files that are already extracted and therefore should not be extracted again.
        to_extract = (
            rgb_files
            - {R3DZipFileProcessor._process_filename(x) for x in os.listdir(rgbfolder)}
            if not redo_everything
            else rgb_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(rgbfolder, members=list(to_extract))
        # TODO: Figure out how to rename in an idempotent way.
        R3DZipFileProcessor._rename_to_sequential(rgbfolder, extension=".jpg")
        # At the same time, rotate the images.
        if self.aspect_ratio > 1:
            self.compress_images(rgbfolder, redo_everything=redo_everything)
        else:
            self.rotate_and_compress_images(rgbfolder, redo_everything=redo_everything)

        # Process the depth images.
        depthfolder = os.path.join(self._extracted_path, "compressed_depths")
        os.makedirs(depthfolder, exist_ok=True)
        to_extract = (
            depth_files
            - {
                R3DZipFileProcessor._process_filename(x)
                for x in os.listdir(depthfolder)
            }
            if not redo_everything
            else depth_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(depthfolder, members=list(to_extract))
        R3DZipFileProcessor._rename_to_sequential(depthfolder, extension=".depth")

        # Process the conf images.
        conffolder = os.path.join(self._extracted_path, "compressed_confs")
        os.makedirs(conffolder, exist_ok=True)
        to_extract = (
            conf_files
            - {R3DZipFileProcessor._process_filename(x) for x in os.listdir(conffolder)}
            if not redo_everything
            else conf_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(conffolder, members=conf_files)
        R3DZipFileProcessor._rename_to_sequential(conffolder, extension=".conf")
        return rgbfolder, depthfolder, conffolder

    def compress_images(self, rgb_path, redo_everything=False):
        logger.info("Compressing images")
        compressed_path = os.path.join(self._extracted_path, "compressed_images")
        os.makedirs(compressed_path, exist_ok=True)
        for f in sorted(os.listdir(rgb_path)):
            if os.path.exists(os.path.join(compressed_path, f)) and not redo_everything:
                continue
            image_path = os.path.join(rgb_path, f)
            img = cv2.imread(image_path)

            # Compress the image.
            compressed_img = cv2.resize(img, (256, 256))
            cv2.imwrite(os.path.join(compressed_path, f), compressed_img)

    def rotate_and_compress_images(self, rgb_path, redo_everything=False):
         # Rotate the images by 90 degrees, since the iphone captures images in portrait mode.
         # We do it this way to make sure the operation is idempotent, since rotating an image
         # 90 degrees and replacing the original one is not.
         logger.info("Rotating images")
         rotated_path = os.path.join(self._extracted_path, "images")
         compressed_path = os.path.join(self._extracted_path, "compressed_images")
         os.makedirs(rotated_path, exist_ok=True)
         os.makedirs(compressed_path, exist_ok=True)
         for f in sorted(os.listdir(rgb_path)):
             if os.path.exists(os.path.join(rotated_path, f)) and not redo_everything:
                 continue
             try:
                 image_path = os.path.join(rgb_path, f)
                 _ = PIL.Image.open(image_path)
                 img = cv2.imread(image_path)
                 img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                 self._last_rotated_image = img
             except PIL.UnidentifiedImageError as e:
                 logger.error(f"Error rotating {f}: {e}")
                 if self._last_rotated_image is not None:
                     img = self._last_rotated_image
                     self._last_rotated_image = None
                 else:
                     raise e

             cv2.imwrite(os.path.join(rotated_path, f), img)
             # Now compress the image.
             compressed_img = cv2.resize(img, (256, 256))
             cv2.imwrite(os.path.join(compressed_path, f), compressed_img)

    def process_poses(self):
        # Process the poses from the metadata file.
        # We will convert the poses to a list of rotation matrices and translation vectors.
        logger.info("Processing poses")
        self.quaternions = []
        self.translation_vectors = []
        init_pose = None
        for pose in self.poses:
            qx, qy, qz, qw, px, py, pz = pose
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
            extrinsic_matrix[:3, -1] = [px, py, pz]

            if init_pose is None:
                init_pose = np.copy(extrinsic_matrix)

            # We will convert the extrinsic matrix to the camera pose.
            # The camera pose is the inverse of the extrinsic matrix.
            relative_pose = np.linalg.inv(init_pose) @ extrinsic_matrix
            transformed_pose = apply_permutation_transform(relative_pose, self.aspect_ratio)
            self.translation_vectors.append(transformed_pose[:3, -1])
            self.quaternions.append(R.from_matrix(transformed_pose[:3, :3]).as_quat())
        quats = np.array(self.quaternions)
        translations = np.array(self.translation_vectors)
        transforms = np.concatenate([translations, quats], axis=1)
        return transforms

    def process_gripper_positions(self, rgb_folder):
        logger.info("Processing gripper positions")
        if self.use_aruco:
            report_path = None
            if self.aruco_report:
                report_path = os.path.join(
                    self._extracted_path, "aruco_detection.csv"
                )
            gripper_labels = label_image_folder_aruco(
                rgb_folder, self.aruco_min, self.aruco_max, report_path=report_path
            )
        else:
            gripper_labels = label_image_folder_new(
                rgb_folder, self.model_path, device=self.device
            )
        return gripper_labels

    @staticmethod
    def _rename_to_sequential(path, extension=".jpg"):
        filenames, file_indices = [], []
        if os.path.exists(os.path.join(path, "rgbd")):
            base_path = os.path.join(path, "rgbd")
        else:
            base_path = path
        for i, f in enumerate(sorted(os.listdir(base_path))):
            filenames.append(f)
            index, ext = f.split(".")
            assert ext == extension[1:]
            file_index = int(index)
            assert file_index >= 0
            file_indices.append(file_index)

        for f, i in zip(filenames, file_indices):
            new_fname_short = f"{i:04d}{extension}"
            new_fname_long = f"{i:06d}{extension}"
            if len(filenames) <= 10_000:
                new_fname = new_fname_short
            else:
                new_fname = new_fname_long
            os.rename(os.path.join(base_path, f), os.path.join(path, new_fname))
            if len(filenames) > 10_000:
                if len(new_fname_short) != len(new_fname_long):
                    # Make sure only one of those exists.
                    assert not (
                        os.path.exists(os.path.join(base_path, new_fname_short))
                        and os.path.exists(os.path.join(base_path, new_fname_long))
                    )

        if os.path.exists(os.path.join(path, "rgbd")):
            assert len(os.listdir(base_path)) == 0
            shutil.rmtree(base_path)

    def validate(self):
        logger.info("Validating the extracted data")
        # TODO: Add validation functions on the images and the poses.
        return True

    def save_transforms(self, transforms, gripper_labels):
        logger.info("Saving the extracted data")
        translations, rotations = transforms[:, :3], transforms[:, 3:]
        new_data = {
            i: {
                "xyz": xyz.tolist(),
                "quats": quats.tolist(),
                "gripper": grp.tolist(),
            }
            for i, (xyz, quats, grp) in enumerate(
                zip(translations, rotations, gripper_labels)
            )
        }
        with open(os.path.join(self._extracted_path, "labels.json"), "w") as f:
            json.dump(new_data, f)

        with open(os.path.join(self._extracted_path, "relative_poses.pkl"), "wb") as f:
            pkl.dump(transforms, f)


def filter_r3d_files_to_process(r3d_paths_file):
    with open(r3d_paths_file, "r") as f:
        r3d_paths = json.load(f)

    to_process = []
    # We will filter out the ones that have already been processed.
    for path in r3d_paths:
        assert path.endswith(".zip")
        completed = os.path.exists(os.path.join(path[:-4], COMPLETION_FILENAME))
        abandoned = os.path.exists(os.path.join(path[:-4], ABANDONED_FILENAME))
        if os.path.exists(path[:-4]) and (completed or abandoned):
            continue
        to_process.append(path)

    return to_process


def filter_r3d_files_completed(r3d_paths_file):
    with open(r3d_paths_file, "r") as f:
        r3d_paths = json.load(f)
    completed = []
    for path in r3d_paths:
        if not path.endswith(".zip"):
            continue
        if os.path.exists(os.path.join(path[:-4], COMPLETION_FILENAME)):
            completed.append(path)
    return completed




def _read_num_frames_from_zip(zip_path):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("metadata") as f:
                metadata_dict = json.load(f)
        return len(metadata_dict.get("poses", []))
    except Exception as e:
        logger.warning(f"Failed to read metadata from {zip_path}: {e}")
        return 0


def _get_env_root(zip_path):
    # Expected: task/home/env/timestamp.zip
    return os.path.dirname(zip_path)


def _cleanup_extracted_path(path_to_remove):
    try:
        if os.path.exists(path_to_remove):
            shutil.rmtree(path_to_remove)
    except Exception as e:
        logger.warning(f"Failed to cleanup artifacts at {path_to_remove}: {e}")


def _compute_env_calibration(
    env_zip_paths,
    model_path,
    aruco_report,
    fallback_min=None,
    fallback_max=None,
):
    # Choose calibration zip: first episode only (must be >= MIN_CALIB_FRAMES)
    calib_zips = []
    if not env_zip_paths:
        return fallback_min, fallback_max, calib_zips

    first_zip = env_zip_paths[0]
    calib_zips.append(first_zip)
    first_frames = _read_num_frames_from_zip(first_zip)
    if first_frames < MIN_CALIB_FRAMES:
        logger.error(
            f"Calibration failed for env: first clip has {first_frames} frames (< {MIN_CALIB_FRAMES})."
        )
        return None, None, calib_zips

    # Compute distances from calibration zips
    all_dists = []
    for zip_path in calib_zips:
        processor = R3DZipFileProcessor(
            zip_path,
            model_path,
            use_aruco=True,
            aruco_min=fallback_min,
            aruco_max=fallback_max,
            aruco_report=aruco_report,
        )
        try:
            processor.process_metadata()
            processor.extract_images()
            rgb_folder = os.path.join(processor._extracted_path, "images")
            if not os.path.exists(rgb_folder):
                rgb_folder = os.path.join(processor._extracted_path, "unrotated_images")
            report_path = None
            if aruco_report:
                report_path = os.path.join(processor._extracted_path, "aruco_detection.csv")
            _, dists = detect_aruco_distances(rgb_folder, report_path=report_path)
            all_dists.extend([d for d in dists if d is not None])
        except Exception as e:
            logger.warning(f"Calibration failed for {zip_path}: {e}")
        finally:
            _cleanup_extracted_path(processor._extracted_path)

    if not all_dists:
        logger.error(
            "No ArUco detections found in calibration clip; failing env calibration."
        )
        return None, None, calib_zips

    calib_min = float(min(all_dists))
    calib_max = float(max(all_dists))

    return calib_min, calib_max, calib_zips


def process_r3d_file(
    file_path,
    model_path,
    use_aruco=False,
    aruco_min=None,
    aruco_max=None,
    aruco_report=False,
):
    logger.info(f"Processing {file_path}")
    processor = R3DZipFileProcessor(
        file_path,
        model_path,
        use_aruco=use_aruco,
        aruco_min=aruco_min,
        aruco_max=aruco_max,
        aruco_report=aruco_report,
    )
    try:
        processor.process()
        logger.info(f"Finished processing {file_path}")
        return True
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(error_message)
        raise


# Define argparse arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--r3d_paths_file",
    type=str,
    required=True,
    help="Path to the file containing the paths to the r3d files.",
)
parser.add_argument(
    "--count_only",
    action="store_true",
    help="If set, will only count the number of r3d files to process.",
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the model used to detect the gripper.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=16,
    help="Number of workers to use to process the r3d files.",
)
parser.add_argument(
    "--start_index",
    type=int,
    default=0,
    help="Index to start processing the r3d files from.",
)
parser.add_argument(
    "--end_index",
    type=int,
    default=-1,
    help="Index to end processing the r3d files at.",
)
parser.add_argument(
    "--use_aruco",
    action="store_true",
    help="If set, will use ArUco markers instead of deep learning model.",
)
parser.add_argument(
    "--aruco_report_path",
    type=str,
    default="",
    help="If set, write per-frame ArUco detection CSV per zip and aggregate to this path.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    r3d_paths_file = args.r3d_paths_file
    model_path = args.model_path
    num_workers = args.num_workers
    start_index = args.start_index
    end_index = args.end_index

    # Filter out the r3d files that have already been processed.
    r3d_paths = filter_r3d_files_to_process(r3d_paths_file)
    if end_index == -1:
        end_index = len(r3d_paths)
    logger.info(f"Number of r3d files to process: {len(r3d_paths)}")
    if args.count_only:
        exit()
    r3d_paths = r3d_paths[start_index:end_index]

    # Process the r3d files.
    if args.use_aruco:
        # Group by env and calibrate per env using the first episode (must be >= MIN_CALIB_FRAMES)
        env_groups = {}
        for p in r3d_paths:
            env_root = _get_env_root(p)
            env_groups.setdefault(env_root, []).append(p)
        for env_root in env_groups:
            env_groups[env_root] = sorted(env_groups[env_root])

        valid_envs = []
        failed_envs = []

        for env_root, env_paths in tqdm(env_groups.items(), desc="Processing envs"):
            calib_min, calib_max, calib_zips = _compute_env_calibration(
                env_paths,
                model_path=model_path,
                aruco_report=bool(args.aruco_report_path),
                fallback_min=None,
                fallback_max=None,
            )
            if calib_min is None or calib_max is None:
                logger.error(f"Env {env_root}: calibration failed. Skipping env.")
                failed_envs.append(env_root)
                # remove any extracted artifacts for this env
                _cleanup_extracted_path(env_root)
                continue
            logger.info(
                f"Env {env_root}: using ArUco min={calib_min:.3f} max={calib_max:.3f} from {len(calib_zips)} calibration clip(s)"
            )
            env_ok = True
            calib_set = set(calib_zips)
            for p in tqdm(env_paths, desc=f"Processing {env_root}"):
                if p in calib_set:
                    logger.info(f"Env {env_root}: skipping calibration zip {p}")
                    continue
                try:
                    process_r3d_file(
                        p,
                        model_path=model_path,
                        use_aruco=args.use_aruco,
                        aruco_min=calib_min,
                        aruco_max=calib_max,
                        aruco_report=bool(args.aruco_report_path),
                    )
                except Exception as e:
                    msg = str(e)
                    if "consecutive undetected frames" in msg or "No ArUco markers found" in msg:
                        logger.error(
                            f"Env {env_root}: skipping episode {p} due to ArUco detection failure: {e}"
                        )
                        _cleanup_extracted_path(p[:-4])
                        env_ok = False
                        continue
                    logger.error(
                        f"Env {env_root}: error while processing {p}: {e}. Skipping remaining episodes in env."
                    )
                    _cleanup_extracted_path(p[:-4])
                    env_ok = False
                    break
            if env_ok:
                valid_envs.append(env_root)
            else:
                failed_envs.append(env_root)
    else:
        with Pool(num_workers) as p:
            p.map(
                partial(
                    process_r3d_file,
                    model_path=model_path,
                    use_aruco=args.use_aruco,
                    aruco_min=None,
                    aruco_max=None,
                    aruco_report=bool(args.aruco_report_path),
                ),
                tqdm(r3d_paths, desc="Processing r3d files"),
            )

    if args.use_aruco:
        logger.info(f"Valid envs ({len(valid_envs)}):")
        for env in valid_envs:
            logger.info(f"  {env}")
        if failed_envs:
            logger.info(f"Failed envs ({len(failed_envs)}):")
            for env in failed_envs:
                logger.info(f"  {env}")

    if args.use_aruco and args.aruco_report_path:
        import csv

        report_dir = os.path.dirname(args.aruco_report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.aruco_report_path, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(
                ["zip_path", "frame_idx", "filename", "detected", "dist_px"]
            )
            for zip_path in r3d_paths:
                report_path = os.path.join(zip_path[:-4], "aruco_detection.csv")
                if not os.path.exists(report_path):
                    continue
                with open(report_path, "r", newline="") as f_in:
                    reader = csv.reader(f_in)
                    header = next(reader, None)
                    if header is None:
                        continue
                    for row in reader:
                        if len(row) < 4:
                            continue
                        writer.writerow([zip_path, row[0], row[1], row[2], row[3]])

    # Update r3d_paths_file to only include completed episodes
    try:
        completed_paths = filter_r3d_files_completed(r3d_paths_file)
        with open(r3d_paths_file, "w") as f:
            json.dump(completed_paths, f, indent=4)
        logger.info(f"Wrote {len(completed_paths)} completed paths to {r3d_paths_file}")
    except Exception as e:
        logger.warning(f"Failed to update r3d_paths_file: {e}")

    logger.info("Finished processing all r3d files.")
