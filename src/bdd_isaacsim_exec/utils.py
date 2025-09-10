# SPDX-License-Identifier:  GPL-3.0-or-later
import os
import cv2
import imageio
from typing import Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdflib.namespace import NamespaceManager
from rdf_utils.naming import get_valid_var_name
from rdf_utils.models.python import URI_PY_TYPE_MODULE_ATTR, import_attr_from_model
from bdd_dsl.models.urirefs import (
    URI_SIM_PRED_HAS_CONFIG,
    URI_SIM_PRED_PATH,
    URI_SIM_TYPE_RES_PATH,
    URI_SIM_TYPE_SYS_RES,
)
from bdd_dsl.utils.common import check_or_convert_ndarray
from bdd_dsl.models.environment import ObjectModel
from bdd_dsl.models.agent import AgentModel
from bdd_isaacsim_exec.uri import URI_SIM_TYPE_ISAAC_RES, URI_TYPE_USD_FILE

from omni.isaac.core.scenes.scene import Scene as IsaacScene
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.sensor import Camera


_CACHED_ASSET_ROOT = None
_CACHED_ID_STRS = set()
OBJ_POSITION_LOWER_BOUNDS_D = [0.35, -0.3, 0.15]
OBJ_POSITION_UPPER_BOUNDS_D = [0.6, 0.3, 0.2]
OBJ_POSITION_LOWER_BOUNDS_N = [0.25, -0.4, 0.15]
OBJ_POSITION_UPPER_BOUNDS_N = [0.6, 0.4, 0.2]


def _get_unique_id_str(id_str: str, max_iteration: int = 100) -> str:
    iter_count = 0
    while True:
        if iter_count > max_iteration:
            raise RuntimeError(f"unable to find unique string after {max_iteration} iterations")
        iter_count += 1

        unique_str = id_str + f"{np.random.randint(65536):04x}"
        if unique_str in _CACHED_ID_STRS:
            continue
        _CACHED_ID_STRS.add(unique_str)

        return unique_str


def get_cached_assets_root_path() -> str:
    """Get Isaacsim assets root path and cache.

    Raise exception if asset root directory can't be found.
    Raises:
        RuntimeError: if asset root directory can't befound
    Returns:
        str: Root directory containing assets from Isaac Sim
    """
    global _CACHED_ASSET_ROOT
    if _CACHED_ASSET_ROOT is not None:
        return _CACHED_ASSET_ROOT

    # These imports are assumed to be called after SimulationApp() call,
    # otherwise my cause import errors
    from omni.isaac.core.utils.nucleus import get_assets_root_path

    _CACHED_ASSET_ROOT = get_assets_root_path()
    if _CACHED_ASSET_ROOT is not None:
        return _CACHED_ASSET_ROOT

    raise RuntimeError("Could not find Isaac Sim assets folder")


def create_rigid_prim_in_scene(
    scene: IsaacScene,
    ns_manager: NamespaceManager,
    model: Union[ObjectModel, AgentModel],
    prim_prefix: str,
) -> RigidPrim:
    id_str = model.id.n3(namespace_manager=ns_manager)
    id_str = get_valid_var_name(id_str)

    # TODO(minhnh): handle initial poses

    prim_configs = {}
    model_configs = model.get_attr(key=URI_SIM_PRED_HAS_CONFIG)
    assert model_configs is not None and isinstance(
        model_configs, dict
    ), f"no configs for {model.id}"
    prim_configs |= model_configs

    if "scale" in prim_configs:
        prim_configs["scale"] = check_or_convert_ndarray(prim_configs["scale"]) / get_stage_units()
    if "color" in prim_configs:
        prim_configs["color"] = check_or_convert_ndarray(prim_configs["color"]) / get_stage_units()

    if "position" not in prim_configs:
        obj_position = np.random.uniform(OBJ_POSITION_LOWER_BOUNDS_N, OBJ_POSITION_UPPER_BOUNDS_N)
        prim_configs["position"] = obj_position / get_stage_units()

    unique_id_str = _get_unique_id_str(id_str=id_str)
    prim_path = find_unique_string_name(
        initial_name=prim_prefix + unique_id_str,
        is_unique_fn=lambda x: not is_prim_path_valid(x),
    )
    obj_name = find_unique_string_name(
        initial_name=unique_id_str,
        is_unique_fn=lambda x: not scene.object_exists(x),
    )

    if URI_TYPE_USD_FILE in model.model_types:
        assert (
            URI_SIM_TYPE_RES_PATH in model.model_types
        ), f"object '{model.id}' has type '{URI_TYPE_USD_FILE}' but not type '{URI_SIM_TYPE_RES_PATH}'"

        asset_path = None
        for path_model_id in model.model_type_to_id[URI_SIM_TYPE_RES_PATH]:
            asset_path = model.models[path_model_id].get_attr(key=URI_SIM_PRED_PATH)
            if asset_path is not None:
                break
        assert (
            asset_path is not None
        ), f"attr '{URI_SIM_PRED_PATH}' not loaded for object model '{model.id}'"

        usd_model_uris = model.model_type_to_id[URI_TYPE_USD_FILE]

        if URI_SIM_TYPE_ISAAC_RES in model.model_types:
            asset_path = get_cached_assets_root_path() + asset_path

        elif URI_SIM_TYPE_SYS_RES in model.model_types:
            assert os.path.exists(
                asset_path
            ), f"Path in USD model(s) '{usd_model_uris}' does not exists: {asset_path}"

        else:
            raise RuntimeError(
                f"unhandled types for USD model(s) '{usd_model_uris}': {model.model_types}"
            )

        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        return scene.add(RigidPrim(prim_path=prim_path, name=obj_name, **prim_configs))

    if URI_PY_TYPE_MODULE_ATTR in model.model_types:
        correct_cls = None
        for model_id in model.model_type_to_id[URI_PY_TYPE_MODULE_ATTR]:
            python_cls = import_attr_from_model(model=model.models[model_id])
            if issubclass(python_cls, RigidPrim) or issubclass(python_cls, Articulation):
                correct_cls = python_cls
                break
        assert (
            correct_cls is not None
        ), f"'{model.id}' has no handled Python class model: {model.models.keys()}"

        return scene.add(correct_cls(name=obj_name, prim_path=prim_path, **prim_configs))

    raise RuntimeError(f"unhandled types for object'{model.id}': {model.model_types}")


def setup_camera_in_scene(
    name: str, position: np.ndarray, orientation: np.ndarray, resolution: tuple, fps: float
) -> Camera:
    """Setup camera in scene.

    Args:
        name (str): Name of the camera
        position (np.ndarray): Position of the camera
        rotation (np.ndarray): Rotation of the camera
        resolution (tuple): Resolution of the camera

    Returns:
        Camera: Camera object
    """
    camera = Camera(
        prim_path=f"/World/Cameras/{name}",
        name=name,
        position=position,
        orientation=orientation,
        frequency=fps,
        resolution=resolution,
    )
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    return camera


def capture_camera_image(camera: Camera) -> np.ndarray:
    """Capture camera image.

    Args:
        camera (Camera): Camera object

    Returns:
        np.ndarray: Captured image as a numpy array
    """
    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError(f"Camera {camera.name} did not return an image")

    if len(rgba.shape) < 3:
        # invalid image returned at sim startup
        raise RuntimeError(f"Camera {camera.name} did not return a valid image, shape={rgba.shape}")

    # Convert RGBA to RGB
    rgb_image = rgba[:, :, :3]
    return rgb_image


def save_single_frame(frame: np.ndarray, filename: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    frame_path = os.path.join(output_dir, filename)
    imageio.imwrite(frame_path, frame)


def save_frames(frames: list, output_dir: str, threads: int = 8):
    """Save frames to disk.

    Args:
        frames (list): List of frames to save
        capture_root_path (str): Root path for saving captures
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, frame in enumerate(frames):
            fn = f"frame_{i:05d}.jpg"
            executor.submit(save_single_frame, frame, fn, output_dir)


def create_video_from_frames(
    capture_root_path: str,
    scenario_name: str,
    frame_rate: int,
    camera_name: str,
    cleanup_frames: bool = False,
):
    """
    Creates a video from a sequence of image frames stored in a directory.
    Args:
        output_dir (str): The directory containing the image frames.
                          Only files with a ".jpg" extension will be used.
        video_path (str): The path where the output video file will be saved.
        frame_rate (int, optional): The frame rate of the output video. Defaults to 20.
    Returns:
        str: The path to the created video file, or None if no frames were found.
    Notes:
        - The image frames are expected to have the same dimensions.
        - The video will be encoded in MP4 format using the 'mp4v' codec.
    """
    frames_dir = os.path.join(capture_root_path, "frames", camera_name)
    vid_dir = os.path.join(capture_root_path, "vids")
    os.makedirs(vid_dir, exist_ok=True)
    video_path = os.path.join(vid_dir, f"{get_valid_var_name(scenario_name)}-{camera_name}.mp4")
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    frame_files.sort()

    if not frame_files:
        raise ValueError(f"No frames found in directory: {frames_dir}")

    # Read the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        video_writer.write(frame)

    video_writer.release()

    # Remove frames after video creation
    if cleanup_frames:
        print(f"Removing frames from {frames_dir} after video creation.")
        for frame_file in frame_files:
            os.remove(os.path.join(frames_dir, frame_file))

    return video_path
