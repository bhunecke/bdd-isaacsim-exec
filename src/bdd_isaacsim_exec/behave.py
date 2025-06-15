# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any, Literal
import time
import numpy as np
import os
from behave import use_fixture
from behave.model import Scenario
from behave.runner import Context
from rdflib import Graph, URIRef
from rdf_utils.uri import try_expand_curie
from rdf_utils.models.python import load_py_module_attr
from rdf_utils.naming import get_valid_var_name
from bdd_dsl.behave import (
    PARAM_AGN,
    PARAM_EVT,
    PARAM_FROM_EVT,
    PARAM_UNTIL_EVT,
    PARAM_OBJ,
    PARAM_WS,
    ParamType,
    load_obj_models_from_table,
    load_agn_models_from_table,
    load_str_params,
    load_ws_models_from_table,
    parse_str_param,
)
from bdd_dsl.execution.common import Behaviour, ExecutionModel
from bdd_dsl.models.urirefs import URI_SIM_PRED_HAS_CONFIG
from bdd_dsl.models.user_story import UserStoryLoader
from bdd_dsl.simulation.common import load_attr_path
from bdd_isaacsim_exec.uri import URI_FRANKA_PANDA, URI_UR_UR10


DIST_THRESHOLD = 0.01
UR10_MAX_EE_SPEED_MEAN = 0.75226
UR10_MAX_EE_SPEED_STD = 0.19432
UR10_SPEED_THRESHOLD = UR10_MAX_EE_SPEED_MEAN + 2 * UR10_MAX_EE_SPEED_STD
PANDA_MAX_EE_SPEED_MEAN = 0.50051
PANDA_MAX_EE_SPEED_STD = 0.22451
PANDA_SPEED_THRESHOLD = PANDA_MAX_EE_SPEED_MEAN + 2 * PANDA_MAX_EE_SPEED_STD
SPEED_THRESHOLD = 1.1


def isaacsim_fixture(context: Context, **kwargs: Any):
    from isaacsim import SimulationApp

    unit_length = kwargs.get("unit_length", 1.0)
    render_type = context.render_type
    if render_type == "headless":
        context.headless = True
        config = { "headless": True }
    elif render_type == "hide_ui":
        context.headless = False
        config = { "hide_ui": True }
    elif render_type == "normal":
        context.headless = False
        config = { "headless": False }
    time_step_sec = context.time_step_sec

    print(f"*** STARTING ISAAC SIM, render_type={render_type} unit_length={unit_length} ****")
    context.simulation_app = SimulationApp(config)

    from omni.isaac.core import World

    context.world = World(stage_units_in_meters=unit_length, physics_dt=time_step_sec)

    yield context.simulation_app

    print("*** CLOSING ISAAC SIM ****")
    context.simulation_app.close()


def before_all_isaac(context: Context, render_type: Literal["headless", "hide_ui", "normal"], enable_capture: bool, time_step_sec: float):
    context.render_type = render_type
    context.enable_capture = enable_capture
    context.time_step_sec = time_step_sec
    use_fixture(isaacsim_fixture, context, unit_length=1.0)

    g = getattr(context, "model_graph", None)
    assert g is not None, "'model_graph' attribute not found in context"

    context.execution_model = ExecutionModel(graph=g)
    context.us_loader = UserStoryLoader(graph=g)


def before_scenario_isaac(context: Context, scenario: Scenario):
    model_graph = getattr(context, "model_graph", None)
    assert model_graph is not None and isinstance(model_graph, Graph)

    us_loader = getattr(context, "us_loader", None)
    assert us_loader is not None and isinstance(us_loader, UserStoryLoader)

    # scenario outline renders each scenario as
    #   SCHEMA: "{outline_name} -- {examples.name}@{row.id}"
    scr_name_splits = scenario.name.split(" -- ")
    assert len(scr_name_splits) > 0, f"unexpected scenario name: {scenario.name}"
    scr_name = scr_name_splits[0]
    try:
        scenario_var_uri = model_graph.namespace_manager.expand_curie(scr_name)
    except ValueError as e:
        raise RuntimeError(
            f"can't parse behaviour URI '{scr_name}' from scenario '{scenario.name}': {e}"
        )

    scenario_var_model = us_loader.load_scenario_variant(
        full_graph=model_graph, variant_id=scenario_var_uri
    )
    scenario_var_model.scene.env_model_loader.register_attr_loaders(
        load_attr_path, load_py_module_attr
    )
    scenario_var_model.scene.agn_model_loader.register_attr_loaders(
        load_attr_path, load_py_module_attr
    )
    context.current_scenario = scenario_var_model

    # TODO(minhnh): handles different task? Isaacsim expects a single task
    from bdd_isaacsim_exec.tasks import load_isaacsim_task

    task = load_isaacsim_task(world=context.world, graph=model_graph, scr_var=scenario_var_model)
    print(f"**** Loaded Isaac Sim Task {task.name}")
    context.task = task

    from bdd_isaacsim_exec.utils import setup_camera_in_scene
    import omni.isaac.core.utils.numpy.rotations as rot_utils

    context.cameras = list()
    context.cameras.append(setup_camera_in_scene(
        name="camera_top",
        resolution=(1077, 480),
        position=np.array([0.3, 0.0, 7.5]),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 180]), degrees=True),
    ))
    context.cameras.append(setup_camera_in_scene(
        name="camera_front",
        resolution=(1077, 480),
        position=np.array([7.0, 0.0, 0.5]),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 180]), degrees=True),
    ))
    context.cameras.append(setup_camera_in_scene(
        name="camera_isometric",
        resolution=(1077, 480),
        position=np.array([5.5, 1.5, 2.0]),
        orientation=np.array([-0.14311696, -0.15081383, -0.02560492,  0.97781241]),
    ))
    context.frame_index = 0
    context.scenario_capture_folder = os.path.join(
        context.root_capture_folder,
        get_valid_var_name(scenario.name),
    )
    context.scenario_frames = {camera.name: [] for camera in context.cameras}


def after_scenario_isaac(context: Context):
    from omni.isaac.core.utils.prims import get_prim_at_path
    from bdd_isaacsim_exec.utils import save_frames, create_video_from_frames

    if context.enable_capture:
        for camera in context.cameras:
            save_frames(
                frames=context.scenario_frames[camera.name],
                capture_root_path=os.path.join(context.scenario_capture_folder, camera.name)
            )
            context.scenario_frames[camera.name] = []
            print(f"*** Frames saved for camera '{camera.name}'")
            video_path = create_video_from_frames(
                capture_root_path=os.path.join(context.scenario_capture_folder, camera.name),
                scenario_name=context.scenario.name,
            )
            print(f"*** Video saved to {video_path}")

            video_relative_path = os.path.join(*os.path.normpath(video_path).split(os.sep)[-3:])
            context.log_data[context.scenario.name]["cameras"].append({
                "name": camera.name,
                "resolution": str(camera.get_resolution()),
                "position": f"[{', '.join(str(x) for x in camera.get_world_pose()[0])}]",
                "orientation": f"[{', '.join(str(x) for x in camera.get_world_pose()[1])}]",
                "video_path": video_relative_path,
                "frame_count": context.frame_index,
            })

    context.task.cleanup_scene_models()
    context.world.clear()


def given_objects_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of object URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"
    for obj_model in load_obj_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        context.task.add_scene_obj_model(obj_model=obj_model)


def given_workspaces_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of object URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"
    for ws_model in load_ws_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        context.task.add_scene_ws_model(ws_model=ws_model, graph=context.model_graph)


def given_agents_isaac(context: Context):
    assert context.table is not None, "no table added to context, expected a list of agent URIs"
    assert context.model_graph is not None, "no 'model_graph' in context, expected an rdflib.Graph"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected an ScenarioVariantModel"
    for agn_model in load_agn_models_from_table(
        table=context.table, graph=context.model_graph, scene=context.current_scenario.scene
    ):
        context.task.add_scene_agn_model(agn_model=agn_model)


def setup_scene_isaac(context: Context):
    # trigger set_up_scene() function in an Isaac Sim BaseTask
    context.world.reset()


def _is_contained(
    obj_position: np.ndarray, ws_bounds: np.ndarray, margin: float = DIST_THRESHOLD
) -> bool:
    assert (
        len(obj_position) == 3 and len(ws_bounds) == 6
    ), f"unexpeced input dims: len(position)={len(obj_position)}, len(ws bounds)={len(ws_bounds)}"
    is_above = np.all(np.greater(obj_position, ws_bounds[:3] - margin))
    is_below = np.all(np.less(obj_position, ws_bounds[3:] + margin))
    return bool(is_above and is_below)


def _is_above(
    obj_position: np.ndarray, ws_bounds: np.ndarray, margin: float = DIST_THRESHOLD
) -> bool:
    assert (
        len(obj_position) == 3 and len(ws_bounds) == 6
    ), f"unexpeced input dims: len(position)={len(obj_position)}, len(ws bounds)={len(ws_bounds)}"
    within_projection = np.all(np.greater(obj_position[:2], ws_bounds[:2] - margin)) and np.all(
        np.less(obj_position[:2], ws_bounds[3:5] + margin)
    )
    is_above = obj_position[2] > ws_bounds[5] - margin
    return bool(within_projection and is_above)


def _is_located_by_ws_type(
    ws_id: URIRef, obj_position: np.ndarray, ws_bounds: np.ndarray, margin: float = DIST_THRESHOLD
) -> bool:
    if "bin" in ws_id:
        return _is_contained(obj_position=obj_position, ws_bounds=ws_bounds, margin=margin)

    if "table" in ws_id:
        return _is_above(obj_position=obj_position, ws_bounds=ws_bounds, margin=margin)

    if "shelf" in ws_id:
        return _is_contained(obj_position=obj_position, ws_bounds=ws_bounds, margin=margin)

    raise RuntimeError(f"is_located: unrecognized ws type: {ws_id}")


def is_located_at_isaac(context: Context, **kwargs):
    from bdd_isaacsim_exec.tasks import MeasurementType

    params = load_str_params(param_names=[PARAM_OBJ, PARAM_WS, PARAM_EVT], **kwargs)

    assert context.model_graph is not None, "no 'model_graph' in context"
    ns_manager = context.model_graph.namespace_manager

    _, obj_uris = parse_str_param(param_str=params[PARAM_OBJ], ns_manager=ns_manager)
    ws_param_type, ws_uris = parse_str_param(param_str=params[PARAM_WS], ns_manager=ns_manager)
    assert (
        len(obj_uris) > 0 and len(ws_uris) > 0
    ), f"is_located_at_isaac: expected at least 1 obj & ws, got obj={obj_uris}, ws={ws_uris}"

    for obj_id in obj_uris:
        assert isinstance(obj_id, URIRef), f"obj id not a URI: {obj_id}"
        context.task.add_measurement(elem_id=obj_id, meas_type=MeasurementType.OBJ_POSE)

    for ws_id in ws_uris:
        assert isinstance(ws_id, URIRef), f"ws id not a URI: {ws_id}"
        context.task.add_measurement(elem_id=ws_id, meas_type=MeasurementType.WS_BOUNDS)

    obs = context.world.get_observations()

    for obj_id in obj_uris:
        assert obj_id in obs, f"is_located: no measurement for obj '{obj_id.n3(ns_manager)}'"
        obj_position = obs[obj_id]["position"]

        is_located = True
        if ws_param_type == ParamType.EXISTS_SET:
            # Or comparision initialized to false
            is_located = False

        all_bounds = {}
        for ws_id in ws_uris:
            assert isinstance(ws_id, URIRef), f"ws id not a URI: {ws_id}"
            assert ws_id in obs, f"is_located: no measurement for ws '{ws_id.n3(ns_manager)}'"
            assert (
                "bounds" in obs[ws_id]
            ), f"bounds not loaded for ws '{ws_id.n3(ns_manager)}, meas: {obs[ws_id]}'"
            ws_bounds = obs[ws_id]["bounds"]
            all_bounds[ws_id.n3(ns_manager)] = ws_bounds

            is_located_cur_ws = _is_located_by_ws_type(
                ws_id=ws_id, obj_position=obj_position, ws_bounds=ws_bounds
            )
            if ws_param_type == ParamType.EXISTS_SET:
                is_located |= is_located_cur_ws
            else:
                is_located &= is_located_cur_ws

        if is_located:
            continue

        context.step_debug_info["fail_info"]["causes"] = ["not_located"]
        context.step_debug_info["fail_info"]["obj_id"] = obj_id.n3(ns_manager)
        context.step_debug_info["fail_info"]["obj_position"] = obj_position.tolist()
        context.step_debug_info["fail_info"]["ws_bounds"] = {}
        for ws_id in all_bounds:
            context.step_debug_info["fail_info"]["ws_bounds"][ws_id] = all_bounds[ws_id].tolist()
        raise AssertionError(
            f"obj '{obj_id.n3(ns_manager)}' (pos={obj_position}) not located at {params[PARAM_WS]}, bounds:\n{all_bounds}"
        )

    evt_uri = try_expand_curie(curie_str=params[PARAM_EVT], ns_manager=ns_manager, quiet=False)
    assert evt_uri is not None, f"can't parse '{params[PARAM_EVT]}' as URI"


def is_sorted_isaac(context: Context, **kwargs):
    from bdd_isaacsim_exec.tasks import MeasurementType

    assert context.model_graph is not None, "no 'model_graph' in context"
    assert (
        context.current_scenario is not None
    ), "no 'current_scenario' in context, expected a ScenarioVariantModel"
    params = load_str_params(param_names=[PARAM_OBJ, PARAM_WS, PARAM_EVT], **kwargs)

    assert context.model_graph is not None, "no 'model_graph' in context"
    ns_manager = context.model_graph.namespace_manager

    _, obj_uris = parse_str_param(param_str=params[PARAM_OBJ], ns_manager=ns_manager)
    _, ws_uris = parse_str_param(param_str=params[PARAM_WS], ns_manager=ns_manager)
    assert (
        len(obj_uris) > 0 and len(ws_uris) > 0
    ), f"is_located_at_isaac: expected at least 1 obj & ws, got obj={obj_uris}, ws={ws_uris}"

    for obj_id in obj_uris:
        assert isinstance(obj_id, URIRef), f"obj id not a URI: {obj_id}"
        context.task.add_measurement(elem_id=obj_id, meas_type=MeasurementType.OBJ_POSE)

    for ws_id in ws_uris:
        assert isinstance(ws_id, URIRef), f"ws id not a URI: {ws_id}"
        context.task.add_measurement(elem_id=ws_id, meas_type=MeasurementType.WS_BOUNDS)

    obs = context.world.get_observations()

    obj_colors = {}
    obj_by_ws = {}
    for ws_id in ws_uris:
        assert ws_id in obs, f"is_sorted: no measurement for ws '{ws_id.n3(ns_manager)}'"
        ws_bounds = obs[ws_id]["bounds"]

        obj_by_ws[ws_id] = []
        for obj_id in obj_uris:
            assert obj_id in obs, f"is_sorted: no measurement for obj '{obj_id.n3(ns_manager)}'"
            # location
            obj_position = obs[obj_id]["position"]
            if _is_located_by_ws_type(ws_id=ws_id, obj_position=obj_position, ws_bounds=ws_bounds):
                obj_by_ws[ws_id].append(obj_id)
            # color
            if obj_id in obj_colors:
                continue
            obj_model = context.task.get_obj_model(obj_id=obj_id)
            model_configs = obj_model.get_attr(key=URI_SIM_PRED_HAS_CONFIG)
            assert "color" in model_configs, f"obj '{obj_id.n3(ns_manager)}' has no color attr"
            obj_colors[obj_id] = model_configs["color"]

    sorted_correct = True
    wrongly_sorted_ws = None
    for ws_id, ws_obj_uris in obj_by_ws.items():
        color = None
        for obj_id in ws_obj_uris:
            if color is None:
                color = obj_colors[obj_id]
                continue

            if obj_colors[obj_id] == color:
                continue

            sorted_correct = False
            wrongly_sorted_ws = ws_id
            break

    if sorted_correct:
        return

    assert isinstance(
        wrongly_sorted_ws, URIRef
    ), f"unexpected val for wrongly sorted WS: {wrongly_sorted_ws}"
    context.step_debug_info["fail_info"]["causes"] = ["wrong_color"]
    context.step_debug_info["fail_info"]["obj_by_ws"] = obj_by_ws
    context.step_debug_info["fail_info"]["obj_colors"] = obj_colors
    context.step_debug_info["fail_info"]["wrongly_sorted_ws"] = wrongly_sorted_ws
    raise AssertionError(
        f"objs '{', '.join(x.n3(ns_manager) for x in obj_by_ws[wrongly_sorted_ws])}'"
        + f" in ws '{wrongly_sorted_ws.n3(ns_manager)}' not of same color"
    )


def move_safely_isaac(context: Context, **kwargs):
    params = load_str_params(param_names=[PARAM_AGN, PARAM_FROM_EVT, PARAM_UNTIL_EVT], **kwargs)

    graph = getattr(context, "model_graph", None)
    assert isinstance(graph, Graph), f"no 'model_graph' of type rdflib.Graph in context: {graph}"

    # agent dependent speed threshold
    task_params = context.task.get_params()
    agn_ids = task_params["agents"]["value"]["uris"]
    assert len(agn_ids) == 1 and isinstance(agn_ids[0], URIRef), f"unexpected agn param: {agn_ids}"
    agn_model = context.task.get_agn_model(agn_id=agn_ids[0])
    if URI_FRANKA_PANDA in agn_model.types:
        threshold = PANDA_SPEED_THRESHOLD
    elif URI_UR_UR10 in agn_model.types:
        threshold = UR10_SPEED_THRESHOLD
    else:
        raise RuntimeError(f"unexpected agn types: {agn_model.types}")

    assert hasattr(
        context, "bhv_observations"
    ), "move_safely_isaac: no 'bhv_observations' in context"

    move_too_fast = False
    ws_moved = False
    debug_info = {}
    debug_info["causes"] = []
    debug_msgs = []

    # eval speed with a moving average filter
    agn_speeds = context.bhv_observations["agn_speeds"]
    filter_horizon = 3
    for i in range(len(agn_speeds) - filter_horizon):
        filtered_speed = np.mean(agn_speeds[i : i + filter_horizon])
        if filtered_speed < threshold:
            continue

        move_too_fast = True
        debug_info["causes"].append("fast_ee")
        debug_info["agn_id"] = params[PARAM_AGN]
        debug_info["ee_speed"] = float(filtered_speed)
        debug_msgs.append(
            f"agent '{params[PARAM_AGN]}' moves EE too fast: {filtered_speed:.5f} m/s > {threshold:.5f} m/s"
        )
        break

    # eval sum of displacements for place workspaces
    ws_displacement_sum = context.bhv_observations["ws_displacement_sum"]
    ws_disp_threshold = DIST_THRESHOLD * 5
    for ws_id, displacement_sum in ws_displacement_sum.items():
        if displacement_sum < ws_disp_threshold:
            continue

        ws_moved = True
        debug_info["causes"].append("ws_moved")
        debug_info["ws_id"] = ws_id.n3(graph.namespace_manager)
        debug_info["displacement_sum"] = displacement_sum
        debug_msgs.append(
            f"ws '{ws_id.n3(graph.namespace_manager)}' was moved too much:"
            + f" displacement_sum={displacement_sum:.5f} m > {ws_disp_threshold:.5f} m"
        )
        break

    # return if all is fine or update debug_info and raise AssertionError
    if (not move_too_fast) and (not ws_moved):
        return

    context.step_debug_info["fail_info"] |= debug_info
    raise AssertionError(
        f"'{params[PARAM_AGN]}' did not move safely. Reason(s):\n- " + "\n- ".join(debug_msgs)
    )


def behaviour_isaac(context: Context, **kwargs):
    from bdd_isaacsim_exec.tasks import MeasurementType
    from bdd_isaacsim_exec.utils import capture_camera_image

    params = load_str_params(param_names=[PARAM_AGN, PARAM_OBJ, PARAM_WS], **kwargs)
    context.task.set_params(
        agn_id_str=params[PARAM_AGN],
        obj_id_str=params[PARAM_OBJ],
        ws_id_str=params[PARAM_WS],
    )

    behaviour_model = getattr(context, "behaviour_model", None)

    graph = getattr(context, "model_graph", None)
    assert isinstance(graph, Graph), f"no 'model_graph' of type rdflib.Graph in context: {graph}"

    if behaviour_model is None:
        exec_model = getattr(context, "execution_model", None)
        assert isinstance(
            exec_model, ExecutionModel
        ), f"no valid 'execution_model' added to the context: {exec_model}"

        behaviour_model = exec_model.load_behaviour_impl(
            context=context,
            ns_manager=graph.namespace_manager,
        )
        context.behaviour_model = behaviour_model

    task_params = context.task.get_params()

    agn_ids = task_params["agents"]["value"]["uris"]
    assert len(agn_ids) == 1 and isinstance(agn_ids[0], URIRef), f"unexpected agn param: {agn_ids}"

    obj_ids = task_params["objects"]["value"]["uris"]
    assert len(obj_ids) == 1 and isinstance(obj_ids[0], URIRef), f"unexpected obj param: {obj_ids}"

    place_ws_ids = task_params["place_workspaces"]["value"]["uris"]
    for uri in place_ws_ids:
        assert isinstance(uri, URIRef), f"unexpected ws param: {uri}"

    render = not context.headless

    bhv = behaviour_model.behaviour
    assert isinstance(
        bhv, Behaviour
    ), f"bhv impl for '{behaviour_model.id.n3(graph.namespace_manager)}' not instance of '{Behaviour}': {bhv}"
    bhv.reset(context=context, agn_id=agn_ids[0], obj_id=obj_ids[0], place_ws_ids=place_ws_ids)

    # Move safely clause requires assertion over a time horizon
    #  safety metric 1 -- end-effector speed
    context.task.add_measurement(elem_id=agn_ids[0], meas_type=MeasurementType.AGN_EE_LINEAR_VEL)
    agn_speeds = []
    #  safety metric 2 -- bins culmulative displacement
    ws_displacement_sums = {}
    ws_displacement_sums_frame = {}
    ws_obj_map = {}
    ws_previous_positions = {}
    for ws_id in place_ws_ids:
        context.task.add_measurement(elem_id=ws_id, meas_type=MeasurementType.WS_POSE)
        ws_displacement_sums[ws_id] = 0
        ws_displacement_sums_frame[ws_id] = 0
        ws_model = context.task.get_ws_model(ws_id=ws_id)
        assert len(ws_model.objects) > 0, f"no obj linked to ws '{ws_id}'"
        for obj_id in ws_model.objects:
            ws_obj_map[ws_id] = obj_id
            break  # only first obj

    time_step_sec = context.time_step_sec
    now = time.process_time()
    loop_end = now + time_step_sec
    exec_times = []
    while context.simulation_app.is_running():
        if bhv.is_finished(context=context):
            break
        context.world.step(render=render)
        # frame capture
        if context.enable_capture:
            for i in range(len(context.cameras)):
                camera = context.cameras[i]
                context.scenario_frames[camera.name].append(capture_camera_image(camera))
                timestamp_unix = time.time()
                scenario_start_time = context.log_data[context.scenario.name]["start_time_unix"]
                timestamp_rel = timestamp_unix - scenario_start_time
                obs = context.world.get_observations()
                for ws_id in place_ws_ids:
                    ws_position = obs[ws_obj_map[ws_id]]["position"]
                    if ws_id in ws_previous_positions:
                        ws_displacement = np.linalg.norm(ws_position - ws_previous_positions[ws_id])
                        ws_displacement_sums_frame[ws_id] += ws_displacement
                context.frame_logs.append({
                    "camera": camera.name,
                    "frame_id": context.frame_index,
                    "filename": f"frame_{context.frame_index:05d}.jpg",
                    "timestamp_unix": timestamp_unix,
                    "timestamp_rel": timestamp_rel,
                    "ee_speed": str(np.linalg.norm(obs[agn_ids[0]]["ee_linear_velocities"])),
                    "ws_displacement": {
                        ws_id.n3(graph.namespace_manager): ws_displacement_sums_frame[ws_id]
                        for ws_id in place_ws_ids
                    },
                })
                ws_displacement_sums_frame = {ws_id: 0 for ws_id in place_ws_ids}
            context.frame_index += 1
        # observations
        obs = context.world.get_observations()
        # behaviour step
        bhv.step(context=context, observations=obs)
        # metrics
        agn_speed = np.linalg.norm(obs[agn_ids[0]]["ee_linear_velocities"])
        agn_speeds.append(agn_speed)
        for ws_id in place_ws_ids:
            ws_position = obs[ws_obj_map[ws_id]]["position"]
            if ws_id in ws_previous_positions:
                ws_displacement = np.linalg.norm(ws_position - ws_previous_positions[ws_id])
                ws_displacement_sums[ws_id] += ws_displacement
            ws_previous_positions[ws_id] = ws_position
        # real time check
        exec_times.append(time.process_time() - now)
        while True:
            # equivalent to do-while loop
            now = time.process_time()
            if now > loop_end:
                break
        while loop_end < now:
            loop_end += time_step_sec

    context.bhv_observations = {
        "agn_speeds": agn_speeds,
        "ws_displacement_sum": ws_displacement_sums,
    }

    if len(agn_speeds) > 0:
        speed_mean = np.mean(agn_speeds)
        speed_std = np.std(agn_speeds)
        speed_min = np.min(agn_speeds)
        speed_max = np.max(agn_speeds)
        context.step_debug_info["ee_speed"] = {}
        context.step_debug_info["ee_speed"]["mean"] = "NaN" if np.isnan(speed_mean) else float(speed_mean)
        context.step_debug_info["ee_speed"]["std"] = "NaN" if np.isnan(speed_std) else float(speed_std)
        context.step_debug_info["ee_speed"]["min"] = "NaN" if np.isnan(speed_min) else float(speed_min)
        context.step_debug_info["ee_speed"]["max"] = "NaN" if np.isnan(speed_max) else float(speed_max)
        context.step_debug_info["ws_displacement"] = {
            ws_id.n3(graph.namespace_manager): ws_displacement_sums[ws_id]
            for ws_id in place_ws_ids
        }
        print(
            "\n\n*** Agent speed statistics: "
            + f" mean={speed_mean:.5f}, std={speed_std:.5f},"
            + f" min={speed_min:.5f}, max={speed_max:.5f}"
        )
    else:
        print("\n*** WARNING: no agent EE speed recorded\n")
    print(
        f"\n\n*** Execution time statistics (secs) for '{len(exec_times)}' loops:"
        + f" mean={np.mean(exec_times):.5f}, std={np.std(exec_times):.5f},"
        + f" min={np.min(exec_times):.5f}, max={np.max(exec_times):.5f}\n\n"
    )
