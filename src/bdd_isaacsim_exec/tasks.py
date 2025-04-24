# SPDX-License-Identifier:  GPL-3.0-or-later
from enum import Enum
from typing import Any
import numpy as np
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import NamespaceManager
from rdf_utils.naming import get_valid_var_name
from rdf_utils.uri import URL_SECORO_M
from bdd_dsl.behave import parse_str_param
from bdd_dsl.models.agent import AgentModel
from bdd_dsl.models.environment import ObjectModel, WorkspaceModel
from bdd_dsl.models.user_story import ScenarioVariantModel, SceneModel
from bdd_isaacsim_exec.utils import create_rigid_prim_in_scene

from omni.isaac.core import World
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.scenes.scene import Scene as IsaacScene
import omni.isaac.core.utils.bounds as bounds_utils
from omni.isaac.sensor import Camera


NS_M_TMPL = Namespace(f"{URL_SECORO_M}/acceptance-criteria/bdd/templates/")
URI_M_BHV_PICKPLACE = NS_M_TMPL["bhv-pickplace"]
URI_M_TASK_PICKPLACE = NS_M_TMPL["task-pickplace"]
URI_M_TASK_SORTING = NS_M_TMPL["task-sorting"]


class MeasurementType(Enum):
    OBJ_POSE = 0
    WS_POSE = 1
    WS_BOUNDS = 2
    AGN_JNT_POSITIONS = 3
    AGN_EE_LINEAR_VEL = 4


class PickPlace(BaseTask):
    _ns_manager: NamespaceManager
    _obj_prims: dict[URIRef, RigidPrim]
    _agn_prims: dict[URIRef, Robot]
    _obj_models: dict[URIRef, ObjectModel]
    _ws_models: dict[URIRef, WorkspaceModel]
    _agn_models: dict[URIRef, AgentModel]
    _task_params: dict
    _measurements: dict[URIRef, set[MeasurementType]]

    def __init__(
        self, scene_model: SceneModel, task_name: str, ns_manger: NamespaceManager, **kwargs: Any
    ) -> None:
        offset = kwargs.get("offset", None)
        BaseTask.__init__(self, name=task_name, offset=offset)

        self._ns_manager = ns_manger
        # TODO(minhnh): handle different scenes for the same task
        self._scene_model = scene_model
        self._obj_prims = {}
        self._agn_prims = {}
        self._obj_models = {}
        self._ws_models = {}
        self._agn_models = {}
        self._task_params = {
            "objects": {"value": {}, "modifiable": True},
            "agents": {"value": {}, "modifiable": True},
            "place_workspaces": {"value": {}, "modifiable": True}
        }
        self._measurements = {}
        self._bb_cache = bounds_utils.create_bbox_cache()

    def get_obj_model(self, obj_id: URIRef) -> ObjectModel:
        assert obj_id in self._obj_models, f"Task {self.name}: no model for obj {obj_id}"
        return self._obj_models[obj_id]

    def get_ws_model(self, ws_id: URIRef) -> WorkspaceModel:
        assert ws_id in self._ws_models, f"Task {self.name}: no model for ws {ws_id}"
        return self._ws_models[ws_id]

    def get_agn_model(self, agn_id: URIRef) -> AgentModel:
        assert agn_id in self._agn_models, f"Task {self.name}: no model for agent {agn_id}"
        return self._agn_models[agn_id]

    def get_agn_prim(self, agn_id: URIRef) -> Robot:
        assert agn_id in self._agn_prims, f"Task {self.name}: no prim for agent {agn_id}"
        return self._agn_prims[agn_id]

    def add_scene_obj_model(self, obj_model: ObjectModel) -> None:
        if obj_model.id in self._obj_models:
            return
        self._obj_models[obj_model.id] = obj_model

    def add_scene_ws_model(self, ws_model: WorkspaceModel, graph: Graph) -> None:
        if ws_model.id in self._ws_models:
            return
        self._ws_models[ws_model.id] = ws_model
        for obj_model in self._scene_model.env_model_loader.load_ws_objects(
            graph=graph, ws_id=ws_model.id
        ):
            self.add_scene_obj_model(obj_model=obj_model)

    def add_measurement(self, elem_id: URIRef, meas_type: MeasurementType) -> None:
        try:
            meas_type = MeasurementType(meas_type)
        except ValueError as e:
            raise RuntimeError(f"Measurement type '{meas_type}' not supported: {e}")
        if elem_id not in self._measurements:
            self._measurements[elem_id] = set()

        if meas_type == MeasurementType.OBJ_POSE:
            assert (
                elem_id in self._obj_models
            ), f"'{meas_type}' for obj '{elem_id.n3(namespace_manager=self._ns_manager)}': obj URI not on record"
        elif meas_type == MeasurementType.WS_POSE or meas_type == MeasurementType.WS_BOUNDS:
            assert (
                elem_id in self._ws_models
            ), f"'{meas_type}' for ws '{elem_id.n3(namespace_manager=self._ns_manager)}': ws URI not on record"
        elif (
            meas_type == MeasurementType.AGN_JNT_POSITIONS
            or meas_type == MeasurementType.AGN_EE_LINEAR_VEL
        ):
            assert (
                elem_id in self._agn_models
            ), f"'{meas_type}' for agn '{elem_id.n3(namespace_manager=self._ns_manager)}': agn URI not on record"
        self._measurements[elem_id].add(meas_type)

    def add_scene_agn_model(self, agn_model: AgentModel) -> None:
        if agn_model.id in self._agn_models:
            return
        self._agn_models[agn_model.id] = agn_model

    def set_up_scene(self, scene: IsaacScene) -> None:
        super().set_up_scene(scene)

        scene.add_default_ground_plane()

        for obj_id, obj_model in self._obj_models.items():
            print(f"*** loading model for object {obj_id}")
            obj_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                model=obj_model,
                prim_prefix="/World/Objects/",
            )
            self._obj_prims[obj_id] = obj_prim

        for agn_id, agn_model in self._agn_models.items():
            print(f"*** loading model for agent {agn_id}")
            agn_prim = create_rigid_prim_in_scene(
                scene=scene,
                ns_manager=self._ns_manager,
                model=agn_model,
                prim_prefix="/World/Agents/",
            )
            assert isinstance(
                agn_prim, Robot
            ), f"Prim for agn '{agn_id}' not a Isaac Robot instance"
            self._agn_prims[agn_id] = agn_prim

    def set_params(self, **kwargs) -> None:
        """Set parameters values.

        Extension of BaseTask.set_params().

        Parameters:
        - obj_id_str: string rep of object URI(s) to be picked and placed
        - agent_id_str: string rep of agent URI(s) to perform pick & place bhv
        - place_ws_id: URI of workspace(s) for placing.
        """
        agn_id_str = kwargs.get("agn_id_str", None)
        assert agn_id_str is not None, f"Isaac Task '{self.name}': arg 'agn_id_str' not specified'"
        obj_id_str = kwargs.get("obj_id_str", None)
        assert obj_id_str is not None, f"Isaac Task '{self.name}': arg 'obj_id_str' not specified'"
        ws_id_str = kwargs.get("ws_id_str", None)
        assert ws_id_str is not None, f"Isaac Task '{self.name}': arg 'ws_id_str not specified'"

        param_type, agn_uris = parse_str_param(param_str=agn_id_str, ns_manager=self._ns_manager)
        assert (
            len(agn_uris) > 0
        ), f"Isaac Task '{self.name}': expected at least 1 agent URI, got: {agn_uris}"
        self._task_params["agents"]["value"]["param_type"] = param_type
        self._task_params["agents"]["value"]["uris"] = agn_uris

        param_type, obj_uris = parse_str_param(param_str=obj_id_str, ns_manager=self._ns_manager)
        assert (
            len(obj_uris) > 0
        ), f"Isaac Task '{self.name}': expected at least 1 obj URI, got: {obj_uris}"
        self._task_params["objects"]["value"]["param_type"] = param_type
        self._task_params["objects"]["value"]["uris"] = obj_uris

        param_type, place_ws_uris = parse_str_param(
            param_str=ws_id_str, ns_manager=self._ns_manager
        )
        assert (
            len(place_ws_uris) > 0
        ), f"Isaac Task '{self.name}': expected at least 1 place WS URI, got: {place_ws_uris}"
        self._task_params["place_workspaces"]["value"]["param_type"] = param_type
        self._task_params["place_workspaces"]["value"]["uris"] = place_ws_uris

        return

    def get_params(self) -> dict:
        """Return parameters values.

        Extension of BaseTask.get_params().

        Parameters:
        - agents: dict of param type & URI(s) of agent(s) performing pick & place
        - objects: dict of param type & URI(s) of object(s) to be picked and placed
        - place_workspaces: dict of param type & URI(s) of workspace(s) where
                            the picked object(s) should be placed

        Returns:
            dict: dictionary containing parameters' values and modifiable flag
        """
        assert (
            len(self._task_params["agents"]["value"]) > 0
            and len(self._task_params["objects"]["value"]) > 0
            and len(self._task_params["place_workspaces"]["value"]) > 0
        ), f"Isaac Task '{self.name}': get_params called before set_params, params not populated"

        return self._task_params

    def get_obj_pose(self, obj_id: URIRef) -> tuple[np.ndarray, np.ndarray]:
        assert obj_id in self._obj_prims, f"Isaac Task '{self.name}': no prim for obj '{obj_id}'"
        obj_position, obj_orientation = self._obj_prims[obj_id].get_world_pose()
        assert (
            len(obj_position) == 3
        ), f"unexpected length for obj {obj_id.n3(namespace_manager=self._ns_manager)} pos: {obj_position}"
        assert isinstance(
            obj_position, np.ndarray
        ), f"unexpected type for obj {obj_id.n3(namespace_manager=self._ns_manager)} pos: {obj_position}"
        assert (
            len(obj_orientation) == 4
        ), f"unexpected length for obj {obj_id.n3(namespace_manager=self._ns_manager)} orn: {obj_orientation}"
        assert isinstance(
            obj_orientation, np.ndarray
        ), f"unexpected type for obj {obj_id.n3(namespace_manager=self._ns_manager)} orn: {obj_orientation}"
        return obj_position, obj_orientation

    def get_ws_pose(self, ws_id: URIRef) -> dict[URIRef, tuple[np.ndarray, np.ndarray]]:
        assert ws_id in self._ws_models, f"Isaac Task '{self.name}': ws '{ws_id}' not on record"
        obj_poses = {}
        for obj_id in self._ws_models[ws_id].objects:
            obj_position, obj_orientation = self.get_obj_pose(obj_id=obj_id)
            obj_poses[obj_id] = {"position": obj_position, "orientation": obj_orientation}
        return obj_poses

    def get_agn_joint_positions(self, agn_id: URIRef) -> np.ndarray:
        assert agn_id in self._agn_prims, f"Isaac Task '{self.name}': no prim for agn '{agn_id}'"
        return self._agn_prims[agn_id].get_joint_positions()

    def get_agn_ee_linear_vel(self, agn_id: URIRef) -> np.ndarray:
        assert agn_id in self._agn_prims, f"Isaac Task '{self.name}': no prim for agn '{agn_id}'"
        return self._agn_prims[agn_id].end_effector.get_linear_velocity()

    def get_observations(self) -> dict:
        """Return observations"""
        obs = {}
        for uri, meas_types in self._measurements.items():
            if uri not in obs:
                obs[uri] = {}

            if MeasurementType.OBJ_POSE in meas_types:
                obj_position, obj_orientation = self.get_obj_pose(obj_id=uri)
                obs[uri] |= {"position": obj_position, "orientation": obj_orientation}

            if MeasurementType.WS_POSE in meas_types:
                ws_obj_poses = self.get_ws_pose(ws_id=uri)
                obs[uri]["objects"] = self._ws_models[uri].objects
                obs |= ws_obj_poses

            if MeasurementType.WS_BOUNDS in meas_types:
                final_bounds = np.repeat([[np.finfo("f").max, np.finfo("f").min]], 3)
                for obj_id in self._ws_models[uri].objects:
                    obj_bounds = bounds_utils.compute_aabb(
                        self._bb_cache, prim_path=self._obj_prims[obj_id].prim_path
                    )
                    final_bounds = np.concatenate(
                        (
                            np.minimum(final_bounds[:3], obj_bounds[:3]),
                            np.maximum(final_bounds[3:], obj_bounds[3:]),
                        )
                    )
                obs[uri]["bounds"] = final_bounds

            if MeasurementType.AGN_JNT_POSITIONS in meas_types:
                agn_joint_positions = self.get_agn_joint_positions(agn_id=uri)
                obs[uri] |= {
                    "joint_positions": agn_joint_positions,
                }

            if MeasurementType.AGN_EE_LINEAR_VEL in meas_types:
                agn_ee_linear_vels = self.get_agn_ee_linear_vel(agn_id=uri)
                obs[uri] |= {"ee_linear_velocities": agn_ee_linear_vels}

        return obs

    def cleanup_scene_models(self) -> None:
        """Should be called before loading obj and agent models.

        Either in before_scenario or after_scenario
        """
        self._obj_models.clear()
        self._ws_models.clear()
        self._agn_models.clear()
        self._obj_prims.clear()
        self._agn_prims.clear()
        self._measurements.clear()


def load_isaacsim_task(
    world: World, graph: Graph, scr_var: ScenarioVariantModel, **kwargs: Any
) -> BaseTask:
    task_name = get_valid_var_name(
        scr_var.scenario.task_id.n3(namespace_manager=graph.namespace_manager)
    )
    try:
        return world.get_task(task_name)
    except Exception:
        # task is not added
        pass

    if (
        scr_var.scenario.task_id == URI_M_TASK_PICKPLACE
        or scr_var.scenario.task_id == URI_M_TASK_SORTING
    ):
        task = PickPlace(
            scene_model=scr_var.scene,
            task_name=task_name,
            ns_manger=graph.namespace_manager,
            **kwargs,
        )
        world.add_task(task)
        return task

    raise RuntimeError(f"unhandled task: {scr_var.scenario.task_id}")
