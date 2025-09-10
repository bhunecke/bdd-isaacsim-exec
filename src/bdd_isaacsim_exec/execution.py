# SPDX-License-Identifier:  GPL-3.0-or-later
from typing import Any, Optional
import numpy as np
from behave.runner import Context
from rdflib import URIRef
from rdflib.namespace import NamespaceManager
from bdd_dsl.execution.common import Behaviour
from bdd_isaacsim_exec.tasks import MeasurementType
from bdd_isaacsim_exec.uri import URI_FRANKA_PANDA, URI_UR_UR10

from omni.isaac.manipulators.controllers import PickPlaceController as GenPickPlaceController
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.objects import VisualCone
from omni.isaac.core.utils.string import find_unique_string_name


class IsaacsimPickPlaceBehaviour(Behaviour):
    agn_id: Optional[URIRef]
    obj_id: Optional[URIRef]
    ws_id: Optional[list[URIRef]]
    _fsm: Optional[GenPickPlaceController]
    _art_ctrl: Optional[ArticulationController]
    _vis_target: Optional[VisualCone]
    _visualize_target_obj: bool

    def __init__(
        self,
        bhv_id: URIRef,
        bhv_types: set[URIRef],
        context: Context,
        ns_manager: NamespaceManager,
        **kwargs: Any,
    ) -> None:
        super().__init__(bhv_id=bhv_id, bhv_types=bhv_types, context=context, **kwargs)
        self._ns_manager = ns_manager
        self._fsm = None
        self._gripper_offset = None

        # add red visual cone to visualize target obj
        self._visualize_target_obj = kwargs.get("visualize_target_obj", True)
        self._vis_target = None

    def is_finished(self, context: Context, **kwargs: Any) -> bool:
        assert (
            self._fsm is not None
        ), f"Behaviour '{self.id}': no FSM loaded, reset() not yest called"
        return self._fsm.is_done()

    def reset(self, context: Context, **kwargs: Any) -> None:
        self.agn_id = kwargs.get("agn_id")
        assert isinstance(self.agn_id, URIRef), f"unexpected 'agn_id' arg: {self.agn_id}"

        self.obj_id = kwargs.get("obj_id")
        assert isinstance(self.obj_id, URIRef), f"unexpected 'obj_id' arg: {self.obj_id}"

        place_ws_ids = kwargs.get("place_ws_ids")
        assert isinstance(place_ws_ids, list), f"unexpected 'place_ws_ids' arg: {place_ws_ids}"
        rand_ws_index = np.random.randint(len(place_ws_ids))
        self.ws_id = place_ws_ids[rand_ws_index]
        assert isinstance(self.ws_id, URIRef), f"unexpected ws URI: {self.ws_id}"

        # add measurements required by behaviour
        context.task.add_measurement(elem_id=self.obj_id, meas_type=MeasurementType.OBJ_POSE)
        context.task.add_measurement(elem_id=self.ws_id, meas_type=MeasurementType.WS_POSE)
        context.task.add_measurement(
            elem_id=self.agn_id, meas_type=MeasurementType.AGN_JNT_POSITIONS
        )

        agn_prim = context.task.get_agn_prim(self.agn_id)
        self._art_ctrl = agn_prim.get_articulation_controller()

        agn_model = context.task.get_agn_model(self.agn_id)
        if URI_FRANKA_PANDA in agn_model.types:
            from omni.isaac.franka.controllers import PickPlaceController as PandaPickPlaceCtrl

            self._fsm = PandaPickPlaceCtrl(
                name="pick_place_controller", gripper=agn_prim.gripper, robot_articulation=agn_prim
            )
            self._gripper_offset = np.array([0, 0.005, 0.0])
        elif URI_UR_UR10 in agn_model.types:
            from omni.isaac.universal_robots.controllers import (
                PickPlaceController as URPickPlaceCtrl,
            )

            self._fsm = URPickPlaceCtrl(
                name="pick_place_controller", gripper=agn_prim.gripper, robot_articulation=agn_prim
            )
            self._gripper_offset = np.array([0, 0.0, 0.02])
        else:
            raise RuntimeError(
                f"Behaviour '{self.id}': unrecognized agent types: {agn_model.types}"
            )
        if self._gripper_offset is None:
            self._gripper_offset = np.zeros(3)

        if self._visualize_target_obj:
            vis_target_name = find_unique_string_name(
                initial_name="visual_target",
                is_unique_fn=lambda x: not context.world.scene.object_exists(x),
            )
            self._vis_target = context.world.scene.add(
                VisualCone(
                    prim_path="/World/Xform/visual_target",
                    name=vis_target_name,
                    color=np.array([1.0, 0.0, 0.0]),
                    orientation=np.array([1, 0, 0, 0]),
                    radius=0.02,
                    height=0.02,
                )
            )

        assert self._fsm is not None, "Behaviour.reset: _fsm is None"
        agn_prim.gripper.open()
        self._fsm.reset()

    def step(self, context: Context, **kwargs: Any) -> Any:
        assert (
            self.agn_id is not None
            and self.obj_id is not None
            and self.ws_id is not None
            and self._fsm is not None
            and self._art_ctrl is not None
        ), f"Behaviour '{self.id}': params are None, step() expects reset() to be called first"
        if self._visualize_target_obj:
            assert (
                self._vis_target is not None
            ), f"Behaviour '{self.id}':visualising target obj enabled but visual shape not initialized"

        obs = kwargs.get("observations")
        assert obs is not None, f"{self.__class__}.step missing 'observations' arg"
        assert self.obj_id in obs, f"target obj '{self.obj_id}' not in observations"
        assert self.ws_id in obs, f"target ws '{self.ws_id}' not in observations"
        assert self.agn_id in obs, f"target agn '{self.agn_id}' not in observations"

        ws_obj_list = list(obs[self.ws_id]["objects"])
        assert len(ws_obj_list) == 1, f"unexpected number of ws objects (not 1): {ws_obj_list}"
        ws_obj_id = ws_obj_list[0]
        assert (
            ws_obj_id in obs
        ), f"obj '{ws_obj_id}' of target place ws '{self.ws_id}' not in observations"

        grasp_position = obs[self.obj_id]["position"]
        if self._vis_target is not None:
            self._vis_target.set_world_pose(
                position=grasp_position + [0, 0, 0.05],
                orientation=obs[self.obj_id]["orientation"],
            )

        actions = self._fsm.forward(
            picking_position=grasp_position,
            placing_position=obs[ws_obj_id]["position"] + [0, 0, 0.05],
            current_joint_positions=obs[self.agn_id]["joint_positions"],
            end_effector_offset=self._gripper_offset,
        )
        self._art_ctrl.apply_action(actions)
