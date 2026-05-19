"""Custom actuator wrappers for XML-defined Robotiq general actuators."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from mjlab.actuator.xml_actuator import XmlActuator, XmlActuatorCfg
from mjlab.entity import Entity


@dataclass(kw_only=True)
class RobotiqGeneralActuatorCfg(XmlActuatorCfg):
    """Wrap a Robotiq `<general>` actuator as a position-like command input."""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> "RobotiqGeneralActuator":
        """Build the actuator wrapper."""
        return RobotiqGeneralActuator(self, entity, target_ids, target_names)


class RobotiqGeneralActuator(XmlActuator):
    """XML actuator wrapper that treats the Robotiq general actuator as position-like."""

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        """Find the matching XML actuators and force a position command field."""
        filtered_target_ids = []
        filtered_target_names = []
        for i, target_name in enumerate(target_names):
            actuator = self._find_actuator_for_target(spec, target_name)
            if actuator is None:
                continue
            self._mjs_actuators.append(actuator)
            filtered_target_ids.append(self._target_ids_list[i])
            filtered_target_names.append(self._target_names[i])

        if len(filtered_target_names) == 0:
            raise ValueError(
                "No Robotiq general XML actuators were found for the configured targets."
            )

        self._target_ids_list = filtered_target_ids
        self._target_names = filtered_target_names
        self._command_field = "position"
