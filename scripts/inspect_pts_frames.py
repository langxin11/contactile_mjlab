"""使用 mjviser 可视化 PTSSpheres 模型中的 site 局部坐标系.

默认行为是静态加载 ``scene_pts_spheres.xml``，并在浏览器中叠加：

- 左右 pad FT site 的局部坐标系
- 左右 3x3 taxel site 的局部坐标系

本脚本会用 ``mujoco.mj_step`` 推进动力学，用于检查闭链约束和 actuator 推动下的
site/sensor 坐标系跟随情况；它不接 ``mjlab`` 环境，也不参与训练。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mjviser
import mujoco
import numpy as np
import typer
import viser
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contactile_mjlab.paths import PTS_SPHERES_SCENE_XML
from contactile_mjlab.tasks.tactile_grasp.constants import (
    LEFT_TAXEL_SITE_NAMES,
    RIGHT_TAXEL_SITE_NAMES,
)

PAD_FT_SITE_NAMES = ("left_pad_ft_site", "right_pad_ft_site")
DEFAULT_FRAME_SCALE = 0.0035
PAD_FRAME_SCALE_FACTOR = 1.35
PAD_AXES_RADIUS_FACTOR = 0.045
TAXEL_AXES_RADIUS_FACTOR = 0.03
PAD_ORIGIN_RADIUS_FACTOR = 0.05
TAXEL_ORIGIN_RADIUS_FACTOR = 0.025


@dataclass(frozen=True)
class FrameOverlay:
    """保存单个 site 坐标系的可视化句柄."""

    site_id: int
    frame_handle: viser.FrameHandle


def _site_id(model: mujoco.MjModel, site_name: str) -> int:
    """按名称查找 site id."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Could not find site {site_name!r} in model.")
    return site_id


def _site_position(data: mujoco.MjData, site_id: int) -> np.ndarray:
    """返回 site 在世界坐标系中的平移."""
    return np.asarray(data.site_xpos[site_id], dtype=float).copy()


def _overlay_position(site_position: np.ndarray, scene_offset: np.ndarray) -> np.ndarray:
    """返回与 ``mjviser`` 当前场景偏移对齐后的可视化位置."""
    return site_position + scene_offset


def _site_rotation_matrix(data: mujoco.MjData, site_id: int) -> np.ndarray:
    """返回 site 在世界坐标系中的旋转矩阵."""
    return np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy()


def _wxyz_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 viser 使用的 ``wxyz`` 四元数."""
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _create_overlay(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    site_name: str,
    frame_scale: float,
    axes_radius_factor: float,
    origin_radius_factor: float,
) -> FrameOverlay:
    """创建单个 site 的坐标系句柄."""
    site_id = _site_id(model, site_name)
    frame_handle = server.scene.add_frame(
        f"/frames/{site_name}",
        axes_length=frame_scale,
        axes_radius=frame_scale * axes_radius_factor,
        origin_radius=frame_scale * origin_radius_factor,
        visible=True,
    )
    return FrameOverlay(
        site_id=site_id,
        frame_handle=frame_handle,
    )


def _update_overlay(
    overlay: FrameOverlay,
    data: mujoco.MjData,
    scene_offset: np.ndarray,
    visible: bool,
) -> None:
    """根据当前 ``MjData`` 更新坐标系位姿."""
    position = _site_position(data, overlay.site_id)
    rotation_matrix = _site_rotation_matrix(data, overlay.site_id)
    overlay.frame_handle.position = _overlay_position(position, scene_offset)
    overlay.frame_handle.wxyz = _wxyz_from_rotation_matrix(rotation_matrix)
    overlay.frame_handle.visible = visible


def _build_overlays(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    frame_scale: float,
) -> tuple[list[FrameOverlay], list[FrameOverlay]]:
    """构建 pad FT site 和 taxel site 的全部可视化句柄."""
    pad_overlays = [
        _create_overlay(
            server=server,
            model=model,
            site_name=site_name,
            frame_scale=frame_scale * PAD_FRAME_SCALE_FACTOR,
            axes_radius_factor=PAD_AXES_RADIUS_FACTOR,
            origin_radius_factor=PAD_ORIGIN_RADIUS_FACTOR,
        )
        for site_name in PAD_FT_SITE_NAMES
    ]
    taxel_overlays = [
        _create_overlay(
            server=server,
            model=model,
            site_name=site_name,
            frame_scale=frame_scale,
            axes_radius_factor=TAXEL_AXES_RADIUS_FACTOR,
            origin_radius_factor=TAXEL_ORIGIN_RADIUS_FACTOR,
        )
        for site_name in (*LEFT_TAXEL_SITE_NAMES, *RIGHT_TAXEL_SITE_NAMES)
    ]
    return pad_overlays, taxel_overlays


def main(
    xml: Annotated[
        Path,
        typer.Option(
            "--xml",
            help="要加载的 MJCF scene XML。",
        ),
    ] = PTS_SPHERES_SCENE_XML,
    show_pad_frames: Annotated[
        bool,
        typer.Option(
            "--show-pad-frames/--no-show-pad-frames",
            help="是否显示左右 pad FT site 的局部坐标系。",
        ),
    ] = True,
    show_taxel_frames: Annotated[
        bool,
        typer.Option(
            "--show-taxel-frames/--no-show-taxel-frames",
            help="是否显示左右 taxel site 的局部坐标系。",
        ),
    ] = True,
    frame_scale: Annotated[
        float,
        typer.Option(
            "--frame-scale",
            min=0.0,
            help="taxel site 坐标系轴长，pad 坐标系会按比例放大。",
        ),
    ] = DEFAULT_FRAME_SCALE,
) -> None:
    """启动 PTSSpheres site 坐标系浏览器可视化."""
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    server = viser.ViserServer()
    pad_overlays, taxel_overlays = _build_overlays(
        server=server,
        model=model,
        frame_scale=frame_scale,
    )

    def _render(scene: mjviser.ViserMujocoScene) -> None:
        """同步 MuJoCo 场景并刷新自定义 frame."""
        scene.update_from_mjdata(data)
        scene_offset = np.asarray(scene._scene_offset, dtype=float)
        for overlay in pad_overlays:
            _update_overlay(
                overlay=overlay,
                data=data,
                scene_offset=scene_offset,
                visible=show_pad_frames,
            )
        for overlay in taxel_overlays:
            _update_overlay(
                overlay=overlay,
                data=data,
                scene_offset=scene_offset,
                visible=show_taxel_frames,
            )

    typer.echo(f"Opening mjviser for {xml}")
    viewer = mjviser.Viewer(
        model=model,
        data=data,
        render_fn=_render,
        server=server,
    )
    viewer.run()


if __name__ == "__main__":
    typer.run(main)
