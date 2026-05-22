"""从基础 2F-85 XML 生成 PTS sphere-taxel 版本.

本脚本的目标不是重新定义整套夹爪模型，而是在保留 ``2f85.xml`` 主体机构、
四连杆、tendon、equality 和 actuator 的前提下，只对左右 pad 区域注入：

- 3x3 sphere taxel 几何
- 每个 taxel 对应的 site
- 每个 taxel 对应的 builtin ``<force>`` sensor
- 每侧 pad 的全局 ``force`` / ``torque`` 参考 site 与 sensor

这样可以把 ``2f85_pts_spheres.xml`` 维护成"可重复生成的派生资产"，而不是只能手工编辑。
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from contactile_mjlab.paths import BASE_XML, PTS_SPHERES_XML


@dataclass(frozen=True)
class TaxelGridSpec:
    """描述规则 taxel 网格在 pad 局部坐标系中的参数.

    网格默认关于 y=0 居中对齐，第一列坐标由 ``cols`` 和 ``col_pitch`` 自动计算。
    """

    x: float
    col_pitch: float
    row_pitch: float
    rows: int
    cols: int

    @property
    def first_col_y(self) -> float:
        """关于 y=0 居中时第一列的中心 y 坐标."""
        return -0.5 * (self.cols - 1) * self.col_pitch


# `TAXEL_GRID` 描述 3x3 taxel patch 的中心布局。
#
# 这里所有坐标都在 `left_pad/right_pad` 的局部坐标系中定义：
# - `x`：沿 pad 法向的中心位置。值越大，sphere 越向 pad 外侧凸出，越容易先接触物体。
# - `x` / `col_pitch` / `row_pitch` / `rows` / `cols`：网格参数。
# - 第一列 ``y`` 关于 y=0 居中自动计算（见 ``TaxelGridSpec.first_col_y``）。
# - 第一行 ``z`` 由 pad 上边缘和"中心行到上边缘距离"自动反推（见 ``_first_row_z()``）。
TAXEL_GRID = TaxelGridSpec(
    x=0.0453,
    col_pitch=0.007,
    row_pitch=0.007,
    rows=3,
    cols=3,
)

# `2f85.xml` 中 pad 两段 box 的 z 向几何参数。
#
# 这里保留成显式常量，是为了让"上边缘距离"这类设计参数能直接映射到当前资产。
PAD_BOX1_POS_Z = 0.1200
PAD_BOX1_HALF_Z = 0.009375
PAD_BOX2_POS_Z = 0.13875
PAD_BOX2_HALF_Z = 0.009375

# 中间一行 taxel 中心到 pad 上边缘的距离。
#
# 当前设为 12 mm，表示 3x3 网格中 `10/11/12` 这一行的中心，
# 与 pad 最上缘之间沿 z 方向相距 0.012 m。
MIDDLE_ROW_TO_TOP_EDGE = 0.012

# 全局 force/torque site 的 x 坐标 —— 取 pad box 几何表面，而非 taxel 外凸位置。
_PAD_FT_SITE_X = 0.043258

# sphere geom 的半径。
#
# 当前 `size` 对应 MuJoCo sphere 的半径而不是直径。
# 修改半径时要和 `col_pitch` / `row_pitch` 一起看：
# - 若希望相邻 sphere 不重叠，通常满足 `2 * TAXEL_RADIUS <= pitch`
# - 若希望接触面更连续，可以允许一定程度的重叠
TAXEL_RADIUS = 0.0028

# taxel site 的可视化半径。
#
# 对当前 builtin `<force>` sensor 来说，主要影响 viewer 中的 site 显示尺寸，
# 不直接改变 sphere geom 的真实接触形状。
TAXEL_SITE_RADIUS = 0.0008
TAXEL_GEOM_KWARGS = {
    "type": "sphere",
    "size": f"{TAXEL_RADIUS:.4f}",
    "mass": "1e-6",
    "friction": "0.7 0.03 0.01",
    "solimp": "0.95 0.99 0.001",
    "solref": "0.004 1",
    "priority": "2",
}
# taxel site 坐标系绕 Y 轴旋转 -90°，使：
#   site.Z = -pad.X (接触法向，向里)
#   site.X = +pad.Z (沿 pad 长度，指向指尖)
#   site.Y = +pad.Y (横向，右手法则)
TAXEL_SITE_QUAT = "1 0 -1 0"
TAXEL_SITE_KWARGS = {
    "size": f"{TAXEL_SITE_RADIUS:.4f}",
    "quat": TAXEL_SITE_QUAT,
}
SIDE_COLOR_BY_NAME = {
    "left": {
        "geom": "0 1 0 0.45",
        "site": "0 1 0 0.8",
    },
    "right": {
        "geom": "0 0 1 0.45",
        "site": "0 0 1 0.8",
    },
}


def _format_xyz(x: float, y: float, z: float) -> str:
    """将三维坐标格式化为 MuJoCo XML 常用字符串."""
    return f"{x:.4f} {y:.4f} {z:.4f}"


def _pad_top_edge_z() -> float:
    """返回当前 pad 几何在局部坐标系中的最上边缘 z 坐标."""
    return max(PAD_BOX1_POS_Z + PAD_BOX1_HALF_Z, PAD_BOX2_POS_Z + PAD_BOX2_HALF_Z)


def _grid_center_z() -> float:
    """返回 taxel 网格中心 (中间行) 的 z 坐标."""
    return _pad_top_edge_z() - MIDDLE_ROW_TO_TOP_EDGE


def _pad_ft_site_pos() -> str:
    """返回 pad 全局 force/torque site 的坐标字符串，位于网格中心."""
    return _format_xyz(_PAD_FT_SITE_X, 0, _grid_center_z())


def _first_row_z(grid: TaxelGridSpec) -> float:
    """根据网格中心 z 坐标反推第一行中心的 z 坐标.

    Args:
        grid: 规则 taxel 网格参数。

    Returns:
        第一行 taxel 中心的 z 坐标。

    Raises:
        ValueError: 当行数不是奇数时抛出。
    """
    if grid.rows % 2 != 1:
        raise ValueError("Current middle-row distance parameterization requires an odd row count.")

    middle_row_index = grid.rows // 2
    return _grid_center_z() - middle_row_index * grid.row_pitch


def _iter_taxel_layout(grid: TaxelGridSpec) -> Iterable[tuple[str, str]]:
    """按行优先顺序生成 taxel 索引和中心坐标.

    Args:
        grid: 规则 taxel 网格参数。

    Yields:
        ``(index, pos)`` 二元组，其中 ``index`` 形如 ``"12"``，
        ``pos`` 是 MuJoCo XML 使用的 ``"x y z"`` 字符串。
    """
    first_row_z = _first_row_z(grid)
    for row in range(grid.rows):
        for col in range(grid.cols):
            index = f"{row}{col}"
            y = grid.first_col_y + col * grid.col_pitch
            z = first_row_z + row * grid.row_pitch
            yield index, _format_xyz(grid.x, y, z)


def _parse_xml(path: Path) -> ET.ElementTree:
    """解析 XML 并保留注释节点."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _find_pad_body(root: ET.Element, side: str) -> ET.Element:
    """定位左右 pad body.

    Args:
        root: 待修改 XML 的根节点。
        side: ``left`` 或 ``right``。

    Returns:
        对应 pad body 的 XML 节点。

    Raises:
        ValueError: 当基础 XML 中找不到对应 pad body 时抛出。
    """
    pad_body = root.find(f".//body[@name='{side}_pad']")
    if pad_body is None:
        raise ValueError(f"Could not find pad body for side={side!r} in base XML.")
    return pad_body


def _ensure_size_limits(root: ET.Element) -> None:
    """为接触数量扩展后的模型补充 ``<size>`` 配置."""
    size_element = root.find("size")
    if size_element is not None:
        size_element.set("njmax", "128")
        size_element.set("nconmax", "256")
        return

    option_index = next(
        (index for index, child in enumerate(root) if child.tag == "option"),
        None,
    )
    if option_index is None:
        raise ValueError("Could not find <option> element in base XML.")

    root.insert(option_index + 1, ET.Element("size", {"njmax": "128", "nconmax": "256"}))


def _append_taxels_to_pad(pad_body: ET.Element, side: str) -> None:
    """向单侧 pad body 追加 3x3 sphere taxel 布局."""
    colors = SIDE_COLOR_BY_NAME[side]
    for index, pos in _iter_taxel_layout(TAXEL_GRID):
        taxel_body = ET.SubElement(
            pad_body,
            "body",
            {
                "name": f"{side}_taxel_body_{index}",
                "pos": pos,
            },
        )
        ET.SubElement(
            taxel_body,
            "geom",
            {
                "name": f"{side}_taxel_geom_{index}",
                "rgba": colors["geom"],
                **TAXEL_GEOM_KWARGS,
            },
        )
        ET.SubElement(
            taxel_body,
            "site",
            {
                "name": f"{side}_taxel_site_{index}",
                "rgba": colors["site"],
                **TAXEL_SITE_KWARGS,
            },
        )

    ET.SubElement(
        pad_body,
        "site",
        {
            "name": f"{side}_pad_ft_site",
            "pos": _pad_ft_site_pos(),
            "quat": TAXEL_SITE_QUAT,
            "size": "0.003",
            "rgba": "1 0 0 0.5",
        },
    )


def _iter_taxel_force_sensors(side: str) -> Iterable[ET.Element]:
    """按行优先顺序生成单侧 taxel force sensor 节点."""
    for index, _ in _iter_taxel_layout(TAXEL_GRID):
        yield ET.Element(
            "force",
            {
                "name": f"{side}_taxel_force_{index}",
                "site": f"{side}_taxel_site_{index}",
            },
        )


def _build_sensor_block() -> ET.Element:
    """构造 PTS 版本需要的全部传感器定义."""
    sensor = ET.Element("sensor")
    for side in ("left", "right"):
        for element in _iter_taxel_force_sensors(side):
            sensor.append(element)

    sensor.append(ET.Element("force", {"name": "left_pad_force", "site": "left_pad_ft_site"}))
    sensor.append(ET.Element("torque", {"name": "left_pad_torque", "site": "left_pad_ft_site"}))
    sensor.append(ET.Element("force", {"name": "right_pad_force", "site": "right_pad_ft_site"}))
    sensor.append(ET.Element("torque", {"name": "right_pad_torque", "site": "right_pad_ft_site"}))
    return sensor


def build_pts_spheres_tree(base_xml: Path) -> ET.ElementTree:
    """从基础 ``2f85.xml`` 构造 PTS sphere-taxel XML 树.

    Args:
        base_xml: 原始 Robotiq 2F-85 XML 路径。

    Returns:
        已插入 taxel 和 sensor 的 XML 树。
    """
    tree = _parse_xml(base_xml)
    root = tree.getroot()
    root.set("model", "Dual_wrist_camera_pts_spheres")
    _ensure_size_limits(root)

    for side in ("left", "right"):
        _append_taxels_to_pad(_find_pad_body(root, side), side)

    root.append(_build_sensor_block())
    return tree


def write_tree(tree: ET.ElementTree, output_xml: Path) -> None:
    """将生成结果格式化写入目标 XML."""
    ET.indent(tree, space="  ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    xml_text = ET.tostring(tree.getroot(), encoding="unicode", short_empty_elements=True)
    output_xml.write_text(f"{xml_text}\n", encoding="utf-8")


def main() -> None:
    """从基础 2F-85 XML 生成 PTS sphere-taxel 派生文件."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-xml", type=Path, default=BASE_XML)
    parser.add_argument("--output-xml", type=Path, default=PTS_SPHERES_XML)
    args = parser.parse_args()

    tree = build_pts_spheres_tree(args.base_xml)
    write_tree(tree, args.output_xml)
    print(f"Generated {args.output_xml} from {args.base_xml}")


if __name__ == "__main__":
    main()
