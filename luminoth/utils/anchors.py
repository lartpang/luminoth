import numpy as np


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base anchor to be used as reference of generating all anchors.
    给定基础大小，纵横比，面积缩放比，返回所有组合的anchors的偏移中心的距离
    (total_aspect_ratios * total_scales, 4) -> (x_min, y_min, x_max, y_max)

    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the wanted widths and heights.

    Scales apply to area of object.

    Args:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to area.

        1. Base size to use for anchors.
        base_size: 256
        2. Scale used for generating anchor sizes.
        scales: [0.25, 0.5, 1, 2]
        3. Aspect ratios used for generating anchors.
        aspect_ratios: [0.5, 1, 2]

    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    # 这里的aspect_ratio_sqrts表示的是纵横比例的开方，在每个base_sacles * base_size之后乘除计算，得
    # 到的就是固定面积下，不同纵横比的记过。
    # 这里先使用了meshgrid，来构建面积缩放比例与纵横比的组合关系，确实很妙。
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    # 组合得到各个anchors的左上角和右下角的坐标相对于中心坐标的偏移量
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )

    return anchors
