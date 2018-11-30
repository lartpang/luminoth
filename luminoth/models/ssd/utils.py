    import numpy as np
    import tensorflow as tf


    def adjust_bboxes(bboxes, old_height, old_width, new_height, new_width):
        """
        Adjusts the bboxes of an image that has been resized.
        对于输入的bboxes进行调整, bboxes / old * new

        Args:
            bboxes: Tensor with shape (num_bboxes, 4).
            old_height: Float. Height of the original image.
            old_width: Float. Width of the original image.
            new_height: Float. Height of the image after resizing.
            new_width: Float. Width of the image after resizing.
        Returns:
            Tensor with shape (num_bboxes, 4), with the adjusted bboxes.
        """
        # x_min, y_min, x_max, y_max = np.split(bboxes, 4, axis=1)
        x_min = bboxes[:, 0] / old_width
        y_min = bboxes[:, 1] / old_height
        x_max = bboxes[:, 2] / old_width
        y_max = bboxes[:, 3] / old_height

        # Use new size to scale back the bboxes points to absolute values.
        x_min = x_min * new_width
        y_min = y_min * new_height
        x_max = x_max * new_width
        y_max = y_max * new_height

        # Concat points and label to return a [num_bboxes, 4] tensor.
        return np.stack([x_min, y_min, x_max, y_max], axis=1)


    def generate_anchors_reference(ratios, scales, num_anchors, feature_map_shape):
        """
        Generate the default anchor for one feat map which we will later convolve
        to generate all the anchors of that feat map.
        生成一个特征图的左上角的一个格子里的默认锚点，我们稍后将进行卷积以生成该特征图的所有锚点

        anchor_reference = generate_anchors_reference(
            anchor_ratios, scales[i: i + 2],
            anchors_per_point[i], feat_map_shape
        )
        """
        heights = np.zeros(num_anchors)
        widths = np.zeros(num_anchors)

        # 先得到那个大的方形的anchor
        if len(scales) > 1:
            # 说明不是最后的特征图, 计算的时候, 先确定那个大框, 即 feature_map_weight*
            # sqrt(min_size, max_size)(特征图也是方形的)
            widths[0] = heights[0] = (np.sqrt(scales[0] * scales[1]) *
                                      feature_map_shape[0])
        # The last endpoint
        else:
            # The last layer doesn't have a subsequent layer with which
            # to generate the second scale from their geometric mean,
            # so we hard code it to 0.99.
            # We should add this parameter to the config eventually.
            # 最后一层没有后续层来得到需要的max_size, 所以这里直接编码为0.99
            heights[0] = scales[0] * feature_map_shape[0] * 0.99
            widths[0] = scales[0] * feature_map_shape[1] * 0.99

        # 这里的纵横比例实际上的数量是不包含前面这个大的方形anchor的, 但是num_anchors却包含了,
        # 所以这里对于ratios进行切片的时候, 要截止到num_anchors-1, 因为大方形anchor是一定有的
        # 而且小的纵横比也是一定有的
        ratios = ratios[:num_anchors - 1]
        heights[1:] = scales[0] / np.sqrt(ratios) * feature_map_shape[0]
        widths[1:] = scales[0] * np.sqrt(ratios) * feature_map_shape[1]

        # Each feature layer forms a grid on image space, so we
        # calculate the center point on the first cell of this grid.
        # Which we'll use as the center for our anchor reference.
        # The center will be the midpoint of the top left cell,
        # given that each cell is of 1x1 size, its center will be 0.5x0.5
        x_center = y_center = 0.5

        # Create anchor reference.
        # 这里的到anchors的(num_of_the_feature_map_anchors, 4)
        # [x_min, y_min, x_max, y_max]
        anchors = np.column_stack([
            x_center - widths / 2,
            y_center - heights / 2,
            x_center + widths / 2,
            y_center + heights / 2,
        ])

        return anchors


    def generate_raw_anchors(feature_maps, anchor_min_scale, anchor_max_scale,
                             anchor_ratios, anchors_per_point):
        """
        Returns a dictionary containing the anchors per feature map.
        返回所有提取出来的特征图的所有的anchors

        Returns:
        anchors: A dictionary with feature maps as keys and an array of anchors
            as values ('[[x_min, y_min, x_max, y_max], ...]') with shape
            (anchors_per_point[i] * endpoints_outputs[i][0]
             * endpoints_outputs[i][1], 4)
        """
        # TODO: Anchor generation needs heavy refactor

        # We interpolate the scales of the anchors from a min and a max scale
        # 线性插值, 从最小值到最大值, 这里保留了最大值, 总共的数量为len(feture_maps), 应该是6
        scales = np.linspace(anchor_min_scale, anchor_max_scale, len(feature_maps))

        anchors = {}
        # 对各个位置上的特征图进行遍历, 提取对应的特征图的大小(高,宽), 获取
        for i, (feat_map_name, feat_map) in enumerate(feature_maps.items()):
            feat_map_shape = feat_map.shape.as_list()[1:3]
            # ques: 若是i是0~5, 那这里对应的sacles最后的范围岂不是超界了?
            # ans: 超出去的元素就不存在了
            anchor_reference = generate_anchors_reference(
                anchor_ratios, scales[i: i + 2],
                anchors_per_point[i], feat_map_shape
            )
            # 生成每个特征图对应的所有anchors
            anchors[feat_map_name] = generate_anchors_per_feat_map(
                feat_map_shape, anchor_reference)

        return anchors


    def generate_anchors_per_feat_map(feature_map_shape, anchor_reference):
        """
        Generate anchor for an image.
        已获得的基于左上角的feature_map的grid的anchors, 进一步获得整个特征图上的所有的anchors

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            # 这五句指令, 实现了长宽范围内的数据的组合对应
            shift_x = np.arange(feature_map_shape[1])
            shift_y = np.arange(feature_map_shape[0])
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = np.reshape(shift_x, [-1])
            shift_y = np.reshape(shift_y, [-1])

            # 对数据进行一下打包, 得到的是(4, HxW)
            shifts = np.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )
            shifts = np.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor
            # 现在相当于得到了所有的grid的左上角的坐标

            # Expand dims to use broadcasting sum.
            # (1, num_grid_anchors, 4) + (HxW, 1, 4) = (HxW, num_grid_anchors, 4)
            all_anchors = (
                    np.expand_dims(anchor_reference, axis=0) +
                    np.expand_dims(shifts, axis=1)
            )
            # Flatten
            # (HxWxnum_grid_anchors, 4)
            return np.reshape(all_anchors, (-1, 4))
