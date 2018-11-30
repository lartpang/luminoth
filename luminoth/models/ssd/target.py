import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import encode


class SSDTarget(snt.AbstractModule):
    """
    Generate SSD target tensors for both probabilities and bounding boxes.
    对于概率和边界框, 生成SSD目标张量, 返回对应anchors的类别标签(0~21)和真实框相对于前景
    anchors自身坐标的偏移量和缩放量

    There are two types of targets, anchor_label and bounding box targets.
    targets有两种类型, 一种是anchor的标签, 一种是边界框targets

    Anchor labels are just the label which best fits each anchor, and therefore
    are the target for that anchor, they are used both for background and
    foreground labels.
    anchor的label表示最匹配anchor的标签, 通常使用背景和前景标签

    Bounding box targets are just the encoded coordinates that anchors labeled
    as foreground should target.
    边界框标签编码被标记为前景的anchors的坐标
    """

    def __init__(self, num_classes, config, variances, seed=None,
                 name='ssd_target'):
        """
        构造类的实例的时候调用
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(SSDTarget, self).__init__(name=name)
        self._num_classes = num_classes
        self._hard_negative_ratio = config.hard_negative_ratio
        self._foreground_threshold = config.foreground_threshold
        self._background_threshold_high = config.background_threshold_high
        self._variances = variances
        self._seed = seed

    def _build(self, probs, all_anchors, gt_boxes):
        """
        在向类的实例传值的时候调用
        Args:
            probs: 这里包含了背景的类别, 所以是 num_classes+1
            all_anchors: A Tensor with anchors for all of SSD's features.
                The shape of the Tensor is (num_anchors, 4).
                所有的anchors的原图上的坐标结果
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
        Returns:
            class_targets: Either a truth value of the anchor (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the anchor is to be ignored in the minibatch.
                The shape of the Tensor is (num_anchors, 1).
                返回各个anchor对应的类别标签
            bbox_offsets_targets: A bounding box regression target for each of
                the anchors that have a greater than zero label. For every
                other anchors we return zeros.
                The shape of the Tensor is (num_anchors, 4).
                返回各个前景anchor对应的坐标偏移量, 其余的返回0
                在all_anchors中前景anchors的位置上更新对应的真实框相对于自身坐标的偏移
                量和缩放量
        """

        all_anchors = tf.cast(all_anchors, tf.float32)
        gt_boxes = tf.cast(gt_boxes, tf.float32)

        # We are going to label each anchor based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # unknown.
        # (num_anchors, 1)
        anchors_label_shape = tf.gather(tf.shape(all_anchors), [0])

        # [-1] ###############################################################

        # -1 * (num_anchors, 1)
        anchors_label = tf.fill(
            dims=anchors_label_shape,
            value=-1.
        )

        # (num_anchors, num_gt)
        overlaps = bbox_overlap_tf(all_anchors, gt_boxes[:, :4])
        # (num_anchors, )
        # 对于每个eanchor和所有真实框的IoU的 最大IoU值
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        #######################################################################
        # 这里开始从anchors的角度来思考, 考虑和它最好的真实框所对应的IoU, 超过阈值,
        # anchors就作为正样本
        #######################################################################

        # Get the index of the best gt_box for each anchor.
        # 对于每个anchor最为接近的真实框
        # (num_anchors, ), 每个元素表示真实框的 对应序号
        best_gtbox_for_anchors_idx = tf.argmax(overlaps, axis=1)

        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum 1 to it because 0 is used for
        # background.
        # 在对于每个anchor最为接近的真实框的类别标签(0~20)上加1, 作为这些anchors的标签
        # (num_anchors, 4)
        best_fg_labels_for_anchors = tf.add(
            tf.gather(gt_boxes[:, 4], best_gtbox_for_anchors_idx),
            1.
        )
        # (num_anchors, ) 依据每个anchors对应的最大的IoU值, 确定前景anchors为true
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )

        # [-1] =====> [-1, 1~20(前景anchor)] ##################################

        # We update anchors_label with the value in
        # best_fg_labels_for_anchors only when the box is foreground.
        # TODO: Replace with a sparse_to_dense with -1 default_value
        # 从前景anchor中将确定的最好的真实框的标签设定为anchors的标签, 其余保持-1不变
        anchors_label = tf.where(
            condition=iou_is_fg,
            x=best_fg_labels_for_anchors,
            y=anchors_label
        )

        #######################################################################
        # 这里开始从真实框的角度来思考, 防止有真实框没有对应的anchors, 所以要考虑和真实框对
        # 应的最好的anchors作为正样本
        #######################################################################

        # (num_gt, ) 对于每个真实框而言, 最好的anchor的位置
        best_anchor_idxs = tf.argmax(overlaps, axis=0)
        # 使用得到的anchors的位置, 生成一个稀疏张量, 大小为(num_anchors, ),
        # 有真实框对应的anchors位置上为True, 这是最起码的正样本
        # 这里为后面的tf.where实际上创造了一个条件张量
        is_best_box = tf.sparse_to_dense(
            sparse_indices=best_anchor_idxs,
            sparse_values=True, default_value=False,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False
        )

        # Now we need to find the anchors that are the best for each of the
        # gt_boxes. We overwrite the previous anchors_label with this
        # because setting the best anchor for each gt_box has priority.
        # 这里与上面基本类似, 只不过这里是在对应的位置上标记类别标签
        best_anchors_gt_labels = tf.sparse_to_dense(
            sparse_indices=best_anchor_idxs,
            sparse_values=gt_boxes[:, 4] + 1,
            default_value=-1,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False,
            name="get_right_labels_for_bestboxes"
        )

        # [-1, 1~20(前景anchor)] =====> [-1, 1~20(+对于每个真实框最接近的anchor)]

        # 修改anchors_label中, 每个真实框对应的最好的anchor的标签为对应的类别
        # 注意, 到这里的时候, 可能会觉得存在一个anchors会对应多个类别, 但是没关系, 这里是一
        # 个更新操作, 这里的优先级更高, 可以覆盖之前的判定
        anchors_label = tf.where(
            condition=is_best_box,
            x=best_anchors_gt_labels,
            y=anchors_label,
            name="update_labels_for_bestbox_anchors"
        )

        # Use the worst backgrounds (the bgs whose probability of being fg is
        # the greatest).
        # (num_anchors, (num_classes+1)[1:]), 选择各个anchors的前景类别的对应概率
        cls_probs = probs[:, 1:]
        # 得到所有anchors的针对各个前景类别的最大概率
        max_cls_probs = tf.reduce_max(cls_probs, axis=1)

        # Exclude boxes with IOU > `background_threshold_high` with any GT.
        # 最终被认定为背景的anchors, 是和所有真实框的最大IoU值小于背景阈值(0.2), 而且又是
        # 标签被标定为小于等于0的anchors
        # 标签小于等于0, 实际上就是标签小于0, 因为标签为0尚未确定
        iou_less_than_bg_tresh_high_filter = tf.less_equal(
            max_overlaps, self._background_threshold_high
        )

        # 这里确定了没有被通过IoU来判定为前景类别的anchors, 从中选择阈值小于背景上限阈值
        # 的, 作为后续的操作对象
        bg_anchors = tf.less_equal(anchors_label, 0)
        bg_overlaps_filter = tf.logical_and(
            iou_less_than_bg_tresh_high_filter, bg_anchors
        )

        # 在非前景anchors中选择和真实框的IoU小于阈值的, 在其位置上, 保留其针对各个前景类
        # 别的最大概率, 留作后面选择背景anchors用, 其余的标记为 -1
        # ques: 这里满足上面的条件的应该是对应的负样本/背景了呀, 怎么还保留可能的概率呢?
        # ans: 这里用作背景的anchors实际上是选择有着较大分类概率, 但是不接近真实框而且还
        #   标签小于-1的anchors
        max_cls_probs = tf.where(
            condition=bg_overlaps_filter,
            x=max_cls_probs,
            y=tf.fill(dims=anchors_label_shape, value=-1.),
        )

        # We calculate up to how many backgrounds we desire based on the
        # final number of foregrounds and the hard minning ratio.
        # 两句指令得到前景anchors数量
        num_fg_mask = tf.greater(anchors_label, 0.0)
        num_fg = tf.cast(tf.count_nonzero(num_fg_mask), tf.float32)

        # 得到背景数量=3*num_fg
        num_bg = tf.cast(num_fg * self._hard_negative_ratio, tf.int32)
        # 从max_clas_prob里选择前num_bg(各个类别概率最大值)的anchors作为背景anchors
        # 索引
        top_k_bg = tf.nn.top_k(max_cls_probs, k=num_bg)
        # 将对应的anchors位置标定位true, 这里当做下面的一个条件
        set_bg = tf.sparse_to_dense(
            sparse_indices=top_k_bg.indices,
            sparse_values=True, default_value=False,
            output_shape=anchors_label_shape,
            validate_indices=False
        )

        # [-1, 1~20(+对于每个真实框最接近的anchor)] =====> [-1, 0, 1~20] #########

        # 设定背景标签0
        anchors_label = tf.where(
            condition=set_bg,
            x=tf.fill(dims=anchors_label_shape, value=0.),
            y=anchors_label
        )

        # Next step is to calculate the proper bbox targets for the labeled
        # anchors based on the values of the ground-truth boxes.
        # We have to use only the anchors labeled >= 1, each matching with
        # the proper gt_boxes

        # Get the ids of the anchors that mater for bbox_target comparison.
        # 只针对前景anchors
        is_anchor_with_target = tf.greater(
            anchors_label, 0
        )
        anchors_with_target_idx = tf.where(
            condition=is_anchor_with_target
        )

        # Get the corresponding ground truth box only for the anchors with
        # target.
        # 从每个anchors对应的最好的真实框索引中, 选择所有前景anchors对应的真实框索引, 进而
        # 确定对应的真实框坐标
        gt_boxes_idxs = tf.gather(
            best_gtbox_for_anchors_idx,
            anchors_with_target_idx
        )
        # Get the values of the ground truth boxes.
        anchors_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # We create the same array but with the anchors
        # 确定所有前景anchors的对应的anchor在原图的坐标
        anchors_with_target = tf.gather_nd(
            all_anchors,
            anchors_with_target_idx
        )

        # We create our targets with bbox_transform
        # 获取所有前景anchors对应的真实框相对于自身坐标的偏移量和缩放量
        bbox_targets = encode(
            anchors_with_target,
            anchors_gt_boxes,
            variances=self._variances
        )

        # We unmap targets to anchor_labels (containing the length of
        # anchors)
        # 在all_anchors中前景anchors的位置上更新对应的bbox_targets
        bbox_targets = tf.scatter_nd(
            indices=anchors_with_target_idx,
            updates=bbox_targets,
            shape=tf.cast(tf.shape(all_anchors), tf.int64)
        )

        return anchors_label, bbox_targets
