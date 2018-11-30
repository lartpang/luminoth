import sonnet as snt
import tensorflow as tf
from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import encode


class RCNNTarget(snt.AbstractModule):
    """Generate RCNN target tensors for both probabilities and bounding boxes.

    Targets for RCNN are based upon the results of the RPN, this can get tricky
    in the sense that RPN results might not be the best and it might not be
    possible to have the ideal amount of targets for all the available ground
    truth boxes.
    RCNN的目标是基于RPN的结果，这可能会变得棘手，因为RPN结果可能不是最好的，并且可能无法为所
    有可用的地面实况框提供理想数量的目标。

    There are two types of targets, class targets and bounding box targets.

    Class targets are used both for background and foreground, while bounding
    box targets are only used for foreground (since it's not possible to create
    a bounding box of "background objects").
    类别目标用在前景背景上, 而边界框目标只是用在前景上(因为背景不会创建边界框)

    A minibatch size determines how many targets are going to be generated and
    how many are going to be ignored. RCNNTarget is responsible for choosing
    which proposals and corresponding targets are included in the minibatch and
    which ones are completely ignored.
    minibatch的大小决定了有多少目标将要被生成和忽略, RCNNTarget主要用来选择哪个提案和对应的
    目标包含在minibatch或者被忽略

    这个类的返回值, 可以看_build来了解, 主要是(num_proposals, 1)的proposals_label, 以及
    (num_proposals, 4)的bbox_targets, 前者是表述平衡过后的提案中各个提案的类别标定(0~20)
    , 后者则表述留下来的提案中的前景提案所对应的最接近的真实框的偏移量和缩放量
    """

    def __init__(self, num_classes, config, seed=None, variances=None,
                 name='rcnn_proposal'):
        """
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(RCNNTarget, self).__init__(name=name)
        self._num_classes = num_classes
        self._variances = variances
        # Ratio of foreground vs background for the minibatch.
        self._foreground_fraction = config.foreground_fraction
        self._minibatch_size = config.minibatch_size
        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low
        self._seed = seed

    def _build(self, proposals, gt_boxes):
        """
        Args:
            proposals: A Tensor with the RPN bounding boxes proposals.
                The shape of the Tensor is (num_proposals, 4).
                RPN得出的边界框提案
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
                真实的边界框提案
        Returns:
            proposals_label: Either a truth value of the proposals (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the proposal is to be ignored in the minibatch.
                The shape of the Tensor is (num_proposals, 1).
                对于每个提案, 返回的是0~类别数目之间的值, 表示对应的类别, -1表示忽略的提案
                对于这个结果, 实际上已经考虑了minibatch的内部的而正负样本之间的平衡的问题
            bbox_targets: A bounding box regression target for each of the
                proposals that have and greater than zero label. For every
                other proposal we return zeros.
                The shape of the Tensor is (num_proposals, 4).
                返回每个有着大于0标签的提案的边界框回归目标, 其他的返回0.
                在前景提案的位置上更新与自身最好的真实框与前景提案之间的偏移量和缩放连量(4个
                值), 其余为0
        """
        # 计算IoU (num_proposals, num_gt_boxes)
        overlaps = bbox_overlap_tf(proposals, gt_boxes[:, :4])
        # overlaps now contains (num_proposals, num_gt_boxes) with the IoU of
        # proposal P and ground truth box G in overlaps[P, G]

        # We are going to label each proposal based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # ignored.
        # tf.gather根据索引获取目标值组成的张量
        # (num_proposals, 4) -> [num_proposals]
        proposals_label_shape = tf.gather(tf.shape(proposals), [0])
        # (num_proposals, ) x -1
        proposals_label = tf.fill(
            dims=proposals_label_shape,
            value=-1.
        )
        # For each overlap there is three possible outcomes for labelling:
        #  if max(iou) < config.background_threshold_low then we ignore.
        #  elif max(iou) <= config.background_threshold_high then we label
        #      background.
        #  elif max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #
        # max_overlaps gets, for each proposal, the index in which we can
        # find the gt_box with which it has the highest overlap.
        # (num_proposals, ) <= (num_proposals, num_gt_boxes)
        # 得到对于每个提案各自与所有真实框之间的最大的IoU
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        iou_is_high_enough_for_bg = tf.greater_equal(
            max_overlaps, self._background_threshold_low
        )
        iou_is_not_too_high_for_bg = tf.less(
            max_overlaps, self._background_threshold_high
        )

        # 获得背景提案集合
        bg_condition = tf.logical_and(
            iou_is_high_enough_for_bg, iou_is_not_too_high_for_bg
        )

        # 背景提案的位置对应的标签为0, 其余的保持原样, 此时为-1
        proposals_label = tf.where(
            condition=bg_condition,
            x=tf.zeros_like(proposals_label, dtype=tf.float32),
            y=proposals_label
        )

        # Get the index of the best gt_box for each proposal.
        # 得到对于每个提案而言最好的IoU的真实框的索引
        # (num_proposals, ) <= (num_proposals, num_gt_boxes)
        overlaps_best_gt_idxs = tf.argmax(overlaps, axis=1)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum it one because 0 is used for
        # background.
        # 对于每个提案, 最好的真实框的类别标签, 对于框的类别标签都要加上一个1, 为了给背景腾出
        # 来一个0标签
        # (num_proposals, ) -> (num_overlaps_best_gt_idxs, )
        best_fg_labels_for_proposals = tf.add(
            tf.gather(gt_boxes[:, 4], overlaps_best_gt_idxs),
            1.
        )

        # 获取前景
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )
        # 获取每个真实框对应的最为接近的提案的索引
        # (num_gt_boxes, 1) <= (num_proposals, num_gt_boxes)
        best_proposals_idxs = tf.argmax(overlaps, axis=0)

        # Set the indices in best_proposals_idxs to True, and the rest to
        # false.
        # tf.sparse_to_dense is used because we know the set of indices which
        # we want to set to True, and we know the rest of the indices
        # should be set to False. That's exactly the use case of
        # tf.sparse_to_dense.
        # sparse_to_dense表示的就是在output_shape大小(num_proposal)的张量上, 设定默认
        # 值为default_value, 而在sparse_indices对应的位置上, 设定为sparse_values
        # 这里也就是将原本的提案中的被真实框有着最好的对应的几个框的位置标定位True
        is_best_box = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_proposals_idxs, [-1]),
            sparse_values=True, default_value=False,
            output_shape=tf.cast(proposals_label_shape, tf.int64),
            validate_indices=False
        )

        # 将每个前景框对应的最好的真实框的类别更新到提案框的标签(num_proposals, )中
        # We update proposals_label with the value in
        # best_fg_labels_for_proposals only when the box is foreground.
        proposals_label = tf.where(
            condition=iou_is_fg,
            x=best_fg_labels_for_proposals,
            y=proposals_label
        )
        # Now we need to find the proposals that are the best for each of the
        # gt_boxes. We overwrite the previous proposals_label with this
        # because setting the best proposal for each gt_box has priority.
        # 下面两个函数实现了对于每个真实框对应的最好的提案的位置上更新对应的类别标签

        # 这里实现了对proposals_label_shape的best_proposals_idxs(
        # 对于每个真实框对应的最好的提案)位置上更新为真实类别+1
        # 其余位置置零
        # 挑选提案, 要使用每个真实框对应的最好的提案框
        best_proposals_gt_labels = tf.sparse_to_dense(
            sparse_indices=tf.reshape(best_proposals_idxs, [-1]),
            sparse_values=gt_boxes[:, 4] + 1,
            default_value=0.,
            output_shape=tf.cast(proposals_label_shape, tf.int64),
            validate_indices=False,
            name="get_right_labels_for_bestboxes"
        )
        # 对每个真实框对应的最好的提案的位置上更新标签
        proposals_label = tf.where(
            condition=is_best_box,
            x=best_proposals_gt_labels,
            y=proposals_label,
            name="update_labels_for_bestbox_proposals"
        )

        # proposals_label now has a value in [0, num_classes + 1] for
        # proposals we are going to use and -1 for the ones we should ignore.
        # But we still need to make sure we don't have a number of proposals
        # higher than minibatch_size * foreground_fraction.
        # 在进行确定要被忽略的提案之前, 先要确定正负样本够不够, 比例合不合适
        max_fg = int(self._foreground_fraction * self._minibatch_size)
        # 所谓前景: 每个真实框对应的最好的边界框, 以及IoU最大的边界框
        # 所以这里或操作, 实现了一个合并
        fg_condition = tf.logical_or(
            iou_is_fg, is_best_box
        )
        # 获得前景的索引
        fg_inds = tf.where(
            condition=fg_condition
        )

        # 删除数量超出比例的前景
        def disable_some_fgs():
            # We want to delete a randomly-selected subset of fg_inds of
            # size `fg_inds.shape[0] - max_fg`.
            # We shuffle along the dimension 0 and then we get the first
            # num_fg_inds - max_fg indices and we disable them.
            shuffled_inds = tf.random_shuffle(fg_inds, seed=self._seed)
            disable_place = (tf.shape(fg_inds)[0] - max_fg)
            # This function should never run if num_fg_inds <= max_fg, so we
            # add an assertion to catch the wrong behaviour if it happens.
            integrity_assertion = tf.assert_positive(
                disable_place,
                message="disable_place in disable_some_fgs is negative."
            )
            with tf.control_dependencies([integrity_assertion]):
                disable_inds = shuffled_inds[:disable_place]
            is_disabled = tf.sparse_to_dense(
                sparse_indices=disable_inds,
                sparse_values=True, default_value=False,
                output_shape=tf.cast(proposals_label_shape, tf.int64),
                # We are shuffling the indices, so they may not be ordered.
                validate_indices=False
            )

            # 要是被忽略的话, 那就直接标签进行取反就可以
            return tf.where(
                condition=is_disabled,
                # We set it to -label for debugging purposes.
                x=tf.negative(proposals_label),
                y=proposals_label
            )

        # Disable some fgs if we have too many foregrounds.
        proposals_label = tf.cond(
            tf.greater(tf.shape(fg_inds)[0], max_fg),
            true_fn=disable_some_fgs,
            false_fn=lambda: proposals_label
        )

        # 确定所有的前景的数量
        total_fg_in_batch = tf.shape(
            tf.where(
                condition=tf.greater(proposals_label, 0)
            )
        )[0]

        # Now we want to do the same for backgrounds.
        # We calculate up to how many backgrounds we desire based on the
        # final number of foregrounds and the total desired batch size.
        max_bg = self._minibatch_size - total_fg_in_batch

        # We can't use bg_condition because some of the proposals that satisfy
        # the IoU conditions to be background may have been labeled as
        # foreground due to them being the best proposal for a certain gt_box.
        bg_mask = tf.equal(proposals_label, 0)
        bg_inds = tf.where(
            condition=bg_mask,
        )

        def disable_some_bgs():
            # Mutatis mutandis, all comments from disable_some_fgs apply.
            shuffled_inds = tf.random_shuffle(bg_inds, seed=self._seed)
            disable_place = (tf.shape(bg_inds)[0] - max_bg)
            integrity_assertion = tf.assert_non_negative(
                disable_place,
                message="disable_place in disable_some_bgs is negative."
            )
            with tf.control_dependencies([integrity_assertion]):
                disable_inds = shuffled_inds[:disable_place]
            is_disabled = tf.sparse_to_dense(
                sparse_indices=disable_inds,
                sparse_values=True, default_value=False,
                output_shape=tf.cast(proposals_label_shape, tf.int64),
                validate_indices=False
            )
            return tf.where(
                condition=is_disabled,
                x=tf.fill(
                    dims=proposals_label_shape,
                    value=-1.
                ),
                y=proposals_label
            )

        proposals_label = tf.cond(
            tf.greater_equal(tf.shape(bg_inds)[0], max_bg),
            true_fn=disable_some_bgs,
            false_fn=lambda: proposals_label
        )

        # Next step is to calculate the proper targets for the proposals labeled
        # based on the values of the ground-truth boxes.
        # We have to use only the proposals labeled >= 1, each matching with
        # the proper gt_boxes
        # 接下来基于真实边界框, 对于标定的预测边界框计算更为合适的target
        # 只需要计算标定值大于等于1(非背景, 未被忽略的提案), 每一个都匹配一个更为合适的真实框

        # Get the ids of the proposals that matter for bbox_target comparisson.
        # 获得前景提案的逻辑索引
        is_proposal_with_target = tf.greater(
            proposals_label, 0
        )
        # 获得前景提案的坐标索引
        proposals_with_target_idx = tf.where(
            condition=is_proposal_with_target
        )

        # Get the corresponding ground truth box only for the proposals with
        # target.
        # 根据前面得到前景提案的索引, 从对于每个提案而言最好的真实框索引中索引数据
        # overlaps_best_gt_idxs (num_proposals, )
        gt_boxes_idxs = tf.gather(
            overlaps_best_gt_idxs,
            proposals_with_target_idx
        )
        # Get the values of the ground truth boxes.
        # 根据索引获得对于每个前景提案而言最好的真实框的数据
        # gather_nd支持对多维的索引
        proposals_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # 这里相当于就是索引前景提案
        # We create the same array but with the proposals
        # proposal (num_proposals, 4), 这样的索引才可以真正保留原坐标的格式
        proposals_with_target = tf.gather_nd(
            proposals,
            proposals_with_target_idx
        )
        # We create our targets with bbox_transform.
        # 计算proposals_gt_boxes与proposals_with_target的相对的偏移量和缩放量
        # 也就是计算对于每个前景提案而言最好的真实框与前景提案之间的偏移量和缩放连量
        bbox_targets_nonzero = encode(
            proposals_with_target,
            proposals_gt_boxes,
            variances=self._variances,
        )

        # We unmap targets to proposal_labels (containing the length of
        # proposals)
        # 使用indices在zeros(update)的矩阵上对应的位置更新数据update
        # 这里的结果就是将前景提案的对应位置上, 更新与自身最好的真实框与前景提案之间的偏移量和
        # 缩放连量
        bbox_targets = tf.scatter_nd(
            indices=proposals_with_target_idx,
            updates=bbox_targets_nonzero,
            shape=tf.cast(tf.shape(proposals), tf.int64)
        )

        proposals_label = proposals_label
        bbox_targets = bbox_targets

        return proposals_label, bbox_targets
