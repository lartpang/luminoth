import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_overlap import bbox_overlap_tf
from luminoth.utils.bbox_transform_tf import encode as encode_tf


class RPNTarget(snt.AbstractModule):
    """RPNTarget: Get RPN's classification and regression targets.
    确定anchors的标签和对应的回归值

    RPNTarget is responsible for:
      * calculating the correct values for both classification and regression
        problems.
        计算分类和回归问题的正确的值
      * defining which anchors and target values are going to be used for the
        RPN minibatch.
        定义哪些anchors和目标将要用来被RPN minibatch使用

    For calculating the correct values for classification (ie. the question of
    "does this anchor refer to an object?") and returning an objectness score,
    we calculate the intersection over union (IoU) between the anchors boxes
    and the ground truth boxes, and use this to categorize anchors. When the
    intersection between anchors and groundtruth is above a threshold, we can
    mark the anchor as an object or as being foreground. In case of not having
    any intersection or having a low IoU value, then we say that the anchor
    refers to background.
    用于计算分类的正确值(例如这个anchor是否指向一个目标?)并且返回目标得分, 我们计算anchors与
    真实值之间的IoU, 并且使用这个来对anchors进行分类, 当这个交集超过一个阈值的时候, 我们标记
    anchor为目标或者是前景. 如果没有任何交集, 或者低的IoU值, 那么我们认为anchor是背景

    For calculating the correct values for the regression, the problem of
    transforming the fixed size anchor into a more suitable bounding box (equal
    to the ground truth box) only applies to the anchors that we consider to be
    foreground.
    为了计算正确的回归值, 转化被认为前景的固定大小的anchor到一个更为合适的框

    RPNTarget is also responsible for selecting which of the anchors are going
    to be used for the minibatch. This is a random process with some
    restrictions on the ratio between foreground and background samples.
    这个函数也被用来选择将要被用在minibatch上的anchors, 这是一个随机的过程, 在前景背景样本熵
    使用一些限制

    For selecting the minibatch, labels are not only set to 0 or 1 (for the
    cases of being background and foreground respectively), but also to -1 for
    the anchors we just want to ignore and not include in the minibatch.
    对于选择的minibatch, 标签不止被设定为0或1(对应于背景和前景), 但也有-1, 指代我们想要忽略
    并且不包含在minibatch中的anchors

    In summary:
      * 1 is positive
          when GT overlap is >= 0.7 (configurable) or for GT max overlap (one
          anchor)
      * 0 is negative
          when GT overlap is < 0.3 (configurable)
      * -1 is don't care
          useful for subsampling negative labels

    Returns:
        labels: label for each anchor
        bbox_targets: bbox regresion values for each anchor
    """

    def __init__(self, num_anchors, config, seed=None, name='anchor_target'):
        super(RPNTarget, self).__init__(name=name)
        self._num_anchors = num_anchors

        self._allowed_border = config.allowed_border
        # We set clobber positive to False to make sure that there is always at
        # least one positive anchor per GT box.
        # 这个参数, 如果阈值太低, 则用N覆盖P
        self._clobber_positives = config.clobber_positives
        # We set anchors as positive when the IoU is greater than
        # `positive_overlap`.
        # 即前面提到的0.7
        self._positive_overlap = config.foreground_threshold
        # We set anchors as negative when the IoU is less than
        # `negative_overlap`.
        # 即前面提到的0.3
        self._negative_overlap = config.background_threshold_high
        # Fraction of the batch to be foreground labeled anchors.
        # 前景标签的anchors占batch的比例
        self._foreground_fraction = config.foreground_fraction
        # batch大小, 即256
        self._minibatch_size = config.minibatch_size

        # When choosing random targets use `seed` to replicate behaviour.
        self._seed = seed

    def _build(self, all_anchors, gt_boxes, im_shape):
        """
        We compare anchors to GT and using the minibatch size and the different
        config settings (clobber, foreground fraction, etc), we end up with
        training targets *only* for the elements we want to use in the batch,
        while everything else is ignored.

        Basically what it does is, first generate the targets for all (valid)
        anchors, and then start subsampling the positive (foreground) and the
        negative ones (background) based on the number of samples of each type
        that we want.
        对于所有的anchors生成一个标签, 然后开始对正负标签进行anchors进行采样, 确定batch

        Args:
            all_anchors:
                A Tensor with all the bounding boxes coords of the anchors.
                Its shape should be (num_anchors, 4).
            gt_boxes:
                A Tensor with the ground truth bounding boxes of the image of
                the batch being processed. Its shape should be (num_gt, 5).
                The last dimension is used for the label.
            im_shape:
                Shape of original image (height, width) in order to define
                anchor targers in respect with gt_boxes.

        Returns:
            Tuple of the tensors of:
                labels: (1, 0, -1) for each anchor.
                    Shape (num_anchors, 1)
                bbox_targets: 4d bbox targets as specified by paper.
                    Shape (num_anchors, 4)
                max_overlaps: Max IoU overlap with ground truth boxes.
                    Shape (num_anchors, 1)
        """
        # Keep only the coordinates of gt_boxes
        # gt_boxes是第二维是5, 这里只是用了前四个元素
        gt_boxes = gt_boxes[:, :4]
        all_anchors = all_anchors[:, :4]

        # Only keep anchors inside the image
        # 只保留图像内的anchors
        (x_min_anchor, y_min_anchor,
         x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

        anchor_filter = tf.logical_and(
            tf.logical_and(
                tf.greater_equal(x_min_anchor, -self._allowed_border),
                tf.greater_equal(y_min_anchor, -self._allowed_border)
            ),
            tf.logical_and(
                tf.less(x_max_anchor, im_shape[1] + self._allowed_border),
                tf.less(y_max_anchor, im_shape[0] + self._allowed_border)
            )
        )

        # We (force) reshape the filter so that we can use it as a boolean mask
        # 强制展开(虽然本身就是一维的吧?)为一维, 以作为掩膜, 来确定有效的图像内的anchors
        anchor_filter = tf.reshape(anchor_filter, [-1])
        # Filter anchors.
        # 仅保留图像内的anchors (num_anchors, 4)
        anchors = tf.boolean_mask(
            all_anchors, anchor_filter, name='filter_anchors')

        # Generate array with the labels for all_anchors.
        # fill在这里创建了一个用标量-1, 填充的固定大小的张量
        # gather利用索引重新组建了一个张量, 这里只是抽取了all_anchors的形状的第一维度大小,
        # 以此确定的张量, 用-1填充
        # 这里也就是为了得到对于每个anchor所对应的种类标签, 预先设定为-1, 后续在进行更改
        # (num_anchors. )
        labels = tf.fill((tf.gather(tf.shape(all_anchors), [0])), -1)
        # 仅保留图像内的anchors的类别
        labels = tf.boolean_mask(labels, anchor_filter, name='filter_labels')

        # Intersection over union (IoU) overlap between the anchors and the
        # ground truth boxes.
        # 获取IoU
        overlaps = bbox_overlap_tf(tf.to_float(anchors), tf.to_float(gt_boxes))

        # Generate array with the IoU value of the closest GT box for each
        # anchor. 获取每个anchors对于所有的真实框的最大IoU值
        # reduce_max 计算指定维度的最大值
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        # 这里为负样本确定了标签
        if not self._clobber_positives:
            # 首先设定背景标签, 这样可以使得正样本可以覆盖负样本, 这样可以保证每个ground
            # truth至少对应一个正样本anchor
            # Assign bg labels first so that positive labels can clobber them.
            # First we get an array with True where IoU is less than
            # self._negative_overlap
            # 获取负样本的对应的真值
            negative_overlap_nonzero = tf.less(
                max_overlaps, self._negative_overlap)

            # Finally we set 0 at True indices
            # labels现在全是-1, 这里的tf.where表现出来和其他语言的三目运算符是相同的
            # condition的真值(也就是非0的负样本)对应位置返回x(即0元素), 假值(也即是为0的
            # 负样本, 即IoU并不是小于0.3的样本)对应返回y(即-1元素)
            # 这里还没有指定正样本
            labels = tf.where(
                condition=negative_overlap_nonzero,
                x=tf.zeros(tf.shape(labels)), y=tf.to_float(labels)
            )

        # 这里是要以真实框的角度来确定正样本#########################################
        # Get the value of the max IoU for the closest anchor for each gt.
        # 每个真实框对应的最大IoU值
        # tf.shape = [num_gt]
        gt_max_overlaps = tf.reduce_max(overlaps, axis=0)

        # Find all the indices that match (at least one, but could be more).
        # 找到所有匹配的的位置, 也就是针对每个真实框对应的IoU最大值的匹配结果(逻辑张量)
        # ques: 这里的squeeze没看明白使用的含义
        # ans: 消除冗余的维度
        gt_argmax_overlaps = tf.squeeze(tf.equal(overlaps, gt_max_overlaps))

        # where只有一个输入的时候, 表示是条件变量, 只返回自身真值的位置, 各个对应关系对应的
        # 位置[[x,y],...,[x,y]], 这里得到的是第一列的值, 形如[x, x, x, ..., x]
        # 第一列代表着有哪些anchors有真实框对应
        gt_argmax_overlaps = tf.where(gt_argmax_overlaps)[:, 0]

        # Eliminate duplicates indices.
        # 消除重复的索引, 这里得到的是**不再重复的元素值**组成的张量
        # 因为同一个anchor可能对应多个真实框, 这里得到的是消除了重复的最终的anchor
        # (1, num_unique_anchors), 保留下的是有真实框对应的anchors序号
        gt_argmax_overlaps, _ = tf.unique(gt_argmax_overlaps)

        # Order the indices for sparse_to_dense compatibility
        # ques: 不太理解这里, 因为gt_argmax_overlaps这里已经是一个一维的张量
        # ans: 这里应该是只是实现了一个排序
        # 这个函数的作用是返回 input 中每行(这里就是每个anchor所对应的IoU)最大的 k 个数，
        # 并且返回它们所在位置的索引, 这里只是承接了数
        gt_argmax_overlaps, _ = tf.nn.top_k(
            gt_argmax_overlaps, k=tf.shape(gt_argmax_overlaps)[-1])

        # 沿着指定的轴进行反转, 现在是坐标值x从小到大排列了
        # (1, num_unique_anchors)
        gt_argmax_overlaps = tf.reverse(gt_argmax_overlaps, [0])

        # Foreground label: for each ground-truth, anchor with highest overlap.
        # When the argmax is many items we use all of them (for consistency).
        # We set 1 at gt_argmax_overlaps_cond indices
        # 在gt_argmax_overlap(也就是真实框对应IOU最大的的anchors)所对应的位置, 替换为
        # true, (1, num_anchors)
        gt_argmax_overlaps_cond = tf.sparse_to_dense(
            gt_argmax_overlaps, tf.shape(labels, out_type=tf.int64),
            True, default_value=False
        )

        # 为剩下的真实框对应的最大IoU的anchors的位置上返回1, 这部分当做正样本, 其余label不
        # 变
        # **对每个标定的ground truth，与其重叠比例IoU最大的anchor记为正样本**
        labels = tf.where(
            condition=gt_argmax_overlaps_cond,
            x=tf.ones(tf.shape(labels)), y=tf.to_float(labels)
        )

        # 这里是从anchors的角度开始确定正样本#######################################
        # Foreground label: above threshold Intersection over Union (IoU)
        # First we get an array with True where IoU is greater or equal than
        # self._positive_overlap
        # 大于阈值0.7的也作为正样本
        positive_overlap_inds = tf.greater_equal(
            max_overlaps, self._positive_overlap)
        # Finally we set 1 at True indices
        # 最终为正样本设定为1, 其余的保留
        labels = tf.where(
            condition=positive_overlap_inds,
            x=tf.ones(tf.shape(labels)), y=labels
        )

        # 为负样本确定了标签
        if self._clobber_positives:
            # 进入到这里的时候, 说明要使用负样本覆盖正样本, 也就是不一定是每个真实框都有一个
            # 对应正样本
            # Assign background labels last so that negative labels can clobber
            # positives. First we get an array with True where IoU is less than
            # self._negative_overlap
            # 得到IoU小于0.3的位置
            negative_overlap_nonzero = tf.less(
                max_overlaps, self._negative_overlap)
            # Finally we set 0 at True indices
            # 为之设定为0标签
            labels = tf.where(
                condition=negative_overlap_nonzero,
                x=tf.zeros(tf.shape(labels)), y=labels
            )

        # Subsample positive labels if we have too many
        def subsample_positive():
            # Shuffle the foreground indices
            # 打乱所有前景数据的索引, 对于所有忽略的anchors的位置, 返回-1,
            # 其余的保持label的原样
            disable_fg_inds = tf.random_shuffle(fg_inds, seed=self._seed)
            # Select the indices that we have to ignore, this is
            # `tf.shape(fg_inds)[0] - num_fg` because we want to get only
            # `num_fg` foreground labels.
            # 得到不想要的前景数据的数量
            disable_place = (tf.shape(fg_inds)[0] - num_fg)
            # 得到不想要的前景数据
            disable_fg_inds = disable_fg_inds[:disable_place]
            # Order the indices for sparse_to_dense compatibility
            # 得到每行最大的k个值, 这里的k的大小等于输入的第二个维度
            # ques: 这里的k的设定有什么含义?
            disable_fg_inds, _ = tf.nn.top_k(
                disable_fg_inds, k=tf.shape(disable_fg_inds)[-1])
            disable_fg_inds = tf.reverse(disable_fg_inds, [0])
            disable_fg_inds = tf.sparse_to_dense(
                disable_fg_inds, tf.shape(labels, out_type=tf.int64),
                True, default_value=False
            )
            # Put -1 to ignore the anchors in the selected indices
            # 对于所有忽略的anchors的位置, 返回-1, 其余的保持label的原样
            return tf.where(
                condition=tf.squeeze(disable_fg_inds),
                x=tf.to_float(tf.fill(tf.shape(labels), -1)), y=labels
            )

        # 想要的前景样本的数量
        num_fg = tf.to_int32(self._foreground_fraction * self._minibatch_size)
        # Get foreground indices, get True in the indices where we have a one.
        # 获得前景样本的位置(labels是[num_anchors, 1])
        fg_inds = tf.equal(labels, 1)
        # We get only the indices where we have True.
        # 获得真值的坐标
        fg_inds = tf.squeeze(tf.where(fg_inds), axis=1)
        fg_inds_size = tf.size(fg_inds)
        # Condition for check if we have too many positive labels.
        subsample_positive_cond = fg_inds_size > num_fg
        # Check the condition and subsample positive labels.
        # 若是实际的正样本的数量多对想要的值, 那就下采样, 否则返回原值
        labels = tf.cond(
            subsample_positive_cond,
            true_fn=subsample_positive, false_fn=lambda: labels
        )

        # Subsample negative labels if we have too many
        def subsample_negative():
            # 具体操作和前面的正样本的采样一致, 只不过这里采样的是负样本
            # Shuffle the background indices
            disable_bg_inds = tf.random_shuffle(bg_inds, seed=self._seed)

            # Select the indices that we have to ignore, this is
            # `tf.shape(bg_inds)[0] - num_bg` because we want to get only
            # `num_bg` background labels.
            disable_place = (tf.shape(bg_inds)[0] - num_bg)
            disable_bg_inds = disable_bg_inds[:disable_place]
            # Order the indices for sparse_to_dense compatibility
            disable_bg_inds, _ = tf.nn.top_k(
                disable_bg_inds, k=tf.shape(disable_bg_inds)[-1])
            disable_bg_inds = tf.reverse(disable_bg_inds, [0])
            disable_bg_inds = tf.sparse_to_dense(
                disable_bg_inds, tf.shape(labels, out_type=tf.int64),
                True, default_value=False
            )
            # Put -1 to ignore the anchors in the selected indices
            return tf.where(
                condition=tf.squeeze(disable_bg_inds),
                x=tf.to_float(tf.fill(tf.shape(labels), -1)), y=labels
            )

        # Recalculate the foreground indices after (maybe) disable some of them

        # Get foreground indices, get True in the indices where we have a one.
        fg_inds = tf.equal(labels, 1)
        # We get only the indices where we have True.
        fg_inds = tf.squeeze(tf.where(fg_inds), axis=1)
        fg_inds_size = tf.size(fg_inds)

        # 开始为负样本进行下采样
        num_bg = tf.to_int32(self._minibatch_size - fg_inds_size)
        # Get background indices, get True in the indices where we have a zero.
        bg_inds = tf.equal(labels, 0)
        # We get only the indices where we have True.
        bg_inds = tf.squeeze(tf.where(bg_inds), axis=1)
        bg_inds_size = tf.size(bg_inds)
        # Condition for check if we have too many positive labels.
        subsample_negative_cond = bg_inds_size > num_bg
        # Check the condition and subsample positive labels.
        labels = tf.cond(
            subsample_negative_cond,
            true_fn=subsample_negative, false_fn=lambda: labels
        )

        # Return bbox targets with shape (anchors.shape[0], 4).

        # Find the closest gt box for each anchor.
        # 为每个anchor确定最大IoU的真实框, 确定对应的坐标
        argmax_overlaps = tf.argmax(overlaps, axis=1)
        # Eliminate duplicates.
        # 消除重复, 得到去除重复后的anchor对应关系
        argmax_overlaps_unique, _ = tf.unique(argmax_overlaps)
        # Filter the gt_boxes.
        # We get only the indices where we have "inside anchors".
        # 只保留图像内部的anchor
        anchor_filter_inds = tf.where(anchor_filter)
        # 从对应的坐标中返回真实框, 与anchors互相对应
        gt_boxes = tf.gather(gt_boxes, argmax_overlaps)

        # 计算真实值相对于anchors的偏移量和缩放量
        bbox_targets = encode_tf(anchors, gt_boxes)

        # For the anchors that aren't foreground, we ignore the bbox_targets.
        # 对于不是前景的bbox_targets值进行忽略
        anchor_foreground_filter = tf.equal(labels, 1)
        bbox_targets = tf.where(
            condition=anchor_foreground_filter,
            x=bbox_targets, y=tf.zeros_like(bbox_targets)
        )

        # We unroll "inside anchors" value for all anchors (for shape
        # compatibility).

        # We complete the missed indices with zeros
        # (because scatter_nd has zeros as default).
        # 在shape形状的全零张量中的indices对应的位置替换为updates对应的数据
        # 这里返回的结果的含义是: 图像内部的anchors对应的真实值相对的偏移量和缩放量
        bbox_targets = tf.scatter_nd(
            indices=tf.to_int32(anchor_filter_inds),
            updates=bbox_targets,
            shape=tf.shape(all_anchors)
        )

        # 将标签更新到对应的0张量的位置上
        labels_scatter = tf.scatter_nd(
            indices=tf.to_int32(anchor_filter_inds),
            updates=labels,
            shape=[tf.shape(all_anchors)[0]]
        )
        # We have to put -1 to ignore the indices with 0 generated by
        # scatter_nd, otherwise it will be considered as background.
        # 将对应的忽略标志放到图像外的anchors的对应的位置上
        labels = tf.where(
            condition=anchor_filter, x=labels_scatter,
            y=tf.to_float(tf.fill(tf.shape(labels_scatter), -1))
        )

        # 在图像内的anchors位置上更新针对每个真实框最大IoU结果
        max_overlaps = tf.scatter_nd(
            indices=tf.to_int32(anchor_filter_inds),
            updates=max_overlaps,
            shape=[tf.shape(all_anchors)[0]]
        )

        return labels, bbox_targets, max_overlaps
