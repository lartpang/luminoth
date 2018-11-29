import sonnet as snt
import tensorflow as tf

from luminoth.utils.bbox_transform_tf import change_order, clip_boxes, decode


class SSDProposal(snt.AbstractModule):
    """
    Transforms anchors and SSD predictions into object proposals.
    转换anchors和预测为目标提案

    Using the fixed anchors and the SSD predictions for both classification
    and regression (adjusting the bounding box), we return a list of proposals
    with assigned class.

    In the process it tries to remove duplicated suggestions by applying non
    maximum suppresion (NMS).

    We apply NMS because the way object detectors are usually scored is by
    treating duplicated detections (multiple detections that overlap the same
    ground truth value) as false positive. It is resonable to assume that there
    may exist such case that applying NMS is completely unnecesary.

    Besides applying NMS it also filters the top N results, both for classes
    and in general. These values are easily modifiable in the configuration
    files.
    """

    def __init__(self, num_classes, config, variances, name='proposal_layer'):
        super(SSDProposal, self).__init__(name=name)
        self._num_classes = num_classes

        # Threshold to use for NMS.
        self._class_nms_threshold = config.class_nms_threshold
        # Max number of proposals detections per class.
        self._class_max_detections = config.class_max_detections
        # Maximum number of detections to return.
        self._total_max_detections = config.total_max_detections
        self._min_prob_threshold = config.min_prob_threshold or 0.0

        self._filter_outside_anchors = config.filter_outside_anchors
        self._variances = variances

    def _build(self, cls_prob, loc_pred, all_anchors, im_shape):
        """
        Args:
            cls_prob: A softmax probability for each anchor where the idx = 0
                is the background class (which we should ignore).
                Shape (total_anchors, num_classes + 1)
                预测类别概率
            loc_pred: A Tensor with the regression output for each anchor.
                Its shape should be (total_anchors, 4).
                预测框偏移缩放量
            all_anchors: A Tensor with the anchors bounding boxes of shape
                (total_anchors, 4), having (x_min, y_min, x_max, y_max) for
                each anchor.
                所有anchors的真实坐标
            im_shape: A Tensor with the image shape in format (height, width).
        Returns:
            prediction_dict with the following keys:
                raw_proposals: The raw proposals i.e. the anchors adjusted
                    using loc_pred.
                proposals: The proposals of the network after appling some
                    filters like negative area; and NMS. It's shape is
                    (final_num_proposals, 4), where final_num_proposals is
                    unknown before-hand (it depends on NMS).
                    The 4-length Tensor for each corresponds to:
                    (x_min, y_min, x_max, y_max).
                proposal_label: It's shape is (final_num_proposals,)
                proposal_label_prob: It's shape is (final_num_proposals,)
        """
        selected_boxes = []
        selected_probs = []
        selected_labels = []
        selected_anchors = []  # For debugging

        # 分析各类别下, 大于最小概率阈值的预测概率和预测偏移缩放量, 进而以此获得预测的边界
        # 框的坐标, 进行边界剪裁, 坐标合理性限定, NMS处理, 得到最终选定的各个类别下的提案
        for class_id in range(self._num_classes):
            # Get the confidences for this class (+ 1 is to ignore background)
            # 获取该类别下, 所有预测框的情况
            class_cls_prob = cls_prob[:, class_id + 1]

            # Filter by min_prob_threshold
            min_prob_filter = tf.greater_equal(
                class_cls_prob, self._min_prob_threshold)
            class_cls_prob = tf.boolean_mask(class_cls_prob, min_prob_filter)
            class_loc_pred = tf.boolean_mask(loc_pred, min_prob_filter)
            # 对所有anchors进行筛选
            anchors = tf.boolean_mask(all_anchors, min_prob_filter)

            # Using the loc_pred and the anchors, we generate the proposals.
            raw_proposals = decode(anchors, class_loc_pred, self._variances)

            # Clip boxes to image.
            clipped_proposals = clip_boxes(raw_proposals, im_shape)

            # Filter proposals that have an non-valid area.
            (x_min, y_min, x_max, y_max) = tf.unstack(
                clipped_proposals, axis=1)
            proposal_filter = tf.greater(
                tf.maximum(x_max - x_min, 0.) * tf.maximum(y_max - y_min, 0.),
                0.
            )
            # 筛选剪裁后的框坐标
            class_proposals = tf.boolean_mask(
                clipped_proposals, proposal_filter)
            # 筛选边界框偏移
            class_loc_pred = tf.boolean_mask(
                class_loc_pred, proposal_filter)
            # 筛选类别概率
            class_cls_prob = tf.boolean_mask(
                class_cls_prob, proposal_filter)
            # 筛选对应的anchors
            proposal_anchors = tf.boolean_mask(
                anchors, proposal_filter)

            # Log results of filtering non-valid area proposals
            # 所有anchors数量
            total_anchors = tf.shape(all_anchors)[0]
            # 所有坐标有效的框数量
            total_proposals = tf.shape(class_proposals)[0]
            # ques: 所有框的数量, 这里数量和anchors应该是一样的吧?
            # ans: 不一样, 未进行坐标和理性判断时框的总数, 但是已经进行了阈值判断
            total_raw_proposals = tf.shape(raw_proposals)[0]

            tf.summary.scalar(
                'invalid_proposals',
                total_proposals - total_raw_proposals, ['ssd']
            )
            tf.summary.scalar(
                'valid_proposals_ratio',
                tf.cast(total_anchors, tf.float32) /
                tf.cast(total_proposals, tf.float32), ['ssd']
            )

            # We have to use the TensorFlow's bounding box convention to use
            # the included function for NMS.
            # After gathering results we should normalize it back.
            class_proposal_tf = change_order(class_proposals)

            # Apply class NMS.
            # 使用该类别下所有预测的框坐标, 和对应的预测概率, 进行非极大值抑制, 得到索引
            # 剩下来的就认为是该类别下的结果, 也就是这个类别选择了这几个预测
            class_selected_idx = tf.image.non_max_suppression(
                class_proposal_tf, class_cls_prob, self._class_max_detections,
                iou_threshold=self._class_nms_threshold
            )

            # Using NMS resulting indices, gather values from Tensors.
            # 获得该类别选择的预测框和对应的类别预测概率
            class_proposal_tf = tf.gather(
                class_proposal_tf, class_selected_idx)
            class_cls_prob = tf.gather(class_cls_prob, class_selected_idx)

            # We append values to a regular list which will later be
            # transformed to a proper Tensor.
            #  获得该类别选择的预测框和对应的类别预测概率
            selected_boxes.append(class_proposal_tf)
            selected_probs.append(class_cls_prob)

            # In the case of the class_id, since it is a loop on classes, we
            # already have a fixed class_id. We use `tf.tile` to create that
            # Tensor with the total number of indices returned by the NMS.
            # 重复张量, 沿着后面指定的各个维度上的次数来进行重复
            # 与下面的的张量里的anchors相对应, 表示其类别标签
            selected_labels.append(
                tf.tile([class_id], [tf.shape(class_selected_idx)[0]])
            )
            # 确定该类别下所有坐标合理概率超过阈值的对应的anchors
            selected_anchors.append(proposal_anchors)

        # We use concat (axis=0) to generate a Tensor where the rows are
        # stacked on top of each other
        # (num_proposals, 4)
        proposals_tf = tf.concat(selected_boxes, axis=0)
        # Return to the original convention.
        proposals = change_order(proposals_tf)
        # (num_proposals, )
        proposal_label = tf.concat(selected_labels, axis=0)
        # (num_proposals, )
        proposal_label_prob = tf.concat(selected_probs, axis=0)
        # # (num_proposals, 4)
        proposal_anchors = tf.concat(selected_anchors, axis=0)

        # Get topK detections of all classes.
        k = tf.minimum(
            self._total_max_detections,
            tf.shape(proposal_label_prob)[0]
        )

        # 主题顺序是按照proposal_label_prob为参考的, 其中有各个类的结果, 顺序大致是按照
        # 类别来的, 下面的都是, 所以使用同一个索引是可以
        top_k = tf.nn.top_k(proposal_label_prob, k=k)

        # 依次获得NMS后前k个最大的预测概率值, 对应的预测框坐标组, 各类别中保留下来的提案对
        # 应的该类别, 对应的参考anchors坐标
        top_k_proposal_label_prob = top_k.values
        top_k_proposals = tf.gather(proposals, top_k.indices)
        top_k_proposal_label = tf.gather(proposal_label, top_k.indices)
        top_k_proposal_anchors = tf.gather(proposal_anchors, top_k.indices)

        return {
            'objects'      : top_k_proposals,
            'labels'       : top_k_proposal_label,
            'probs'        : top_k_proposal_label_prob,
            'raw_proposals': raw_proposals,
            'anchors'      : top_k_proposal_anchors,
        }
