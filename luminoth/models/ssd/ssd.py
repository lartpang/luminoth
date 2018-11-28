import numpy as np
import sonnet as snt
import tensorflow as tf
from sonnet.python.modules.conv import Conv2D

from luminoth.models.ssd.feature_extractor import SSDFeatureExtractor
from luminoth.models.ssd.proposal import SSDProposal
from luminoth.models.ssd.target import SSDTarget
from luminoth.models.ssd.utils import (adjust_bboxes, generate_raw_anchors)
from luminoth.utils.bbox_transform import clip_boxes
from luminoth.utils.losses import smooth_l1_loss


class SSD(snt.AbstractModule):
    """
    SSD: Single Shot MultiBox Detector
    """

    def __init__(self, config, name='ssd'):
        super(SSD, self).__init__(name=name)
        self._config = config.model
        self._num_classes = config.model.network.num_classes
        self._debug = config.train.debug
        self._seed = config.train.seed
        self._anchor_max_scale = config.model.anchors.max_scale
        self._anchor_min_scale = config.model.anchors.min_scale
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self.image_shape = [config.dataset.image_preprocessing.fixed_height,
                            config.dataset.image_preprocessing.fixed_width]
        self._anchors_per_point = config.model.anchors.anchors_per_point
        self._loc_loss_weight = config.model.loss.localization_loss_weight
        # TODO: Why not use the default LOSSES collection?
        self._losses_collections = ['ssd_losses']

    def _build(self, image, gt_boxes=None, is_training=False):
        """
        Returns bounding boxes and classification probabilities.

        Args:
            image: A tensor with the image.
                Its shape should be `(height, width, 3)`.
            gt_boxes: A tensor with all the ground truth boxes of that image.
                Its shape should be `(num_gt_boxes, 5)`
                Where for each gt box we have (x1, y1, x2, y2, label),
                in that order.
            is_training: A boolean to whether or not it is used for training.

        Returns:
            A dictionary with the following keys:
            predictions: 预测结果
                proposal_prediction: 关于提案的信息
                A dictionary with:
                    proposals: The proposals of the network after appling some
                        filters like negative area; and NMS
                        最终保留下来的提案
                    proposals_label: A tensor with the label for each proposal.
                        对于每个提案的标签判定
                    proposals_label_prob: A tensor with the softmax probability
                        for the label of each proposal.
                        对于每个提案的softmax概率
                bbox_offsets: A tensor with the predicted bbox_offsets
                    预测的边界框的偏移量
                class_scores: A tensor with the predicted classes scores
                    预测的类列得分/置信度
        """
        # Reshape image
        self.image_shape.append(3)  # Add channels to shape
        image.set_shape(self.image_shape)
        image = tf.expand_dims(image, 0, name='hardcode_batch_size_to_1')

        # Generate feature maps from image
        self.feature_extractor = SSDFeatureExtractor(
            self._config.base_network, parent_name=self.module_name
        )
        # 获取特征图
        # ques: 这里的特征图对应的是多个卷积层还是只是一个卷积层的输出
        # ans: 是所有需要研究的特征图
        feature_maps = self.feature_extractor(image, is_training=is_training)

        # Build a MultiBox predictor on top of each feature layer and collect
        # the bounding box offsets and the category score logits they produce
        bbox_offsets_list = []
        class_scores_list = []
        # 对于不同的输出特征图进行遍历, 进行预测
        for i, feat_map in enumerate(feature_maps.values()):
            multibox_predictor_name = 'MultiBox_{}'.format(i)
            with tf.name_scope(multibox_predictor_name):
                # 这里使得预测的结果的数量和后面生成anchors的数量是一致的,
                # 并且也是对应的
                num_anchors = self._anchors_per_point[i]

                # Predict bbox offsets
                # 用3x3卷积预测坐标偏移量
                bbox_offsets_layer = Conv2D(
                    num_anchors * 4, [3, 3],
                    name=multibox_predictor_name + '_offsets_conv'
                )(feat_map)

                # (HxWxnum_anchors, 4) 这里的H,W应该是和上面特征图的H,W是一致的
                bbox_offsets_flattened = tf.reshape(
                    bbox_offsets_layer, [-1, 4]
                )
                # 获取所有的预测框的偏移量
                bbox_offsets_list.append(bbox_offsets_flattened)

                # Predict class scores
                # 使用3x3卷积预测类别(包含背景类)
                class_scores_layer = Conv2D(
                    num_anchors * (self._num_classes + 1), [3, 3],
                    name=multibox_predictor_name + '_classes_conv',
                )(feat_map)
                class_scores_flattened = tf.reshape(
                    class_scores_layer, [-1, self._num_classes + 1]
                )
                # 获取所有的预测框类别判定
                class_scores_list.append(class_scores_flattened)

        # 组合所有的预测边界框偏移量, 类别得分, 并计算对应的softmax概率
        # (num_bboxes, 4) (相对于后面要生成的anchors) ###########################
        bbox_offsets = tf.concat(
            bbox_offsets_list, axis=0, name='concatenate_all_bbox_offsets'
        )
        # (num_bboxes, 21) 对应得分
        class_scores = tf.concat(
            class_scores_list, axis=0, name='concatenate_all_class_scores'
        )
        # (num_bboxes, 21) 对应概率 (也是在针对后面要生成的anchors) ###############
        class_probabilities = tf.nn.softmax(
            class_scores, axis=-1, name='class_probabilities_softmax'
        )

        # 这里的anchors不同于上面的预测结果, 而是根据特征图生成的参考框 ###############

        # Generate anchors (generated only once, therefore we use numpy)
        # 基于各个卷积层的特征图, 使用anchor参数, 生成所有的anchors(坐标基于特征图)
        raw_anchors_per_featmap = generate_raw_anchors(
            feature_maps, self._anchor_min_scale, self._anchor_max_scale,
            self._anchor_ratios, self._anchors_per_point
        )

        anchors_list = []
        # 遍历所有的特征图, 将其映射到原图, 并进行剪裁
        for i, (feat_map_name, feat_map) in enumerate(feature_maps.items()):
            # TODO: Anchor generation should be simpler. We should create
            #       them in image scale from the start instead of scaling
            #       them to their feature map size.
            # 这里的feat_map大大小应该是(num_batch, height, weight, channel)
            feat_map_shape = feat_map.shape.as_list()[1:3]
            # anchors从特征图映射到原图(坐标基于原图)
            scaled_bboxes = adjust_bboxes(
                raw_anchors_per_featmap[feat_map_name], feat_map_shape[0],
                feat_map_shape[1], self.image_shape[0], self.image_shape[1]
            )
            clipped_bboxes = clip_boxes(scaled_bboxes, self.image_shape)
            anchors_list.append(clipped_bboxes)
        # 将所有的anchors的原图上的坐标结果进行合并
        anchors = np.concatenate(anchors_list, axis=0)
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

        # This is the dict we'll return after filling it with SSD's results
        prediction_dict = {}

        # Generate targets for training
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

            # Generate targets
            target_creator = SSDTarget(
                self._num_classes, self._config.target, self._config.variances
            )
            # 返回各个anchor对应的类别标签(0~21),以及前景anchors位置上更新的对应真实框
            # 相对自身坐标的偏移量和缩放量(其余位置为0)
            # 这里的类别标签, 是对那些
            # 1. 在所有真实框中的最大的IoU值大于阈值的anchors,
            # 2. 以及那些所有真实框的最好anchors
            # 根据这些对应的最好的真实框来确定的, 这里使用IoU来确定的正样本的
            # class_targets, 和预测值无关, 只是背景样本里用了下预测的类别概率
            class_targets, bbox_offsets_targets = target_creator(
                class_probabilities, anchors, gt_boxes
            )

            # Filter the predictions and targets that we will ignore during
            # training due to hard negative mining. We use class_targets to
            # know which ones to ignore (they are marked as -1 if they are to
            # be ignored)
            # 确定前景anchors对应的各类数据, 包括:
            #   参考的anchors
            #   真实框 相对于anchors的偏移缩放
            #   anchors的对应的类别标签
            #   anchors对应的类别预测得分
            #   anchors对应的类别预测概率
            #   预测出的 提案框相对于anchors的偏移缩放
            # note: 这里对于每一个anchors都是预测一组偏移量, 是一一对应的
            with tf.name_scope('hard_negative_mining_filter'):
                predictions_filter = tf.greater_equal(class_targets, 0)
                anchors = tf.boolean_mask(anchors, predictions_filter)
                bbox_offsets_targets = tf.boolean_mask(
                    bbox_offsets_targets, predictions_filter)
                class_targets = tf.boolean_mask(
                    class_targets, predictions_filter)
                class_scores = tf.boolean_mask(
                    class_scores, predictions_filter)
                class_probabilities = tf.boolean_mask(
                    class_probabilities, predictions_filter)
                bbox_offsets = tf.boolean_mask(
                    bbox_offsets, predictions_filter)

            # Add target tensors to prediction dict
            # 和真实值有关系的几个数据
            prediction_dict['target'] = {
                'cls'         : class_targets,
                'bbox_offsets': bbox_offsets_targets,
                'anchors'     : anchors
            }

        # Add network's raw output to prediction dict
        # 和预测相关的几个数据
        prediction_dict['cls_pred'] = class_scores
        prediction_dict['loc_pred'] = bbox_offsets

        # 到此为止, 得到了所有的anchors调整后的预测结果, 但是这时候的结果并没有其他的处理
        # 只是在训练的时候就可以了, 但是预测或者调试输出的时候, 数据还需要进行进一步筛选

        # We generate proposals when predicting, or when debug=True for
        # generating visualizations during training.
        if not is_training or self._debug:
            proposals_creator = SSDProposal(
                self._num_classes, self._config.proposals,
                self._config.variances
            )
            proposals = proposals_creator(
                class_probabilities, bbox_offsets, anchors,
                tf.cast(tf.shape(image)[1:3], tf.float32)
            )
            prediction_dict['classification_prediction'] = proposals

        # Add some non essential metrics for debugging
        if self._debug:
            prediction_dict['all_anchors'] = anchors
            prediction_dict['cls_prob'] = class_probabilities

        return prediction_dict

    def loss(self, prediction_dict, return_all=False):
        """
        Compute the loss for SSD.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use different main keys:

                cls_pred: A dictionary with the classes classification.
                loc_pred: A dictionary with the localization predictions
                target: A dictionary with the targets for both classes and
                    localizations.

        Returns:
            A tensor for the total loss.
        """

        with tf.name_scope('losses'):
            # 类别预测得分结果
            cls_pred = prediction_dict['cls_pred']
            # 调整后的anchors对应的类别标签(这个是直接从真实框身上得来的)
            cls_target = tf.cast(prediction_dict['target']['cls'], tf.int32)
            # Transform to one-hot vector
            cls_target_one_hot = tf.one_hot(
                cls_target, depth=self._num_classes + 1,
                name='cls_target_one_hot'
            )

            # We get cross entropy loss of each proposal.
            # TODO: Optimization opportunity: We calculate the probabilities
            #       earlier in the program, so if we used those instead of the
            #       logits we would not have the need to do softmax here too.
            cross_entropy_per_proposal = (
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target_one_hot, logits=cls_pred
                )
            )

            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_targets`.
            # 一个是预测的偏移缩放值, 一个是真实框的偏移缩放值
            bbox_offsets = prediction_dict['loc_pred']
            bbox_offsets_targets = (prediction_dict['target']['bbox_offsets'])

            # We only want the non-background labels bounding boxes.
            not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
            bbox_offsets_positives = tf.boolean_mask(
                bbox_offsets, not_ignored, name='bbox_offsets_positives')
            bbox_offsets_target_positives = tf.boolean_mask(
                bbox_offsets_targets, not_ignored,
                name='bbox_offsets_target_positives'
            )

            # Calculate the smooth l1 regression loss between the flatten
            # bboxes offsets  and the labeled targets.
            reg_loss_per_proposal = smooth_l1_loss(
                bbox_offsets_positives, bbox_offsets_target_positives)

            cls_loss = tf.reduce_sum(cross_entropy_per_proposal)
            bbox_loss = tf.reduce_sum(reg_loss_per_proposal)

            # Following the paper, set loss to 0 if there are 0 bboxes
            # assigned as foreground targets.
            safety_condition = tf.not_equal(
                tf.shape(bbox_offsets_positives)[0], 0
            )
            final_loss = tf.cond(
                safety_condition,
                true_fn=lambda: (
                        (cls_loss + bbox_loss * self._loc_loss_weight) /
                        tf.cast(tf.shape(bbox_offsets_positives)[0], tf.float32)
                ),
                false_fn=lambda: 0.0
            )
            tf.losses.add_loss(final_loss)
            total_loss = tf.losses.get_total_loss()

            prediction_dict['reg_loss_per_proposal'] = reg_loss_per_proposal
            prediction_dict['cls_loss_per_proposal'] = (
                cross_entropy_per_proposal
            )

            tf.summary.scalar(
                'cls_loss', cls_loss,
                collections=self._losses_collections
            )

            tf.summary.scalar(
                'bbox_loss', bbox_loss,
                collections=self._losses_collections
            )

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            if return_all:
                return {
                    'total_loss': total_loss,
                    'cls_loss'  : cls_loss,
                    'bbox_loss' : bbox_loss
                }
            else:
                return total_loss

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        ssd network.
        """
        summaries = [
            tf.summary.merge_all(key=self._losses_collections[0])
        ]

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.base_network.trainable:
            pretrained_trainable_vars = (
                self.feature_extractor.get_trainable_vars()
            )
            tf.logging.info('Training {} vars from pretrained module.'.format(
                len(pretrained_trainable_vars)))
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_base_network_checkpoint_vars(self):
        return self.feature_extractor.get_base_network_checkpoint_vars()

    def get_checkpoint_file(self):
        return self.feature_extractor.get_checkpoint_file()
