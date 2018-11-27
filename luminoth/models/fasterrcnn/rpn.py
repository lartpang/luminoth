"""
RPN - Region Proposal Network
"""

import sonnet as snt
import tensorflow as tf

from sonnet.python.modules.conv import Conv2D

from .rpn_target import RPNTarget
from .rpn_proposal import RPNProposal
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import (
    get_initializer, layer_summaries, variable_summaries,
    get_activation_function
)


class RPN(snt.AbstractModule):

    def __init__(self, num_anchors, config, debug=False, seed=None,
                 name='rpn'):
        """RPN - Region Proposal Network.

        Given an image (as feature map) and a fixed set of anchors, the RPN
        will learn weights to adjust those anchors so they better look like the
        ground truth objects, as well as scoring them by "objectness" (ie. how
        likely they are to be an object vs background).

        The final result will be a set of rectangular boxes ("proposals"),
        each associated with an objectness score.

        Note: this module can be used independently of Faster R-CNN.
        """
        super(RPN, self).__init__(name=name)
        self._num_anchors = num_anchors
        self._num_channels = config.num_channels
        self._kernel_shape = config.kernel_shape

        self._debug = debug
        self._seed = seed

        self._rpn_initializer = get_initializer(
            config.rpn_initializer, seed=seed
        )
        # According to Faster RCNN paper we need to initialize layers with
        # "from a zero-mean Gaussian distribution with standard deviation 0.01
        # 初始化分类器,预测器,以及一些正则化参数
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed
        )
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed
        )
        self._regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale
        )

        self._l1_sigma = config.l1_sigma

        # We could use normal relu without any problems.
        # 确定RPN的激活函数, 这里使用正常ReLU
        self._rpn_activation = get_activation_function(
            config.activation_function
        )

        self._config = config

    def _instantiate_layers(self):
        """
        Instantiates all convolutional modules used in the RPN.
        创建RPN中的层结构, 即所谓的实例化检具体的层
        """

        # 3x3的卷基层
        self._rpn = Conv2D(
            output_channels=self._num_channels,
            kernel_shape=self._kernel_shape,
            initializers={'w': self._rpn_initializer},
            regularizers={'w': self._regularizer},
            name='conv'
        )

        # 分类分支, 输出的通道数目是 2k(k即为anchors数目)
        self._rpn_cls = Conv2D(
            output_channels=self._num_anchors * 2, kernel_shape=[1, 1],
            initializers={'w': self._cls_initializer},
            regularizers={'w': self._regularizer},
            padding='VALID',
            name='cls_conv'
        )

        # BBox prediction is 4 values * number of anchors.
        # 预测分支, 输出通道数目为 4k
        self._rpn_bbox = Conv2D(
            output_channels=self._num_anchors * 4, kernel_shape=[1, 1],
            initializers={'w': self._bbox_initializer},
            regularizers={'w': self._regularizer},
            padding='VALID',
            name='bbox_conv'
        )

    def _build(self, conv_feature_map, im_shape, all_anchors,
               gt_boxes=None, is_training=False):
        """Builds the RPN model subgraph.
        构建RPN模型计算图, 对应的是RPN的输入输出

        Args:
            conv_feature_map: A Tensor with the output of some pretrained
                network. Its dimensions should be
                `[1, feature_map_height, feature_map_width, depth]` where depth
                is 512 for the default layer in VGG and 1024 for the default
                layer in ResNet.
                按照VGG理解, 这里是conv5-3的卷积特征图, 特征是512维
            im_shape: A Tensor with the shape of the original image.
                原始图像大小, 可能需要来确定缩小的比例.
            all_anchors: A Tensor with all the anchor bounding boxes. Its shape
                should be
                [feature_map_height * feature_map_width * total_anchors, 4]
                这个是输入的对应的所有的anchors候选
            gt_boxes: A Tensor with the ground-truth boxes for the image.
                Its dimensions should be `[total_gt_boxes, 5]`, and it should
                consist of [x1, y1, x2, y2, label], being (x1, y1) -> top left
                point, and (x2, y2) -> bottom right point of the bounding box.
                真实标签使用左上和右下角确定的

        Returns:
            返回预测值和对应的类别
            prediction_dict: A dict with the following keys:
                proposals: A Tensor with a variable number of proposals for
                    objects on the image.
                scores: A Tensor with a "objectness" probability for each
                    proposal. The score should be the output of the softmax for
                    object.

                If training is True, then some more Tensors are added to the
                prediction dictionary to be used for calculating the loss.
                训练的时候会有额外的一些预测用来计算损失, 可以看下面的loss部分的介绍, 是需
                要这些值的

                rpn_cls_prob: A Tensor with the probability of being
                    background and foreground for each anchor.
                    前景背景的概率, 这个没有下面的得分值重要
                rpn_cls_score: A Tensor with the cls score of being background
                    and foreground for each anchor (the input for the softmax).
                    每个anchor的前景背景类别得分
                rpn_bbox_pred: A Tensor with the bounding box regression for
                    each anchor.
                    边界框回归结果
                rpn_cls_target: A Tensor with the target for each of the
                    anchors. The shape is [num_anchors,].
                    分类目标, 不明白这里所谓的"target"代表什么??????
                rpn_bbox_target: A Tensor with the target for each of the
                    anchors. In case of ignoring the anchor for the target then
                    we still have a bbox target for each anchors, and it's
                    filled with zeroes when ignored.
                    这里是边界框目标.如果对于目标忽略了anchor, 那么对于每个anchor,
                    我们还是会有一个边界框目标, 当他被忽略时, 将会用0填充?????

        """
        # We start with a common conv layer applied to the feature map.
        # 开始进入回归与分类各自的RPN分支
        self._instantiate_layers()
        # 将anchor和RPN预测转化为原始图像上的目标的提案
        self._proposal = RPNProposal(
            self._num_anchors, self._config.proposals, debug=self._debug
        )

        # Tuple of the tensors of:
        #     labels: (1, 0, -1) for each anchor.
        #         Shape (num_anchors, 1)
        #     bbox_targets: 4d bbox targets as specified by paper.
        #         Shape (num_anchors, 4)
        #         图像内部的anchors对应的真实值相对的偏移量和缩放量
        #     max_overlaps: Max IoU overlap with ground truth boxes.
        #         Shape (num_anchors, 1)
        #         每个anchor对应的IoU最大的真实框的索引
        self._anchor_target = RPNTarget(
            self._num_anchors, self._config.target, seed=self._seed
        )

        prediction_dict = {}

        # Get the RPN feature using a simple conv net. Activation function
        # can be set to empty.
        # 得到第一个3x3卷积的结果, 后加了个ReLU激活函数
        rpn_conv_feature = self._rpn(conv_feature_map)
        rpn_feature = self._rpn_activation(rpn_conv_feature)

        # Then we apply separate conv layers for classification and regression.
        # 获得类别得分和边界框预测, 这里各自只是经过了一次1x1的卷积, 就直接得到结果.
        rpn_cls_score_original = self._rpn_cls(rpn_feature)
        rpn_bbox_pred_original = self._rpn_bbox(rpn_feature)
        # rpn_cls_score_original has shape (1, H, W, num_anchors * 2)
        # rpn_bbox_pred_original has shape (1, H, W, num_anchors * 4)
        # where H, W are height and width of the pretrained feature map.
        # 因为使用的是3x3 padding=1 以及 1x1的卷积, 所以宽高不变, 而且这里也不能变,
        # 因为还要与原始的特征图所对应

        # Convert (flatten) `rpn_cls_score_original` which has two scalars per
        # anchor per location to be able to apply softmax.
        # 这里的操作是实现了一个flatten的操作, 但是对于每个anchor都有对应的两个值
        # 也就是前景和背景的概率(目标/非目标)
        rpn_cls_score = tf.reshape(rpn_cls_score_original, [-1, 2])
        # Now that `rpn_cls_score` has shape (H * W * num_anchors, 2), we apply
        # softmax to the last dim.
        # 对每个anchors应用一个softmax分类器得到类别预测
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        # 数据存档
        prediction_dict['rpn_cls_prob'] = rpn_cls_prob
        prediction_dict['rpn_cls_score'] = rpn_cls_score

        # 与上面类似的操作, 进行了把各个anchor全部展开
        # Flatten bounding box delta prediction for easy manipulation.
        # We end up with `rpn_bbox_pred` having shape (H * W * num_anchors, 4).
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred_original, [-1, 4])

        # 数据存档
        prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred

        # We have to convert bbox deltas to usable bounding boxes and remove
        # redundant ones using Non Maximum Suppression (NMS).
        # 将bbox增量转换为可用的边界框并使用NMS删除冗余
        # 这些操作都在self._proposal中实现了, 输出来的就是调整处理后的山下来的结果
        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape)

        # 数据存档
        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']

        if self._debug:
            prediction_dict['proposal_prediction'] = proposal_prediction

        if gt_boxes is not None:
            # When training we use a separate module to calculate the target
            # values we want to output.
            # 训练的时候, 使用一个独立的模块来计算想要的目标输出值
            # labels, bbox_targets, max_overlap
            (rpn_cls_target, rpn_bbox_target,
             rpn_max_overlap) = self._anchor_target(
                all_anchors, gt_boxes, im_shape
            )

            # 数据存档
            prediction_dict['rpn_cls_target'] = rpn_cls_target
            prediction_dict['rpn_bbox_target'] = rpn_bbox_target

            # ques: 这里的rpn_max_overlap代表的是什么
            # ans: 表示anchors对应的最大的IoU的真实框索引
            if self._debug:
                prediction_dict['rpn_max_overlap'] = rpn_max_overlap
                variable_summaries(rpn_bbox_target, 'rpn_bbox_target', 'full')

        # Variables summaries.
        # 可视化时使用的数据摘要
        variable_summaries(prediction_dict['scores'], 'rpn_scores', 'reduced')
        variable_summaries(rpn_cls_prob, 'rpn_cls_prob', 'reduced')
        variable_summaries(rpn_bbox_pred, 'rpn_bbox_pred', 'reduced')

        if self._debug:
            variable_summaries(rpn_feature, 'rpn_feature', 'full')
            variable_summaries(
               rpn_cls_score_original, 'rpn_cls_score_original', 'full')
            variable_summaries(
               rpn_bbox_pred_original, 'rpn_bbox_pred_original', 'full')

            # Layer summaries.
            layer_summaries(self._rpn, 'full')
            layer_summaries(self._rpn_cls, 'full')
            layer_summaries(self._rpn_bbox, 'full')

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_score: Score for being an object or not for each anchor
                in the image. Shape: (num_anchors, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                * 1: for positive labels
                * 0: for negative labels
                * -1: for labels we should ignore.
                Shape: (num_anchors, )
                对于anchor的真实标记, 这里应该是以IoU来判定的:
                对每个proposal，计算其与所有ground truth的重叠比例IoU, 筛选出与每个
                proposal重叠比例最大的ground truth.
                如果proposal的最大IoU大于0.5则为目标(前景), 标签值(label)为对应
                ground truth的目标分类如果IoU小于0.5且大于0.1则为背景，标签值为0
                ques: 这里的-1该如何理解?
                ans: 要忽略的部分, 因为并不总是所有的提案都要被用到
            rpn_bbox_target: Bounding box output delta target for rpn.
                Shape: (num_anchors, 4)
                这里输出的边界框的目标偏移量.
            rpn_bbox_pred: Bounding box output delta prediction for rpn.
                Shape: (num_anchors, 4)
                边界框的输出预测偏移量
        Returns:
            返回一个多任务损失
            Multiloss between cls probability and bbox target.
        """

        rpn_cls_score = prediction_dict['rpn_cls_score']
        rpn_cls_target = prediction_dict['rpn_cls_target']

        rpn_bbox_target = prediction_dict['rpn_bbox_target']
        rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

        with tf.variable_scope('RPNLoss'):
            # Flatten already flat Tensor for usage as boolean mask filter.
            rpn_cls_target = tf.cast(tf.reshape(
                rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
            # Transform to boolean tensor mask for not ignored.
            # 返回不应该被忽略的标签的逻辑张量, 可以用来作为一个实际需要处理的标签的
            # 掩膜
            labels_not_ignored = tf.not_equal(
                rpn_cls_target, -1, name='labels_not_ignored')

            # Now we only have the labels we are going to compare with the
            # cls probability.
            # 这里的掩膜函数可以提取张量里的对应于掩膜真值的位置上的数值, 进而获得将
            # 要用来比较的类别概率和标签
            labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
            cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)

            # We need to transform `labels` to `cls_score` shape.
            # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
            # 对于各个类别的分数匹配对应的标签, 对标签进行one-hot编码
            # ques: 目的是什么
            # ans: 计算交叉熵是需要使用onehot编码的
            cls_target = tf.one_hot(labels, depth=2)

            # Equivalent to log loss
            # 计算类别的对数损失, 这里使用的是softmax交叉熵的形式,
            # 计算labels和logits的交叉熵
            ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=cls_target, logits=cls_score
            )
            prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor

            # 计算回归损失
            # Finally, we need to calculate the regression loss over
            # `rpn_bbox_target` and `rpn_bbox_pred`.
            # We use SmoothL1Loss.
            rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

            # We only care for positive labels (we ignore backgrounds since
            # we don't have any bounding box information for it).
            # 只用正样本, 来计算回归损失
            positive_labels = tf.equal(rpn_cls_target, 1)
            rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
            rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

            # We apply smooth l1 loss as described by the Fast R-CNN paper.
            reg_loss_per_anchor = smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_target, sigma=self._l1_sigma
            )

            prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

            # Loss summaries.
            tf.summary.scalar('batch_size', tf.shape(labels)[0], ['rpn'])
            foreground_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 1))
            background_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 0))
            tf.summary.scalar(
                'foreground_cls_loss',
                tf.reduce_mean(foreground_cls_loss), ['rpn'])
            tf.summary.histogram(
                'foreground_cls_loss', foreground_cls_loss, ['rpn'])
            tf.summary.scalar(
                'background_cls_loss',
                tf.reduce_mean(background_cls_loss), ['rpn'])
            tf.summary.histogram(
                'background_cls_loss', background_cls_loss, ['rpn'])
            tf.summary.scalar(
                'foreground_samples', tf.shape(rpn_bbox_target)[0], ['rpn'])

            # 计算均值
            return {
                'rpn_cls_loss': tf.reduce_mean(ce_per_anchor),
                'rpn_reg_loss': tf.reduce_mean(reg_loss_per_anchor),
            }
