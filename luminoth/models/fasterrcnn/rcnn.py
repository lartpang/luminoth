import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rcnn_proposal import RCNNProposal
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget
from luminoth.models.fasterrcnn.roi_pool import ROIPoolingLayer
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.vars import (
    get_initializer, layer_summaries, variable_summaries,
    get_activation_function
)


class RCNN(snt.AbstractModule):
    """RCNN: Region-based Convolutional Neural Network.
    这里就是RPN之后的操作, 包含了RoI Pooling操作和之后的分类回归操作

    Given region proposals (bounding boxes on an image) and a feature map of
    that image, RCNN adjusts the bounding boxes and classifies each region as
    either background or a specific object class.

    Steps:
        1. Region of Interest Pooling. Extract features from the feature map
           (based on the proposals) and convert into fixed size tensors
           (applying extrapolation).RoI池化
        2. Two fully connected layers generate a smaller tensor for each
           region.雙層全連接
        3. A fully conected layer outputs the probability distribution over the
           classes (plus a background class), and another fully connected layer
           outputs the bounding box regressions (one 4-d regression for each of
           the possible classes).一個輸出概率, 一個輸出迴歸

    Using the class probability, filter regions classified as background. For
    the remaining regions, use the class probability together with the
    corresponding bounding box regression offsets to generate the final object
    bounding boxes, with classes and probabilities assigned.
    使用类概率过滤背景区域. 对于留下来的区域, 使用类概率和对应的边界框偏移结果来生成最终的目标
    边界框, 包含类别和概率
    """

    def __init__(self, num_classes, config, debug=False, seed=None,
                 name='rcnn'):
        super(RCNN, self).__init__(name=name)
        self._num_classes = num_classes
        # List of the fully connected layer sizes used before classifying and
        # adjusting the bounding box.
        # 配置层的通道数, 例如这里是[4096, 4096], 代表两层通道数为4096的全连接层
        self._layer_sizes = config.layer_sizes
        self._activation = get_activation_function(config.activation_function)
        self._dropout_keep_prob = config.dropout_keep_prob
        self._use_mean = config.use_mean
        self._variances = config.target_normalization_variances

        self._rcnn_initializer = get_initializer(
            config.rcnn_initializer, seed=seed
        )
        self._cls_initializer = get_initializer(
            config.cls_initializer, seed=seed
        )
        self._bbox_initializer = get_initializer(
            config.bbox_initializer, seed=seed
        )
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)

        self._l1_sigma = config.l1_sigma

        # Debug mode makes the module return more detailed Tensors which can be
        # useful for debugging.
        self._debug = debug
        self._config = config
        self._seed = seed

    def _instantiate_layers(self):
        """
        定义两个全连接层和后面的分类回归层
        :return: 无
        """

        # We define layers as an array since they are simple fully connected
        # ones and it should be easy to tune it from the network config.
        # 定义两个全连接层, layer_size表示输出通道数
        self._layers = [
            snt.Linear(
                layer_size,
                name='fc_{}'.format(i),
                initializers={'w': self._rcnn_initializer},
                regularizers={'w': self.regularizer},
            )
            for i, layer_size in enumerate(self._layer_sizes)
        ]
        # We define the classifier layer having a num_classes + 1 background
        # since we want to be able to predict if the proposal is background as
        # well.
        # 定义一个分类的全连接层(21类)
        self._classifier_layer = snt.Linear(
            self._num_classes + 1, name='fc_classifier',
            initializers={'w': self._cls_initializer},
            regularizers={'w': self.regularizer},
        )

        # The bounding box adjustment layer has 4 times the number of classes
        # We choose which to use depending on the output of the classifier
        # layer
        # 输出通道数为类别数目x4, 也就是4x20, 每个类别对应4个值, 这个是独立于分类器的输出
        self._bbox_layer = snt.Linear(
            self._num_classes * 4, name='fc_bbox',
            initializers={'w': self._bbox_initializer},
            regularizers={'w': self.regularizer}
        )

        # ROIPoolingLayer is used to extract the feature from the feature map
        # using the proposals.
        # self._config.roi表示的就是选择crop, 7x7, valid的池化配置
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)
        # RCNNTarget is used to define a minibatch and the correct values for
        # each of the proposals.
        # 用来定义minibatch和提案的正确值
        # todo: 这里有待阅读
        self._rcnn_target = RCNNTarget(
            self._num_classes, self._config.target, variances=self._variances,
            seed=self._seed
        )
        # RCNNProposal generates the final bounding boxes and tries to remove
        # duplicates.
        # todo: 有待阅读, 不过感觉和RPNProposal函数的功能有些类似, 主要是NMS/移除重复
        self._rcnn_proposal = RCNNProposal(
            self._num_classes, self._config.proposals,
            variances=self._variances
        )

    def _build(self, conv_feature_map, proposals, im_shape, base_network,
               gt_boxes=None, is_training=False):
        """
        Classifies & refines proposals based on the pooled feature map.
        基于得出的池化特征图来进行分类优化提案

        Args:
            conv_feature_map: The feature map of the image, extracted
                using the pretrained network.
                Shape: (num_proposals, pool_height, pool_width, 512).
                卷积输出的特征图, 若是使用VGG, 对应的是14x14x512
            proposals: A Tensor with the bounding boxes proposed by the RPN.
                Shape: (total_num_proposals, 4).
                Encoding: (x1, y1, x2, y2).
                RPN得出的边界框
            im_shape: A Tensor with the shape of the image in the form of
                (image_height, image_width).
            gt_boxes (optional): A Tensor with the ground truth boxes of the
                image.
                Shape: (total_num_gt, 5).
                Encoding: (x1, y1, x2, y2, label).
            is_training (optional): A boolean to determine if we are just using
                the module for training or just inference.

        Returns:
            prediction_dict: a dict with the object predictions.
                It should have the keys:
                objects:
                labels:
                probs:

                rcnn:
                target:

        """
        self._instantiate_layers()

        prediction_dict = {'_debug': {}}

        # 真实数据存在
        if gt_boxes is not None:
            # 获得提案目标值, 和边界框偏移量
            # todo: rcnn_target的代码需要阅读
            proposals_target, bbox_offsets_target = self._rcnn_target(
                proposals, gt_boxes)

            if is_training:
                with tf.name_scope('prepare_batch'):
                    # We flatten to set shape, but it is already a flat Tensor.
                    # 返回提案目标值为大于等于0的提案的对应为真的逻辑张量, 表示对应的类别
                    # 每个提案只对应一个类别
                    in_batch_proposals = tf.reshape(
                        tf.greater_equal(proposals_target, 0), [-1]
                    )
                    # 获取有效的提案
                    proposals = tf.boolean_mask(
                        proposals, in_batch_proposals)
                    # 获取有效的边界框对应的真实值相对anchor的偏移量和缩放
                    bbox_offsets_target = tf.boolean_mask(
                        bbox_offsets_target, in_batch_proposals)
                    # 获取有效的提案对应的真实类别标定值
                    proposals_target = tf.boolean_mask(
                        proposals_target, in_batch_proposals)

            prediction_dict['target'] = {
                'cls': proposals_target,
                'bbox_offsets': bbox_offsets_target,
            }

        # 在特征图上利用proposals实现RoI Pooling操作
        roi_prediction = self._roi_pool(proposals, conv_feature_map, im_shape)

        if self._debug:
            # Save raw roi prediction in debug mode.
            prediction_dict['_debug']['roi'] = roi_prediction

        pooled_features = roi_prediction['roi_pool']
        features = base_network._build_tail(
            pooled_features, is_training=is_training
        )

        # 在最后一层全连接之前, 这里主要针对HW维度, 进行全局平均池化
        if self._use_mean:
            # We avg our height and width dimensions for a more
            # "memory-friendly" Tensor.
            features = tf.reduce_mean(features, [1, 2])

        # We treat num proposals as batch number so that when flattening we
        # get a (num_proposals, flatten_pooled_feature_map_size) Tensor.
        # flatten会保留batch数, 得出的是一个二维张量
        flatten_features = tf.contrib.layers.flatten(features)
        net = tf.identity(flatten_features)

        # 训练的时候, RoI Pooling之后要用dropout
        if is_training:
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        if self._debug:
            prediction_dict['_debug']['flatten_net'] = net

        # After flattening we are left with a Tensor of shape
        # (num_proposals, pool_height * pool_width * 512).
        # The first dimension works as batch size when applied to snt.Linear.
        # 开始构建总体的RCNN网络层
        for i, layer in enumerate(self._layers):
            # Through FC layer.
            net = layer(net)

            # Apply activation and dropout.
            variable_summaries(
                net, 'fc_{}_preactivationout'.format(i), 'reduced'
            )
            # 使用的relu6:  min(max(features, 0), 6)
            net = self._activation(net)
            if self._debug:
                prediction_dict['_debug']['layer_{}_out'.format(i)] = net

            variable_summaries(net, 'fc_{}_out'.format(i), 'reduced')
            if is_training:
                net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        # 创建softmax分类分支
        cls_score = self._classifier_layer(net)
        cls_prob = tf.nn.softmax(cls_score, axis=1)
        # 创建框回归分支
        bbox_offsets = self._bbox_layer(net)

        prediction_dict['rcnn'] = {
            'cls_score': cls_score,
            'cls_prob': cls_prob,
            'bbox_offsets': bbox_offsets,
        }

        # Get final objects proposals based on the probabilty, the offsets and
        # the original proposals.
        # 得到最终的预测结果, 基于概率, 偏移量, 和原始提案
        # todo: rcnn_proposal
        proposals_pred = self._rcnn_proposal(
            proposals, bbox_offsets, cls_prob, im_shape)

        # objects, objects_labels, and objects_labels_prob are the only keys
        # that matter for drawing objects.
        prediction_dict['objects'] = proposals_pred['objects']
        prediction_dict['labels'] = proposals_pred['proposal_label']
        prediction_dict['probs'] = proposals_pred['proposal_label_prob']

        if self._debug:
            prediction_dict['_debug']['proposal'] = proposals_pred

        # Calculate summaries for results
        variable_summaries(cls_prob, 'cls_prob', 'reduced')
        variable_summaries(bbox_offsets, 'bbox_offsets', 'reduced')

        if self._debug:
            variable_summaries(pooled_features, 'pooled_features', 'full')
            layer_summaries(self._classifier_layer, 'full')
            layer_summaries(self._bbox_layer, 'full')

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for RCNN based on:
        返回类别损失和回归损失, 基于cls_score, cls_prob, bbox_offsets, cls_target,
        bbox_offsets_target,

        Args:
            prediction_dict with keys:
                rcnn: 研究的是预测结果
                    cls_score: shape (num_proposals, num_classes + 1)
                        Has the class scoring for each the proposals. Classes
                        are 1-indexed with 0 being the background.
                        针对各个类别(包含背景), 各个提案区域对应的得分

                    cls_prob: shape (num_proposals, num_classes + 1)
                        Application of softmax on cls_score.
                        针对各个类别(包含背景), 各个提案区域对应的概率, 也就是cls_score
                        的softmax结果

                    bbox_offsets: shape (num_proposals, num_classes * 4)
                        Has the offset for each proposal for each class.
                        We have to compare only the proposals labeled with the
                        offsets for that label.
                        针对各个类别(不包含背景), 各个提案区域对应的坐标偏移量(4个值)
                        只需要比较标定的提案和那个标签的偏移量

                target: 研究的是真实标签
                    对于类别而言, 就是各个提案对应的正确的类别标签;
                    对于边界框而言, 各个提案对于真实标签的真实偏移量
                    cls_target: shape (num_proposals,)
                        Has the correct label for each of the proposals.
                        0 => background
                        1..n => 1-indexed classes

                    bbox_offsets_target: shape (num_proposals, 4)
                        ground truth相对anchor的偏移量和缩放量
                        Has the true offset of each proposal for the true
                        label.
                        In case of not having a true label (non-background)
                        then it's just zeroes.

        Returns:
            loss_dict with keys:
                rcnn_cls_loss: The cross-entropy or log-loss of the
                    classification tasks between then num_classes + background.
                rcnn_reg_loss: The smooth L1 loss for the bounding box
                    regression task to adjust correctly labeled boxes.

        """
        with tf.name_scope('RCNNLoss'):
            # 预测得分
            # (num_proposals, num_classes + 1)
            cls_score = prediction_dict['rcnn']['cls_score']
            # Cast target explicitly as int32.
            # 真实类别
            # (num_proposals, )
            cls_target = tf.cast(
                prediction_dict['target']['cls'], tf.int32
            )

            # First we need to calculate the log loss betweetn cls_prob and
            # cls_target, 需要计算分类概率的对数损失

            # 只计算正样本的损失
            # We only care for the targets that are >= 0
            # 寻找要保留, 不忽略的样本, 作为有效的样本
            not_ignored = tf.reshape(tf.greater_equal(
                cls_target, 0), [-1], name='not_ignored')
            # We apply boolean mask to score, prob and target.
            # 确定有效样本的类别预测得分
            cls_score_labeled = tf.boolean_mask(
                cls_score, not_ignored, name='cls_score_labeled')
            # 确定有效样本的真实类别
            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')

            tf.summary.scalar(
                'batch_size',
                tf.shape(cls_score_labeled)[0], ['rcnn']
            )

            # 将真实的类别转化为one-hot编码, 现在的cls_target_one_hot转化为
            # (num_proposal, 21)
            # Transform to one-hot vector
            cls_target_one_hot = tf.one_hot(
                cls_target_labeled, depth=self._num_classes + 1,
                name='cls_target_one_hot'
            )

            # We get cross entropy loss of each proposal.
            # 计算有效提案的真实类别和类别预测得分之间的交叉熵
            # 这里计算的时候一个表述的是样本分类的概率, 一个表述的是样本的真实类, 相当于只在
            # 对应的真实类别上进行了计算
            cross_entropy_per_proposal = (
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.stop_gradient(cls_target_one_hot),
                    logits=cls_score_labeled
                )
            )

            if self._debug:
                prediction_dict['_debug']['losses'] = {}
                # Save the cross entropy per proposal to be able to
                # visualize proposals with high and low error.
                prediction_dict['_debug']['losses'][
                    'cross_entropy_per_proposal'
                ] = (
                    cross_entropy_per_proposal
                )

            # Second we need to calculate the smooth l1 loss between
            # `bbox_offsets` and `bbox_offsets_target`.
            # 预测框相对anchor中心位置的偏移量以及宽高的缩放量t与ground truth相对anchor
            # 的偏移量和缩放量之间的smooth L1损失
            #  (num_proposals, num_classes * 4)
            bbox_offsets = prediction_dict['rcnn']['bbox_offsets']
            # (num_proposals, 4)
            bbox_offsets_target = (
                prediction_dict['target']['bbox_offsets']
            )

            # We only want the non-background labels bounding boxes.
            # 只计算类别标定值大于0的提案对应的边界框, 回归这边只计算非背景的有效框
            # (num_proposals, )
            not_ignored = tf.reshape(tf.greater(cls_target, 0), [-1])
            # (num_proposals, num_classes * 4)
            bbox_offsets_labeled = tf.boolean_mask(
                bbox_offsets, not_ignored, name='bbox_offsets_labeled')
            # (num_proposals, 4)
            bbox_offsets_target_labeled = tf.boolean_mask(
                bbox_offsets_target, not_ignored,
                name='bbox_offsets_target_labeled'
            )

            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')
            # `cls_target_labeled` is based on `cls_target` which has
            # `num_classes` + 1 classes.
            # for making `one_hot` with depth `num_classes` to work we need
            # to lower them to make them 0-index.
            # 对于one-hot编码, 需要索引从0开始, 非背景的标签是从1开始的, 所以直接减1就可以
            cls_target_labeled = cls_target_labeled - 1
            cls_target_one_hot = tf.one_hot(
                cls_target_labeled, depth=self._num_classes,
                name='cls_target_one_hot'
            )
            # 进行one-hot编码后, 数据的格式发生了改变
            # cls_target now is (num_proposals, num_classes)

            # (num_proposals x num_classes, 4)
            bbox_flatten = tf.reshape(
                bbox_offsets_labeled, [-1, 4], name='bbox_flatten')

            # We use the flatten cls_target_one_hot as boolean mask for the
            # bboxes.
            # 将cls_target_one_hot转化为一维的张量, 作为bboxes的掩膜来进行操作
            # 现在的cls_target_one_hot形状为(num_porposals, num_classes),
            # 也就是(n, 20), 进行reshape操作后应该是(n x 20, )
            cls_flatten = tf.cast(tf.reshape(
                cls_target_one_hot, [-1]), tf.bool, 'cls_flatten_as_bool')

            # bbox_flatten本身就是nx4的大小, 被一个一维的掩膜进行处理,
            # 这里确定了每个提案所对应的真实类别下的框的预测偏移量
            bbox_offset_cleaned = tf.boolean_mask(
                bbox_flatten, cls_flatten, 'bbox_offset_cleaned')

            # Calculate the smooth l1 loss between the "cleaned" bboxes
            # offsets (that means, the useful results) and the labeled
            # targets.
            # 计算预测框相对anchor中心位置的偏移量以及宽高的缩放量与ground truth相对
            # anchor的偏移量和缩放量的之间的smoothL1损失
            reg_loss_per_proposal = smooth_l1_loss(
                bbox_offset_cleaned, bbox_offsets_target_labeled,
                sigma=self._l1_sigma
            )

            tf.summary.scalar(
                'rcnn_foreground_samples',
                tf.shape(bbox_offset_cleaned)[0], ['rcnn']
            )

            if self._debug:
                # Also save reg loss per proposals to be able to visualize
                # good and bad proposals in debug mode.
                prediction_dict['_debug']['losses'][
                    'reg_loss_per_proposal'
                ] = (
                    reg_loss_per_proposal
                )

            # reduce_* 系列函数, axis=None 表示最终的结果只有一个值
            return {
                'rcnn_cls_loss': tf.reduce_mean(cross_entropy_per_proposal),
                'rcnn_reg_loss': tf.reduce_mean(reg_loss_per_proposal),
            }
