import sonnet as snt
import tensorflow as tf
from sonnet.python.modules.conv import Conv2D
from tensorflow.contrib.layers.python.layers import utils

from luminoth.models.base import BaseNetwork

VALID_SSD_ARCHITECTURES = set([
    'truncated_vgg_16',
])


class SSDFeatureExtractor(BaseNetwork):

    def __init__(self, config, parent_name=None, name='ssd_feature_extractor',
                 **kwargs):
        super(SSDFeatureExtractor, self).__init__(config, name=name, **kwargs)
        if self._architecture not in VALID_SSD_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                self._architecture
            ))
        self.parent_name = parent_name
        self.activation_fn = tf.nn.relu

    def _init_vgg16_extra_layers(self):
        self.conv6 = Conv2D(1024, [3, 3], rate=6, name='conv6')
        self.conv7 = Conv2D(1024, [1, 1], name='conv7')
        self.conv8_1 = Conv2D(256, [1, 1], name='conv8_1')
        self.conv8_2 = Conv2D(512, [3, 3], stride=2, name='conv8_2')
        self.conv9_1 = Conv2D(128, [1, 1], name='conv9_1')
        self.conv9_2 = Conv2D(256, [3, 3], stride=2, name='conv9_2')
        self.conv10_1 = Conv2D(128, [1, 1], name='conv10_1')
        self.conv10_2 = Conv2D(256, [3, 3], padding='VALID', name='conv10_2')
        self.conv11_1 = Conv2D(128, [1, 1], name='conv11_1')
        self.conv11_2 = Conv2D(256, [3, 3], padding='VALID', name='conv11_2')

    def _build(self, inputs, is_training=True):
        """
        Args:
            inputs: A Tensor of shape `(batch_size, height, width, channels)`.

        Returns:
            A dict of feature maps to be consumed by an SSD network
        """
        # TODO: Is there a better way to manage scoping in these cases?
        scope = self.module_name
        if self.parent_name:
            scope = self.parent_name + '/' + scope

        base_net_endpoints = super(SSDFeatureExtractor, self)._build(
            inputs, is_training=is_training)['end_points']

        if self.truncated_vgg_16_type:
            # 把网络的后续后续部分搭建出来
            # As it is pointed out in SSD and ParseNet papers, `conv4_3` has a
            # different features scale compared to other layers, to adjust it
            # we need to add a spatial normalization before adding the
            # predictors.
            vgg_conv4_3 = base_net_endpoints[scope + '/vgg_16/conv4/conv4_3']
            tf.summary.histogram('conv4_3_hist', vgg_conv4_3)

            # conv4_3之后的特征图需要进行Normalization处理
            with tf.variable_scope('conv_4_3_norm'):
                # Normalize through channels dimension (dim=3)
                # 沿着第4个(axis=3)维度进行归一化, 这里主要是沿着channel方向进行归一化
                vgg_conv4_3_norm = tf.nn.l2_normalize(
                    vgg_conv4_3, 3, epsilon=1e-12
                )

                # 文献 ICLR 2016, ParseNet: Looking wider to see better 指出，
                # conv4_3 相比较于其他的 layers，有着不同的 feature scale，使用
                # ParseNet 中的 L2 normalization 技术将 conv4_3 feature map 中每一
                # 个位置的 feature norm scale 到 20，并且在 back-propagation 中学习
                # 这个 scale。
                # 原文：https://blog.csdn.net/u010167269/article/details/52563573
                # Scale.
                scale_initializer = tf.ones(
                    [1, 1, 1, vgg_conv4_3.shape[3]]
                ) * 20.0  # They initialize to 20.0 in paper

                # 创建变量
                # https://blog.csdn.net/u013713117/article/details/66001439
                scale = tf.get_variable(
                    'gamma',
                    dtype=vgg_conv4_3.dtype.base_dtype,
                    initializer=scale_initializer
                )

                # 相当于就是在整体张量上乘以了一个20
                vgg_conv4_3_norm = tf.multiply(vgg_conv4_3_norm, scale)
                tf.summary.histogram('conv4_3_normalized_hist', vgg_conv4_3)

            # 把变量放入一个集合，把很多变量变成一个列表
            # https://blog.csdn.net/UESTC_C2_403/article/details/72415791
            tf.add_to_collection('FEATURE_MAPS', vgg_conv4_3_norm)

            # The original SSD paper uses a modified version of the vgg16
            # network, which we'll modify here
            vgg_network_truncation_endpoint = base_net_endpoints[
                scope + '/vgg_16/conv5/conv5_3']
            tf.summary.histogram(
                'conv5_3_hist',
                vgg_network_truncation_endpoint
            )

            # Extra layers for vgg16 as detailed in paper
            with tf.variable_scope('extra_feature_layers'):
                self._init_vgg16_extra_layers()
                # 改动版本的pool5
                net = tf.nn.max_pool(
                    vgg_network_truncation_endpoint, [1, 3, 3, 1],
                    padding='SAME', strides=[1, 1, 1, 1], name='pool5'
                )
                net = self.conv6(net)
                net = self.activation_fn(net)
                net = self.conv7(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv7_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv8_1(net)
                net = self.activation_fn(net)
                net = self.conv8_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv8_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv9_1(net)
                net = self.activation_fn(net)
                net = self.conv9_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv9_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv10_1(net)
                net = self.activation_fn(net)
                net = self.conv10_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv10_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

                net = self.conv11_1(net)
                net = self.activation_fn(net)
                net = self.conv11_2(net)
                net = self.activation_fn(net)
                tf.summary.histogram('conv11_hist', net)
                tf.add_to_collection('FEATURE_MAPS', net)

            # This parameter determines onto which variables we try to load the
            # pretrained weights
            self.pretrained_weights_scope = scope + '/vgg_16'

        # It's actually an ordered dict
        return utils.convert_collection_to_dict('FEATURE_MAPS')

    def get_trainable_vars(self):
        """
        Returns a list of the variables that are trainable.

        Returns:
            trainable_variables: a tuple of `tf.Variable`.
        """
        return snt.get_variables_in_module(self)
