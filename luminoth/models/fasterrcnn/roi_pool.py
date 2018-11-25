import sonnet as snt
import tensorflow as tf

# Types of RoI "pooling"
CROP = 'crop'
ROI_POOLING = 'roi_pooling'


class ROIPoolingLayer(snt.AbstractModule):
    """ROIPoolingLayer applies ROI Pooling (or tf.crop_and_resize).
    构建RoI Pooliing层

    RoI pooling or RoI extraction is used to extract fixed size features from a
    variable sized feature map using variabled sized bounding boxes. Since we
    have proposals of different shapes and sizes, we need a way to transform
    them into a fixed size Tensor for using FC layers.
    从可变大小的特征图中提取固定大小的特征来使用可变大小的边界框. 为全连接层固定输入大小

    There are two basic ways to do this, the original one in the FasterRCNN's
    paper is RoI Pooling, which as the name suggests, it maxpools directly from
    the region of interest, or proposal, into a fixed size Tensor.
    有两个基本的方法来处理:
    原始的方式是使用RoI Pooling, 如论文所述, 这个正如名字所暗示, 直接最大汇聚感兴趣区域.

    The alternative way uses TensorFlow's image utility operation called,
    `crop_and_resize` which first crops an Tensor using a normalized proposal,
    and then applies extrapolation to resize it to the desired size,
    generating a fixed size Tensor.
    另一个方式是使用tensorflow使用的程序操作, 'crop_and_resize', 首先使用一个标准化的提案,
    来剪裁, 然后应用一个扩展来放缩到想要的大小

    Since there isn't a std support implemenation of RoIPooling, we apply the
    easier but still proven alternatve way.
    这里使用的并不是标准的感兴趣区域汇聚的实现, 使用了第二个方式
    """
    def __init__(self, config, debug=False, name='roi_pooling'):
        super(ROIPoolingLayer, self).__init__(name=name)
        self._pooling_mode = config.pooling_mode.lower() # crop
        self._pooled_width = config.pooled_width
        self._pooled_height = config.pooled_height
        self._pooled_padding = config.padding
        self._debug = debug

    def _get_bboxes(self, roi_proposals, im_shape):
        """
        Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
        in TensorFlow's order (y1, x1, y2, x2).
        获得标准化后的感兴趣区域(0~1之间), 也就是坐标除以对应的宽高

        Args:
            roi_proposals: A Tensor with the bounding boxes of shape
                (total_proposals, 5), where the values for each proposal are
                (x_min, y_min, x_max, y_max).
            im_shape: A Tensor with the shape of the image (height, width).

        Returns:
            bboxes: A Tensor with normalized bounding boxes in TensorFlow's
                format order. Its should is (total_proposals, 4).
                返回标准化后的边界框, 坐标顺序遵循tf的标准
        """
        with tf.name_scope('get_bboxes'):
            # 转化类型
            im_shape = tf.cast(im_shape, tf.float32)

            # 解包操作
            x1, y1, x2, y2 = tf.unstack(
                roi_proposals, axis=1
            )

            # 归一化坐标值
            x1 = x1 / im_shape[1]
            y1 = y1 / im_shape[0]
            x2 = x2 / im_shape[1]
            y2 = y2 / im_shape[0]

            # 打包
            bboxes = tf.stack([y1, x1, y2, x2], axis=1)

            return bboxes

    def _roi_crop(self, roi_proposals, conv_feature_map, im_shape):
        """
        RoI区域剪裁
        :param roi_proposals: 对应于原始图像的区域提案
        :param conv_feature_map: 卷积层特征图
        :param im_shape: 图像大小, 用来将原始图像的区域提案进行缩放, 这里村子这一个固定的
            比例关系, 原图上的RoI与原图的比例, 和特征图上的RoI与特征图的比例是一致的, 所以
            这里进行了放缩, 得到了这个比例, 进而直接用在特征图上就可以实现对应的区域的剪裁
        :return: 对特征图上的各个感兴趣区域池化处理后的7x7的输出
        """
        # Get normalized bounding boxes. (total_num_proposal, 4)
        bboxes = self._get_bboxes(roi_proposals, im_shape)
        # Generate fake batch ids
        bboxes_shape = tf.shape(bboxes)
        batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
        # Apply crop and resize with extracting a crop double the desired size.
        # 该函数的意思是conv_feature_map中根据bboxes[i]来剪裁提取对应的第batch_ids[i]图
        # 像的对应的位置内容, 利用双线性插值或者最邻近插值来进行调整剪裁的大小到
        # [self._pooled_width * 2, self._pooled_height * 2]
        # 这里的batch_ids描述了框与batch中图像的对应关系, 不过这里的值是全0的, 看来对应的
        # 是只处理一个图像, 这里放缩到了14x14的输出
        crops = tf.image.crop_and_resize(
            conv_feature_map, bboxes, batch_ids,
            [self._pooled_width * 2, self._pooled_height * 2], name="crops"
        )

        # Applies max pool with [2,2] kernel to reduce the crops to half the
        # size, and thus having the desired output.
        # 使用2x2大小的核, 在HW的维度上进行步长为2的最大池化
        # 数据格式为NHWC
        prediction_dict = {
            'roi_pool': tf.nn.max_pool(
                crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding=self._pooled_padding
            ),
        }

        if self._debug:
            prediction_dict['bboxes'] = bboxes
            prediction_dict['crops'] = crops
            prediction_dict['batch_ids'] = batch_ids
            prediction_dict['conv_feature_map'] = conv_feature_map

        return prediction_dict

    def _roi_pooling(self, roi_proposals, conv_feature_map, im_shape):
        """
        这里没有实现, 使用roi_crop处理池化操作
        :param roi_proposals:
        :param conv_feature_map:
        :param im_shape:
        :return:
        """
        raise NotImplementedError()

    def _build(self, roi_proposals, conv_feature_map, im_shape):
        if self._pooling_mode == CROP:
            return self._roi_crop(roi_proposals, conv_feature_map, im_shape)
        elif self._pooling_mode == ROI_POOLING:
            return self._roi_pooling(roi_proposals, conv_feature_map, im_shape)
        else:
            raise NotImplementedError(
                'Pooling mode {} does not exist.'.format(self._pooling_mode))
