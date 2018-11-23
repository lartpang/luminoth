import tensorflow as tf


def get_width_upright(bboxes):
    """
    计算图向后中心点坐标
    :param bboxes: 框集合, 是左上角和右下角坐标的集合
    :return: width, height, urx, ury: 其中urx, ury为框的中心坐标
    """
    with tf.name_scope('BoundingBoxTransform/get_width_upright'):
        bboxes = tf.cast(bboxes, tf.float32)
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = x2 - x1 + 1.
        height = y2 - y1 + 1.

        # Calculate up right point of bbox (urx = up right x)
        urx = x1 + .5 * width
        ury = y1 + .5 * height

        return width, height, urx, ury


def encode(bboxes, gt_boxes, variances=None):
    """
    计算ground truth相对anchor的偏移量和缩放量
    :param bboxes: anchors边框坐标
    :param gt_boxes: 真实的边框坐标
    :param variances: 比例参数
    :return: ground truth相对anchor的偏移量和缩放量量
    """
    with tf.name_scope('BoundingBoxTransform/encode'):
        (bboxes_width, bboxes_height,
         bboxes_urx, bboxes_ury) = get_width_upright(bboxes)

        (gt_boxes_width, gt_boxes_height,
         gt_boxes_urx, gt_boxes_ury) = get_width_upright(gt_boxes)

        if variances is None:
            variances = [1., 1.]

        targets_dx = (gt_boxes_urx - bboxes_urx)/(bboxes_width * variances[0])
        targets_dy = (gt_boxes_ury - bboxes_ury)/(bboxes_height * variances[0])

        targets_dw = tf.log(gt_boxes_width / bboxes_width) / variances[1]
        targets_dh = tf.log(gt_boxes_height / bboxes_height) / variances[1]

        targets = tf.concat(
            [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)

        return targets


def decode(roi, deltas, variances=None):
    """
    利用感兴趣区域的真实坐标值和偏移量来进行调整
    使用参考边框（xcenter, ycenter, width, height），学习预测偏移量（Δxcenter,
    Δycenter,Δwidth,Δheight），因此我们只得到一些小数值的预测结果并挪动参考变量就可以达
    到更好的拟合结果

    :param roi: 参考边框, 这里是左上角和右下角的坐标
    :param deltas: 偏移量dx, dy, dw, dh
    :param variances: 一个用于调节偏移量的比例参数
    :return: 从偏移量和参考值得到的预测的左上角和右下角坐标
    """
    with tf.name_scope('BoundingBoxTransform/decode'):
        (roi_width, roi_height,
         roi_urx, roi_ury) = get_width_upright(roi)

        # 拆分偏移量
        dx, dy, dw, dh = tf.split(deltas, 4, axis=1)

        if variances is None:
            variances = [1., 1.]

        # 根据预测框相对anchor中心位置的偏移量以及宽高的缩放量, 进而的到预结果
        pred_ur_x = dx * roi_width * variances[0] + roi_urx
        pred_ur_y = dy * roi_height * variances[0] + roi_ury
        pred_w = tf.exp(dw * variances[1]) * roi_width
        pred_h = tf.exp(dh * variances[1]) * roi_height

        # 从预测框的中心坐标转化为预测框左上角的坐标
        bbox_x1 = pred_ur_x - 0.5 * pred_w
        bbox_y1 = pred_ur_y - 0.5 * pred_h

        # This -1. extra is different from reference implementation.
        # 转化为右下角的坐标
        bbox_x2 = pred_ur_x + 0.5 * pred_w - 1.
        bbox_y2 = pred_ur_y + 0.5 * pred_h - 1.

        # 返回预测出来的左上角和右下角坐标
        bboxes = tf.concat(
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=1)

        return bboxes


def clip_boxes(bboxes, imshape):
    """
    Clips bounding boxes to image boundaries based on image shape.
    对四个坐标值超出图像大小的的进行调整
    直接判断与图像宽高的大小关系

    Args:
        bboxes: Tensor with shape (num_bboxes, 4)
            where point order is x1, y1, x2, y2.

        imshape: Tensor with shape (2, )
            where the first value is height and the next is width.

    Returns
        Tensor with same shape as bboxes but making sure that none
        of the bboxes are outside the image.
    """
    with tf.name_scope('BoundingBoxTransform/clip_bboxes'):
        # 转换数据类型
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        imshape = tf.cast(imshape, dtype=tf.float32)

        # 参数拆分
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
        width = imshape[1]
        height = imshape[0]
        # x1,x2小于宽度大于0
        x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

        # y1,y2小于高度大于0
        y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

        # 参数组合
        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes


def change_order(bboxes):
    """Change bounding box encoding order.
    更改使用的坐标的顺序, 因为这里使用的坐标顺序和tensorflow的规定的有差异

    TensorFlow works with the (y_min, x_min, y_max, x_max) order while we work
    with the (x_min, y_min, x_max, y_min).

    While both encoding options have its advantages and disadvantages we
    decided to use the (x_min, y_min, x_max, y_min), forcing use to switch to
    TensorFlow's every time we want to use a std function that handles bounding
    boxes.

    Args:
        bboxes: A Tensor of shape (total_bboxes, 4)

    Returns:
        bboxes: A Tensor of shape (total_bboxes, 4) with the order swaped.
    """
    with tf.name_scope('BoundingBoxTransform/change_order'):
        first_min, second_min, first_max, second_max = tf.unstack(
            bboxes, axis=1
        )
        bboxes = tf.stack(
            [second_min, first_min, second_max, first_max], axis=1
        )
        return bboxes


if __name__ == '__main__':
    import numpy as np

    bboxes = tf.placeholder(tf.float32)
    bboxes_val = [[10, 10, 20, 22]]

    gt_boxes = tf.placeholder(tf.float32)
    gt_boxes_val = [[11, 13, 34, 31]]

    imshape = tf.placeholder(tf.int32)
    imshape_val = (100, 100)

    deltas = encode(bboxes, gt_boxes)
    decoded_bboxes = decode(bboxes, deltas)
    final_decoded_bboxes = clip_boxes(decoded_bboxes, imshape)

    with tf.Session() as sess:
        final_decoded_bboxes = sess.run(final_decoded_bboxes, feed_dict={
            bboxes: bboxes_val,
            gt_boxes: gt_boxes_val,
            imshape: imshape_val,
        })

        assert np.all(gt_boxes_val == final_decoded_bboxes)
