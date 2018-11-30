# Faster-RCNN模型简要梳理

> 阅读完成了分模块的代码后, 对主文件 fasterrcnn.py 进行一下梳理.

1. 从基础网络中得到卷积特征图 conv_feature_map
    ```python
    self.base_network = TruncatedBaseNetwork(config.model.base_network)
    ...
    conv_feature_map = self.base_network(
        tf.expand_dims(image, 0), is_training=is_training
    )
    ```
    
    truncated_basse_network.py: Feature extractor for images using a regular CNN.
      
2. 搭建好网络后, 使用_generate_anchors()生成anchors
    
    ```python
    all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
    ... 
    def _generate_anchors(self, feature_map_shape):
       """生成以所有特征图点为中心在原图上对应的的anchors具体坐标结果"""
    ``` 

3. 利用类RPN构建区域提案网络, 生成初步预测
    
    ```python
    self._rpn = RPN(
        self._num_anchors, self._config.model.rpn,
        debug=self._debug, seed=self._seed
    )
    ...
    rpn_prediction = self._rpn(
        conv_feature_map, image_shape, all_anchors,
        gt_boxes=gt_boxes, is_training=is_training
    )
    ```

4. 利用类RCNN构建最后的分支网络部分, 进一步进行调整, 得到最终的预测结果

    ```python
    self._rcnn = RCNN(
        self._num_classes, self._config.model.rcnn,
        debug=self._debug, seed=self._seed
    )
    ...
    proposals = tf.stop_gradient(rpn_prediction['proposals'])
    classification_pred = self._rcnn(
        conv_feature_map, proposals,
        image_shape, self.base_network,
        gt_boxes=gt_boxes, is_training=is_training
    )
    ```

5. 计算损失

    ```python
    self._rpn = RPN(
        self._num_anchors, self._config.model.rpn,
        debug=self._debug, seed=self._seed
    )
    if self._with_rcnn:
        # The RCNN submodule which classifies RPN's proposals and
        # classifies them as background or a specific class.
        self._rcnn = RCNN(
            self._num_classes, self._config.model.rcnn,
            debug=self._debug, seed=self._seed
        )   
 
    ...
 
    # RPN损失
    rpn_loss_dict = self._rpn.loss(
        prediction_dict['rpn_prediction'])
    rpn_loss_dict['rpn_cls_loss'] = (
        rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
    rpn_loss_dict['rpn_reg_loss'] = (
        rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)
     
    prediction_dict['rpn_loss_dict'] = rpn_loss_dict

    # RCNN损失
    if self._with_rcnn:
        rcnn_loss_dict = self._rcnn.loss(
            prediction_dict['classification_prediction'])
        rcnn_loss_dict['rcnn_cls_loss'] = (
            rcnn_loss_dict['rcnn_cls_loss'] * self._rcnn_cls_loss_weight)
        rcnn_loss_dict['rcnn_reg_loss'] = (
            rcnn_loss_dict['rcnn_reg_loss'] * self._rcnn_reg_loss_weight)

        prediction_dict['rcnn_loss_dict'] = rcnn_loss_dict
    else:
        rcnn_loss_dict = {}

    all_losses_items = (
        list(rpn_loss_dict.items()) + list(rcnn_loss_dict.items()))  
    ```
    
要注意的几点:

代码中涉及到了三类数据, 参考anchors, 预测的预测框, 真实框, 这些数据中, 从真实框的角度得到的几个数据都是有后缀`_target`的