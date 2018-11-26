# Faster-RCNN模型简要梳理

> 阅读完成了分模块的代码后, 对主文件 fasterrcnn.py 进行一下梳理.

1. 从基础网络中得到卷积特征图 conv_feature_map
    ```python
     conv_feature_map = self.base_network(
            tf.expand_dims(image, 0), is_training=is_training
        )
    ```
      
2. 搭建好网络后, 使用_generate_anchors()生成anchors
    
    ```python
    all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
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
