import  numpy as np
import  tensorflow as tf


class resnet(object):

    def __init__(self,hps, image,labels,model):

        self.hps=hps            # 超参数
        self.image =image       # 输入图像格式[batch ,img_w,img_h,3]
        self.labels =labels     # 输入标签格式[batch,label]
        self.model =model       # 模式训练或者测试


    def _resdual(self ,x,in_filter,out_filter,stride,activate_before_residual=False):
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):

                x = self._b