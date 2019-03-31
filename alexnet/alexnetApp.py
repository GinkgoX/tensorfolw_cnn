# AlexNet实现
import tensorflow as tf
import numpy as np


# 卷积层
# group=2时等于AlexNet分上下两部分
def convLayer(x, kHeight, kWidth, strideX, strideY, featureNum, name, padding="SAME", groups=1):
    # 获取channel数
    channel = int(x.get_shape()[-1])
    # 定义卷积的匿名函数
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=[1, strideY, strideX, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel / groups, featureNum])
        b = tf.get_variable("b", shape=[featureNum])
        # 将张量分解成子张量,划分后的输入和权重
        xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
        wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)
        # 分别提取feature map
        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        # feature map整合
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)
        out = tf.nn.bias_add(mergeFeatureMap, b)
        # relu后的结果
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name=scope.name)

# 全连接层
def fcLayer(x, inputD, outputD, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

# alexNet模型
class alexNet(object):
    def __init__(self, x, keepPro, classNum, modelPath="bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.MODELPATH = modelPath
        self.buildCNN()

    def buildCNN(self):
        # 卷积层1
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        # 最大池化层,池化窗口3*3,步长2*2
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        lrn1 = tf.nn.lrn(pool1, depth_radius=2, alpha=2e-05,beta=0.75, bias=1.0, name='norm1')
        # 卷积层2
        conv2 = convLayer(lrn1, 5, 5, 1, 1, 256, "conv2", groups=2)
        # 最大池化层,池化窗口3*3,步长2*2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        lrn2 = tf.nn.lrn(pool2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='lrn2')
        # 卷积层3
        conv3 = convLayer(lrn2, 3, 3, 1, 1, 384, "conv3")
        # 卷积层4
        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)
        # 卷积层5
        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        # 最大池化层,池化窗口3*3,步长2*2
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        # 全连接层1
        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = tf.nn.dropout(fc1, self.KEEPPRO)
        # 全连接层2
        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = tf.nn.dropout(fc2, self.KEEPPRO)
        # 全连接层3
        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    # 加载modeel
    def loadModel(self, sess):
        wDict = np.load(self.MODELPATH, encoding="bytes").item()
        # 模型中的层
        for name in wDict:
            if name not in []:
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            # bias 只有一维
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            # weights
                            sess.run(tf.get_variable('w', trainable=False).assign(p))


import os
import cv2
import caffe_classes


# AlexNet测试
if __name__=='__main__':
    dropoutPro = 1
    classNum = 1000
    testPath = "testimage"
    # 读取测试图像
    testImg = []
    for f in os.listdir(testPath):
        testImg.append(cv2.imread(testPath + "/" + f))

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])
    # alexNet模型
    model = alexNet(x, dropoutPro, classNum)
    score = model.fc3
    print (score)
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载模型
        model.loadModel(sess)
        for i, img in enumerate(testImg):
            # resize成网络输入大小,去均值
            test = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            # test拉成tensor
            test = test.reshape((1, 227, 227, 3))
            # 取概率最大类的下标
            maxx = np.argmax(sess.run(softmax, feed_dict={x: test}))
            # 概率最大的类
            res = caffe_classes.class_names[maxx]
            print(res)
            # 设置字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 显示类的名字
            cv2.putText(img, res, (int(img.shape[0] / 3), int(img.shape[1] / 3)), font, 1, (0, 0, 255), 2)
            # 显示
            cv2.imshow("test", img)
            cv2.waitKey(0)
