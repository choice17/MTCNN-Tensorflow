#!/usr/bin/python3
# -*- coding: utf-8 -*-

#MIT License
#
#Copyright (c) 2018 Iván de Paz Centeno
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# IMPORTANT:
#
# This code is derivated from the MTCNN implementation of David Sandberg for Facenet
# (https://github.com/davidsandberg/facenet/)
# It has been rebuilt from scratch, taking the David Sandberg's implementation as a reference.
# The code improves the readibility, fixes several mistakes in the definition of the network (layer names)
# and provides the keypoints of faces as outputs along with the bounding boxes.
#

import cv2
import numpy as np
import pkg_resources
import tensorflow as tf
from mtcnn.layer_factory import LayerFactory
from mtcnn.network import Network
from mtcnn.exceptions import InvalidImage

__author__ = "Iván de Paz Centeno"


class PNet(Network):
    """
    Network to propose areas with faces.
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, None, None, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=10, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=16, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_conv(name='conv4-1', kernel_size=(1, 1), channels_output=2, stride_size=(1, 1), relu=False)
        layer_factory.new_softmax(name='prob1', axis=3)

        layer_factory.new_conv(name='conv4-2', kernel_size=(1, 1), channels_output=4, stride_size=(1, 1),
                               input_layer_name='prelu3', relu=False)

    def _feed(self, image):
        return self._session.run(['pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'], feed_dict={'pnet/input:0': image})


class RNet(Network):
    """
    Network to refine the areas proposed by PNet
    """

    def _config(self):

        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 24, 24, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=28, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=48, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(2, 2), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_fully_connected(name='fc1', output_count=128, relu=False)  # shouldn't the name be "fc1"?
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)   # shouldn't the name be "fc2-1"?
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu4')

    def _feed(self, image):
        return self._session.run(['rnet/fc2-2/fc2-2:0', 'rnet/prob1:0'], feed_dict={'rnet/input:0': image})


class ONet(Network):
    """
    Network to retrieve the keypoints
    """
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 48, 48, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_max_pool(name='pool3', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv4', kernel_size=(2, 2), channels_output=128, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc1', output_count=256, relu=False)
        layer_factory.new_prelu(name='prelu5')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu5')

        layer_factory.new_fully_connected(name='fc2-3', output_count=10, relu=False, input_layer_name='prelu5')

    def _feed(self, image):
        return self._session.run(['onet/fc2-2/fc2-2:0', 'onet/fc2-3/fc2-3:0', 'onet/prob1:0'],
                                 feed_dict={'onet/input:0': image})


class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """
    def __init__(self, pad_result: tuple=None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class MTCNN(object):
    """
    Allows to perform MTCNN Detection ->
        a) Detection of faces (with the confidence probability)
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    """

    def __init__(self, weights_file: str=None, min_face_size: int=20, steps_threshold: list=None,
                 scale_factor: float=0.709):
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        if weights_file is None:
            weights_file = pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy')

        self.__min_face_size = min_face_size
        self.__steps_threshold = steps_threshold
        self.__scale_factor = scale_factor

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.__graph = tf.Graph()

        with self.__graph.as_default():
            self.__session = tf.Session(config=config, graph=self.__graph)

            weights = np.load(weights_file).item()
            self.__pnet = PNet(self.__session, False)
            self.__pnet.set_weights(weights['PNet'])

            self.__rnet = RNet(self.__session, False)
            self.__rnet.set_weights(weights['RNet'])

            self.__onet = ONet(self.__session, False)
            self.__onet.set_weights(weights['ONet'])

    @property
    def min_face_size(self):
        return self.__min_face_size
    
    @min_face_size.setter
    def min_face_size(self, mfc=20):
        try:
            self.__min_face_size = int(mfc)
        except ValueError:
            self.__min_face_size = 20
    
    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.__scale_factor, factor_count)]
            min_layer = min_layer * self.__scale_factor
            factor_count += 1

        return scales

    @staticmethod
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(imap, reg, scale, t):

        # use heatmap to generate bounding boxes
        stride = 2
        cellsize = 12

        imap = np.transpose(imap)
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])

        y, x = np.where(imap >= t)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        score = imap[(y, x)]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

        if reg.size == 0:
            reg = np.empty(shape=(0, 3))

        bb = np.transpose(np.vstack([y, x]))

        q1 = np.fix((stride * bb + 1)/scale)
        q2 = np.fix((stride * bb + cellsize)/scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

        return boundingbox, reg

    @staticmethod
    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.

        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    @staticmethod
    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __rerec(bbox):
        # convert bbox to square
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        l = np.maximum(w, h)
        bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(boundingbox, reg):
        # calibrate bounding boxes
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox

    def detect_faces(self, img, net_type='ONet') -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        if net_type == 'PNet': stage_included_idx = 1
        elif net_type == 'RNet': stage_included_idx = 2
        elif net_type == 'ONet': stage_included_idx = 3

        height, width, _ = img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self.__min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)


        stages = [self.__stage1, self.__stage2, self.__stage3][:stage_include_idx]

        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in stages:
            result = stage(img, result[0], result[1])

        [total_boxes, points] = result

        bounding_boxes = []

        for bounding_box, keypoints in zip(total_boxes, points.T):

            bounding_boxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                        'left_eye': (int(keypoints[0]), int(keypoints[5])),
                        'right_eye': (int(keypoints[1]), int(keypoints[6])),
                        'nose': (int(keypoints[2]), int(keypoints[7])),
                        'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                        'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }
            )

        return bounding_boxes

    def __stage1(self, image, scales: list, stage_status: StageStatus):
        """
        First stage of the MTCNN.
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        total_boxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image, scale)

            img_x = np.expand_dims(scaled_image, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            out = self.__pnet.feed(img_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = self.__generate_bounding_box(out1[0, :, :, 1].copy(),
                                                    out0[0, :, :, :].copy(), scale, self.__steps_threshold[0])

            # inter-scale nms
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numboxes = total_boxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = self.__rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return total_boxes, status

    def __stage2(self, img, total_boxes, stage_status:StageStatus):
        """
        Second stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """

        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, stage_status

        # second stage
        tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

        for k in range(0, num_boxes):
            tmp = np.zeros((int(stage_status.tmph[k]), int(stage_status.tmpw[k]), 3))

            tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = \
                img[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

            else:
                return np.empty(shape=(0,)), stage_status


        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self.__rnet.feed(tempimg1)

        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self.__steps_threshold[1])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = self.__nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.__rerec(total_boxes.copy())

        return total_boxes, stage_status

    def __stage3(self, img, total_boxes, stage_status: StageStatus):
        """
        Third stage of the MTCNN.

        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """
        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, np.empty(shape=(0,))

        total_boxes = np.fix(total_boxes).astype(np.int32)

        status = StageStatus(self.__pad(total_boxes.copy(), stage_status.width, stage_status.height),
                             width=stage_status.width, height=stage_status.height)

        tempimg = np.zeros((48, 48, 3, num_boxes))

        for k in range(0, num_boxes):

            tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

            tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = \
                img[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty(shape=(0,)), np.empty(shape=(0,))

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self.__onet.feed(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])

        score = out2[1, :]

        points = out1

        ipass = np.where(score > self.__steps_threshold[2])

        points = points[:, ipass[0]]

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1

        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1

        if total_boxes.shape[0] > 0:
            total_boxes = self.__bbreg(total_boxes.copy(), np.transpose(mv))
            pick = self.__nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

        return total_boxes, points

    def __del__(self):
        self.__session.close()

    '''training section'''
    def read_and_decode(filename_queue, label_type, shape):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)

    image = (image - 127.5) * (1. / 128.0)
    image.set_shape([shape * shape * 3])
    image = tf.reshape(image, [shape, shape, 3])
    label = tf.decode_raw(features['label_raw'], tf.float32)

    if label_type == 'cls':
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        label.set_shape([2])
    elif label_type == 'bbx':
        label.set_shape([4])
    elif label_type == 'pts':
        label.set_shape([10])

    return image, label


def inputs(filename, batch_size, num_epochs, label_type, shape):

    with tf.device('/cpu:0'):
        if not num_epochs:
            num_epochs = None

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                filename, num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, label_type, shape)

        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            min_after_dequeue=1000)

        return images, sparse_labels


def train_net(Net, training_data, base_lr, loss_weight,
              train_mode, num_epochs=[1, None, None],
              batch_size=64, weight_decay=4e-3,
              load_model=False, load_filename=None,
              save_model=False, save_filename=None,
              num_iter_to_save=10000,
              gpu_memory_fraction=1):

    images = []
    labels = []
    tasks = ['cls', 'bbx', 'pts']
    shape = 12
    if Net.__name__ == 'RNet':
        shape = 24
    elif Net.__name__ == 'ONet':
        shape = 48
    for index in range(train_mode):
        image, label = inputs(filename=[training_data[index]],
                              batch_size=batch_size,
                              num_epochs=num_epochs[index],
                              label_type=tasks[index],
                              shape=shape)
        images.append(image)
        labels.append(label)
    while len(images) is not 3:
        images.append(tf.placeholder(tf.float32, [None, shape, shape, 3]))
        labels.append(tf.placeholder(tf.float32))
    net = Net((('cls', images[0]), ('bbx', images[1]), ('pts', images[2])),
              weight_decay_coeff=weight_decay)

    print('all trainable variables:')
    all_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in all_vars:
        print(var)

    print('all local variable:')
    local_variables = tf.local_variables()
    for l_v in local_variables:
        print(l_v.name)

    prefix = str(all_vars[0].name[0:5])
    out_put = net.get_all_output()
    cls_output = tf.reshape(out_put[0], [-1, 2])
    bbx_output = tf.reshape(out_put[1], [-1, 4])
    pts_output = tf.reshape(out_put[2], [-1, 10])

    # cls loss
    softmax_loss = loss_weight[0] * \
        tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels[0],
                                                logits=cls_output))
    weight_losses_cls = net.get_weight_decay()['cls']
    losses_cls = softmax_loss + tf.add_n(weight_losses_cls)

    # bbx loss
    square_bbx_loss = loss_weight[1] * \
        tf.reduce_mean(tf.squared_difference(bbx_output, labels[1]))
    weight_losses_bbx = net.get_weight_decay()['bbx']
    losses_bbx = square_bbx_loss + tf.add_n(weight_losses_bbx)

    # pts loss
    square_pts_loss = loss_weight[2] * \
        tf.reduce_mean(tf.squared_difference(pts_output, labels[2]))
    weight_losses_pts = net.get_weight_decay()['pts']
    losses_pts = square_pts_loss + tf.add_n(weight_losses_pts)

    global_step_cls = tf.Variable(1, name='global_step_cls', trainable=False)
    global_step_bbx = tf.Variable(1, name='global_step_bbx', trainable=False)
    global_step_pts = tf.Variable(1, name='global_step_pts', trainable=False)

    train_cls = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_cls, global_step=global_step_cls)
    train_bbx = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_bbx, global_step=global_step_bbx)
    train_pts = tf.train.AdamOptimizer(learning_rate=base_lr) \
                        .minimize(losses_pts, global_step=global_step_pts)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    config.gpu_options.allow_growth = True

    loss_agg_cls = [0]
    loss_agg_bbx = [0]
    loss_agg_pts = [0]
    step_value = [1, 1, 1]

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=200000)
        if load_model:
            saver.restore(sess, load_filename)
        else:
            net.load(load_filename, sess, prefix)
        if save_model:
            save_dir = os.path.split(save_filename)[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                choic = np.random.randint(0, train_mode)
                if choic == 0:
                    _, loss_value_cls, step_value[0] = sess.run(
                        [train_cls, softmax_loss, global_step_cls])
                    loss_agg_cls.append(loss_value_cls)
                elif choic == 1:
                    _, loss_value_bbx, step_value[1] = sess.run(
                        [train_bbx, square_bbx_loss, global_step_bbx])
                    loss_agg_bbx.append(loss_value_bbx)
                else:
                    _, loss_value_pts, step_value[2] = sess.run(
                        [train_pts, square_pts_loss, global_step_pts])
                    loss_agg_pts.append(loss_value_pts)

                if sum(step_value) % (100 * train_mode) == 0:
                    agg_cls = sum(loss_agg_cls) / len(loss_agg_cls)
                    agg_bbx = sum(loss_agg_bbx) / len(loss_agg_bbx)
                    agg_pts = sum(loss_agg_pts) / len(loss_agg_pts)
                    print(
                        'Step %d for cls: loss = %.5f' %
                        (step_value[0], agg_cls), end='. ')
                    print(
                        'Step %d for bbx: loss = %.5f' %
                        (step_value[1], agg_bbx), end='. ')
                    print(
                        'Step %d for pts: loss = %.5f' %
                        (step_value[2], agg_pts))
                    loss_agg_cls = [0]
                    loss_agg_bbx = [0]
                    loss_agg_pts = [0]

                if save_model and (step_value[0] % num_iter_to_save == 0):
                    saver.save(sess, save_filename, global_step=step_value[0])

        except tf.errors.OutOfRangeError:
            print(
                'Done training for %d epochs, %d steps.' %
                (num_epochs[0], step_value[0]))
        finally:
            coord.request_stop()

        coord.join(threads)