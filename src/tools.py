# MIT License
#
# Copyright (c) 2017 Baoming Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import re

import numpy as np
import tensorflow as tf
import cv2

import datetime


"""mine"""

def showtime(func):

    def func_wrapper(*args, **kwargs):
        print('Starting %s %s' % (func.__name__, datetime.datetime.now()))
        _ =  func(*args, **kwargs)
        print('Finish !!! %s %s' % (func.__name__, datetime.datetime.now()))
        return _
            
    

    return func_wrapper
    





"""From official MTCNN tools"""
"""import from mtcnn-wangbm"""

def view_bar(num, total):

    rate = float(num) / total
    rate_num = int(rate * 100) + 1
    r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()


def int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_model_filenames(model_dir):

    files = os.listdir(model_dir)
    pnet = [s for s in files if 'pnet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    rnet = [s for s in files if 'rnet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    onet = [s for s in files if 'onet' in s and
                                os.path.isdir(os.path.join(model_dir, s))]
    if pnet and rnet and onet:
        if len(pnet) == 1 and len(rnet) == 1 and len(onet) == 1:
            _, pnet_data = get_meta_data(os.path.join(model_dir, pnet[0]))
            _, rnet_data = get_meta_data(os.path.join(model_dir, rnet[0]))
            _, onet_data = get_meta_data(os.path.join(model_dir, onet[0]))
            return (pnet_data, rnet_data, onet_data)
        else:
            raise ValueError('There should not be more '
                             'than one dir for each model')
    else:
        return get_meta_data(model_dir)


def get_meta_data(model_dir):

    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model '
                         'directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than '
                         'one meta file in the model directory (%s)'
                         % model_dir)
    meta_file = meta_files[0]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^[A-Za-z]+-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                data_file = step_str.groups()[0]
    return (os.path.join(model_dir, meta_file),
            os.path.join(model_dir, data_file))


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):

    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * (1. / 128.0)
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out0[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out0[1, :]
        points = out2
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:10:2, :] = np.tile(w, (5, 1)) * \
            (points[0:10:2, :] + 1) / 2 + \
            np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[1:11:2, :] = np.tile(h, (5, 1)) * \
            (points[1:11:2, :] + 1) / 2 + \
            np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points


def detect_face_12net(img, minsize, pnet, threshold, factor):

    factor_count = 0
    total_boxes = np.empty((0, 9))
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * (1. / 128.0)
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold)

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
    return total_boxes


def detect_face_24net(img, minsize, pnet, rnet, threshold, factor):

    factor_count = 0
    total_boxes = np.empty((0, 9))
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        out = pnet(img_x)
        out0 = out[0]
        out1 = out[1]
        boxes, _ = generateBoundingBox(out0[0, :, :, 1].copy(),
                                       out1[0, :, :, :].copy(),
                                       scale,
                                       threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4,
                                              total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
            total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k],
                :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if (tmp.shape[0] > 0 and tmp.shape[1] > 0 or
                    tmp.shape[0] == 0 and tmp.shape[1] == 0):
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 0, 1, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out0[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(),
                                 np.expand_dims(score[ipass].copy(), 1)])
        mv = out1[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.5, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
    return total_boxes


def nms(boxes, threshold, method):

    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    s_sort = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while s_sort.size > 0:
        i = s_sort[-1]
        pick[counter] = i
        counter += 1
        idx = s_sort[0:-1]
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
        s_sort = s_sort[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def bbreg(boundingbox, reg):

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


def generateBoundingBox(imap, reg, scale, t):

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
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)],
                                  dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


def pad(total_boxes, w, h):

    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
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


def rerec(bboxA):

    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    size = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - size * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - size * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(size, (2, 1)))
    return bboxA


def imresample(img, sz):

    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data


def IoU(box, boxes):

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):

    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

""" ===============From MTCNN-Tensorlfow===================""" 
def logger(msg):
    """
        log message
    """
    now = time.ctime()
    print("[%s] %s" % (now, msg))

def createDir(p):
    if not os.path.exists(p):
        os.mkdir(p)
#shuffle in the same way
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def getDataFromTxt(txt, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    #get dirname
    dirname = os.path.dirname(txt)
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path
        # bounding box, (x1, y1, x2, y2)
        #bbox = (components[1], components[2], components[3], components[4])
        bbox = (components[1], components[3], components[2], components[4])        
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int,bbox))
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        #normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        result.append((img_path, BBox(bbox), landmark))
    return result

def getPatch(img, bbox, point, padding):
    """
        Get a patch iamge around the given point in bbox with padding
        point: relative_point in [0, 1] in bbox
    """
    point_x = bbox.x + point[0] * bbox.w
    point_y = bbox.y + point[1] * bbox.h
    patch_left = point_x - bbox.w * padding
    patch_right = point_x + bbox.w * padding
    patch_top = point_y - bbox.h * padding
    patch_bottom = point_y + bbox.h * padding
    patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
    patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
    return patch, patch_bbox


def processImage(imgs):
    """
        process images before feeding to CNNs
        imgs: N x 1 x W x H
    """
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        imgs[i] = (img - 127.5) / 128
    return imgs

def dataArgument(data):
    """
        dataArguments
        data:
            imgs: N x 1 x W x H
            bbox: N x BBox
            landmarks: N x 10
    """
    pass



class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]
        
        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    #offset
    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    #absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    #landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    #change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    #self.w bounding-box width
    #self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

#just for RNet and ONet 
def read_single_tfrecord(tfrecord_file, batch_size, net):

    filename_queue = tf.train.string_input_producer([tfrecord_file],shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),#one image  one record
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/roi': tf.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.FixedLenFeature([10], tf.float32)
        }
    )
    if net == 'PNet':
        image_size = 12
    elif net == 'RNet':
        image_size = 24
    else:
        image_size = 48
    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = (tf.cast(image, tf.float32)-127.5) / 128
    
    # image = tf.image.per_image_standardization(image)
    label = tf.cast(image_features['image/label'], tf.float32)
    roi = tf.cast(image_features['image/roi'],tf.float32)
    landmark = tf.cast(image_features['image/landmark'],tf.float32)
    image, label,roi,landmark = tf.train.batch(
        [image, label,roi,landmark],
        batch_size=batch_size,
        num_threads=2,
        capacity=1 * batch_size
    )
    label = tf.reshape(label, [batch_size])
    roi = tf.reshape(roi,[batch_size,4])
    landmark = tf.reshape(landmark,[batch_size,10])
    return image, label, roi,landmark

def read_multi_tfrecords(tfrecord_files, batch_sizes, net):

    pos_dir, part_dir,neg_dir, landmark_dir = tfrecord_files
    pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size = batch_sizes
    pos_image, pos_label, pos_roi, pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, net)    
    part_image, part_label, part_roi, part_landmark = read_single_tfrecord(part_dir, part_batch_size, net)    
    neg_image, neg_label, neg_roi,neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, net)
    landmark_image, landmark_label, landmark_roi, landmark_landmark = read_single_tfrecord(landmark_dir, landmark_batch_size, net)
    
    print (pos_image.get_shape())
    print (part_image.get_shape())
    print (neg_image.get_shape())
    print (landmark_image.get_shape())

    images = tf.concat([pos_image, part_image,neg_image,landmark_image], 0, name="concat/image")
    labels = tf.concat([pos_label, part_label,neg_label,landmark_label], 0, name="concat/label")
    rois = tf.concat([pos_roi, part_roi, neg_roi, landmark_roi], 0, name="concat/roi")
    landmarks = tf.concat([pos_landmark, part_landmark, neg_landmark, landmark_landmark], 0, name="concat/landmark")
    
    print (images.get_shape())
    print (labels.get_shape())
    print (rois.get_shape())
    print (landmarks.get_shape())
    return images,labels,rois,landmarks
    
def read():
    BATCH_SIZE = 64
    net = 'PNet'
    dataset_dir = "imglists/PNet"
    landmark_dir = os.path.join(dataset_dir,'train_PNet_ALL_few.tfrecord_shuffle')
    images, labels, rois, landmarks  = read_single_tfrecord(landmark_dir, BATCH_SIZE, net)
    
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:

                im_batch, label_batch, roi_batch, landmark_batch = sess.run([images, labels,rois,landmarks])
                i+=1
        except tf.errors.OutOfRangeError:
            print("finish！！！")
        finally:
            coord.request_stop()
        coord.join(threads)

    num_landmark = len(np.where(label_batch==-2)[0])
    print (num_landmark)
    num_batch, h, w, c = im_batch.shape
    for i in range(num_batch):

        cc = cv2.resize(im_batch[i],(120,120))
        print (label_batch)
        
        for j in range(5):
            cv2.circle(cc, (int(landmark_batch[i][2*j]*120), int(landmark_batch[i][2*j+1]*120)), 3, (0, 0, 255))
        
        cv2.imshow("lala",cc)
        cv2.waitKey()

def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images#all image pathes
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

def read_and_write_annotation(base_dir, dir):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(dir, 'r')
    f = open('/home/thinkjoy/data/mtcnn_data/imagelists/train.txt', 'w')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        im = cv2.imread(imagepath)
        h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            text = ''
            text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2] - 1
            ymax = ymin + face_box[3] - 1
            text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    f.close()
    return data

def get_path(base_dir, filename):
    return os.path.join(base_dir, filename)

def get_minibatch(imdb, num_classes, im_size):
    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        h, w, c = im.shape
        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = im/127.5
        processed_ims.append(im_tensor)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    im_array = np.asarray(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array}

    return data, label

def get_testbatch(imdb):
    # print(len(imdb))
    assert len(imdb) == 1, "Single batch only"
    # im = cv2.imread(imdb[0])
    im = cv2.imread(imdb)
    im_array = im
    data = {'data': im_array}
    return data

def show_landmark(face, landmark):
    """
        view face with landmark for visualization
    """
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (0,0,0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)


#rotate(img, f_bbox,bbox.reprojectLandmark(landmarkGt), 5)
#img: the whole image
#BBox:object
#landmark:
#alpha:angle
def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    #whole image rotate
    #pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,(img.shape[1],img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    #crop face 
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)


def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    #mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth
    return (face_flipped_by_x, landmark_)

def randomShift(landmarkGt, shift):
    """
        Random Shift one time
    """
    diff = np.random.rand(5, 2)
    diff = (2*diff - 1) * shift
    landmarkP = landmarkGt + diff
    return landmarkP

def randomShiftWithArgument(landmarkGt, shift):
    """
        Random Shift more
    """
    N = 2
    landmarkPs = np.zeros((N, 5, 2))
    for i in range(N):
        landmarkPs[i] = randomShift(landmarkGt, shift)
    return landmarkPs
