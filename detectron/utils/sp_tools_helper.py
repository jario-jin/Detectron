
"""Spire Image Tools helper functions.
Details see https://pan.baidu.com/s/196x3tbpPV7vdJqsXfONk3g
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os
import numpy as np
import pycocotools.mask as mask_util

from detectron.core.config import cfg


def output_annotations(img_name, img, cls_boxes_i, cls_segs_i, cls_kps_i, classes):
    if cfg.DEBUG.OUTPUT_ANNO:
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, cfg.DEBUG.ANNO_PATH)):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR, cfg.DEBUG.ANNO_PATH))
        img_name = os.path.basename(img_name)
        f = open(os.path.join(cfg.OUTPUT_DIR, cfg.DEBUG.ANNO_PATH, img_name + '.json'), 'w')
        f.write(
            b'{{"file_name":"{:s}","height":{},"width":{},"annos":['.format(img_name, img.shape[0], img.shape[1]))
        only_one = True
        for ci, boxes in enumerate(cls_boxes_i):
            if ci > 0 and isinstance(boxes, np.ndarray):
                if cls_segs_i is not None and len(cls_segs_i) > ci and len(cls_segs_i[ci]) > 0:
                    masks = mask_util.decode(cls_segs_i[ci])
                for b in range(len(boxes)):
                    if boxes[b, 4] > cfg.DEBUG.BBOX_THRS:
                        if not only_one:
                            f.write(b',')
                        x = boxes[b, 0]
                        y = boxes[b, 1]
                        w = boxes[b, 2] - x + 1
                        h = boxes[b, 3] - y + 1
                        f.write(
                            b'{{"area":{},"bbox":[{},{},{},{}],"score":{},"category_name":"{}"'.format(
                                w * h, x, y, w, h, boxes[b, 4], classes[ci]))
                        if cls_segs_i is not None and len(cls_segs_i) > ci and len(cls_segs_i[ci]) > 0:
                            f.write(b',"segmentation":[')
                            mask = masks[..., b]
                            (_, contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            point_one = True
                            for contour in contours:
                                for cp in range(contour.shape[0]):
                                    if point_one:
                                        f.write(b'{},{}'.format(contour[cp, 0, 0], contour[cp, 0, 1]))
                                    else:
                                        f.write(b',{},{}'.format(contour[cp, 0, 0], contour[cp, 0, 1]))
                                    point_one = False
                            f.write(b']')
                        if cls_kps_i is not None and len(cls_kps_i) > ci and len(cls_kps_i[ci]) > 0:
                            f.write(b',"keypoints":[')
                            point_one = True
                            kp_ci_b = cls_kps_i[ci][b]
                            for ki in range(kp_ci_b.shape[1]):
                                kp_thresh = 0
                                if kp_ci_b[2, ki] > 2:
                                    kp_thresh = 2
                                if point_one:
                                    f.write(b'{},{},{}'.format(round(kp_ci_b[0, ki]), round(kp_ci_b[1, ki]), kp_thresh))
                                else:
                                    f.write(
                                        b',{},{},{}'.format(round(kp_ci_b[0, ki]), round(kp_ci_b[1, ki]), kp_thresh))
                                point_one = False
                            f.write(b']')
                        f.write(b'}')
                        only_one = False
        f.write(b']}')
        f.close()
