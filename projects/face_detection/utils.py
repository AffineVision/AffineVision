import cv2
import numpy as np

from affinevision.transforms import matrix2d, warp
from affinevision.transforms.boxes import bbox_affine

class NaiveTransformer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def __call__(self, item):
        image = item['image']
        h, w = image.shape[:2]

        scale = np.array(self.input_size) / np.array([w, h])
        matrix = matrix2d.scale(scale)

        image = warp.affine(image, matrix, self.input_size)
        bboxes = item['bboxes']
        bboxes = bbox_affine(bboxes, matrix)

        keypoints = item['keypoints']
        keypoints = keypoints.reshape(keypoints.shape[0], -1, 2)
        keypoints = keypoints @ matrix[:2, :2].T + matrix[:2, 2]
        keypoints = keypoints.reshape(keypoints.shape[0], -1)

        res = {
            'image': image,
            'bboxes': bboxes,
            'keypoints': keypoints
        }

        return res
class FaceBoxTransformer(object):
    def __init__(self, *args, **kwargs):
        pass


class JigsawTransformer(object):
    def __init__(self, dataset, input_size, ):
        self.dataset = dataset
        self.input_size = input_size

        pass

    def _dynamic_jigsaw(self, dst, target_region, ):
        pass