import cv2
import numpy as np
from bestvision.datasets.wider_face import WiderFace
from bestvision.transforms.boxes import bbox2abox, abox2bbox
from bestvision.utils.draw import draw_bboxes, draw_keypoints
from bestvision.transforms import matrix2d, warp

def transform(item):
    bboxes = item.pop('bboxes')
    aboxes = bbox2abox(bboxes)
    image = item['image']
    draw_bboxes(image, bboxes)
    h, w = image.shape[:2]
    matrix = matrix2d.shear(0.1, 0.1) @ matrix2d.hflip(w) @ matrix2d.center_rotate_scale_cw((w/2, h/2), 30, 1)
    image = warp.affine(image, matrix, (w, h))
    aboxes = matrix @ aboxes
    item['image'] = image
    item['bboxes'] = abox2bbox(aboxes)
    points = item['keypoints'].reshape(-1, 2)
    points = points @ matrix[:2, :2].T + matrix[:2, 2]
    points = points.reshape(-1, 10)
    item['keypoints'] = points
    return item


if __name__ == "__main__":
    
    data = WiderFace(data_dir="data/WIDER_FACE/WIDER_train")

    for item in data | transform:
        image = draw_bboxes(item['image'], item['bboxes'])
        draw_keypoints(image, item['keypoints'])
        cv2.imshow("v", image)
        cv2.waitKey()
         