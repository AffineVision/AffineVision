import cv2 as cv
import numpy as np

def draw_bboxes(image, bboxes, labels=None, scores=None, colors=None, thickness=0):
    shiftbits = 4
    multiplier = 1 << shiftbits
    bboxes = bboxes.reshape(-1, 2, 2)
    bboxes = (bboxes * multiplier).astype(np.int32)

    # TODO: support label and scores drawing with conrresponding colors

    default_color = (0, 255, 0)

    for box in bboxes:
        pt1 = tuple(box[0].tolist())
        pt2 = tuple(box[1].tolist())
        cv.rectangle(image, pt1, pt2, default_color, thickness=thickness, shift=shiftbits)

    return image

def draw_keypoints(image, keypoints, colors=None, radius=1):
    # keypoints: [N, k, 2]
    # colors: color for each index of keypoints
    shiftbits = 4
    multiplier = 1 << shiftbits
    keypoints = keypoints.reshape(keypoints.shape[0], -1, 2)
    keypoints = (keypoints * multiplier).astype(np.int32)


    for shape in keypoints:
        for i, pt in enumerate(shape):
            pt = tuple(pt.tolist())
            # TODO: different color for each index
            cv.circle(image, pt, radius=radius, thickness=radius, shift=shiftbits)
        
    return image