import numpy as np

# naming convention: 
# bbox means bounding box encoded with bounding points
# cbox means bounding box encoded with center and size
# abox means affine box encoded as affine matrix
#                                  [  u_x,   u_y, cx]
#                                  [  v_x,   v_y, cy]
#                                  [    0,     0,  1]
# rbox means bounding rotated box encoding with [cx, cy, w, h, angle]
def bbox2abox(bboxes, radians=None):
    # bboxes: [*, 4]
    # radians: box angle in radian
    vectors = (bboxes[..., 2:] - bboxes[..., :2]) / 2
    centers = (bboxes[..., 2:] + bboxes[..., :2]) / 2

    x_vec = vectors[..., 0]
    y_vec = vectors[..., 1]
    zeros = np.zeros(x_vec.shape)
    ones = np.ones(x_vec.shape)
    aboxes = np.stack([x_vec, zeros, centers[..., 0], zeros, y_vec, centers[..., 1], zeros, zeros, ones], axis=-1)
    # reshape
    shape = (*x_vec.shape, 3, 3)
    aboxes = aboxes.reshape(shape)

    if radians is not None:
        cos = np.cos(radians)
        sin = np.sin(radians)
        rotate = np.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], axis=-1).reshape(shape)
        aboxes = rotate @ aboxes

    return aboxes

def abox2bbox(aboxes):
    """covnert affine boxes to bounding point box

    reference: https://www.iquilezles.org/www/articles/ellipses/ellipses.htm

    Args:
        aboxes ([np.ndarray]): affine boxes shape with [*, 3, 3]
    """

    c = aboxes[..., :2, 2]
    e = np.linalg.norm(aboxes[..., :2, :2], ord=2, axis=-1)
    bboxes = np.concatenate([c - e, c + e], axis=-1)
    return bboxes

def bbox2cbox(bboxes):
    sizes = bboxes[..., 2:] - bboxes[..., :2]
    centers = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    cboxes = np.concatenate([centers, sizes], axis=-1)
    return cboxes


def cbox2bbox(bboxes):
    halfs = bboxes[..., 2:] / 2
    lt = bboxes[..., :2] - halfs
    rb = bboxes[..., 2:] + halfs
    bboxes = np.concatenate([lt, rb], axis=-1)
    return bboxes

def cbox2abox(bboxes):
    # TODO
    pass
