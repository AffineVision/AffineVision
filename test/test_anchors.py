from bestvision.ops.anchors import RBoxAnchors

from projects.face_detection.config import _C as cfg

if __name__ == "__main__":
    anchors = RBoxAnchors(cfg.input_size, cfg.strides, cfg.feature_sizes, cfg.anchor_layouts)
    anchors.cuda()
    pass