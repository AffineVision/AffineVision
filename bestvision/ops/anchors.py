import torch
import torch.nn as nn

from numbers import Number


def _canonical_rbox_anchor(anchor):
    x, y, r = 0, 0, 0
    if isinstance(anchor, Number):
        w, h = anchor, anchor
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 1:
        w = h = anchor[0]
    elif isinstance(anchor, (list, tuple)) and len(anchor) == 2:
        w, h = anchor
    elif isinstance(anchor, (list, tuple)) and  len(anchor) == 4:
        x, y, w, h = anchor
    elif isinstance(anchor, (list, tuple)) and  len(anchor) == 5:
        x, y, w, h, r = anchor
    else:
        raise ValueError(f'Invalid anchor setting with value: {anchor}')
    return [x, y, w, h, r]

class RBoxAnchors(nn.Module):
    def __init__(self, input_size, strides, feature_sizes, anchor_layouts):
        super().__init__()

        self.input_size = input_size
        self.strides = strides
        self.feature_sizes = feature_sizes
        self.anchor_layouts = anchor_layouts

        anchors = self.generate_anchors(strides, feature_sizes, anchor_layouts)
        self.register_buffer('anchors', anchors)

    @staticmethod
    def generate_layer_anchors(stride, feature_size, anchor_layout):
        anchor_layout = [_canonical_rbox_anchor(v) for v in anchor_layout]
        anchor_layout = torch.tensor(anchor_layout, dtype=torch.float32)  # [k, 5]

        # generate offset grid
        fw, fh = feature_size
        vx = torch.arange(0.5, fw, dtype=torch.float32) * stride
        vy = torch.arange(0.5, fh, dtype=torch.float32) * stride
        vy, vx = torch.meshgrid(vy, vx)
        offsets = torch.stack([vx, vy], dim=-1) # [fh, fw, 2]

        anchors = anchor_layout.repeat(fh, fw, 1, 1) # [fh, fw, k, 5]
        anchors[:, :, :, :2] += offsets[:, :, None, :] # [fh, fw, k, 5]
        return anchors

    @staticmethod
    def generate_anchors(strides, feature_sizes, anchor_layouts):
        anchors = []
        for stride, feature_size, anchor_layout in zip(strides, feature_sizes, anchor_layouts):
            layer_anchors = RBoxAnchors.generate_layer_anchors(stride, feature_size, anchor_layout)
            layer_anchors = layer_anchors.reshape(-1, 5)
            anchors.append(layer_anchors)
        anchors = torch.cat(anchors, dim=0)
        return anchors

    def update(self, input_size, feature_sizes):
        pass

    def encode_bboxes(self, bboxes):
        pass

    def decode_bboxes(self, deltas):
        pass

    def encode_keypoints(self, keypoints):
        pass

    def decode_keypoints(self, deltas):
        pass

    def match(self, labels, bboxes):
        pass

    def forward(self, labels, bboxes):
        # TODO : return match results
        # Return
        #  - max_iou
        #  - best_iou_index
        #  - best_iou_value
        pass
