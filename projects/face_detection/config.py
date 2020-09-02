from yacs.config import CfgNode as CN


_C = CN()

_C.input_size = (320, 320)
_C.strides = [8, 16, 32, 64]
_C.feature_sizes = [(40, 40), (20, 20), (10, 10), (5, 5)]
_C.anchor_layouts = [
    [10, 16, 24],
    [32, 48],
    [64, 96],
    [126, 192, 256]
]

