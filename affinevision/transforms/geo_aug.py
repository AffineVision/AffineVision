from . import matrix2d


class GeoAug(object):
    def __init__(self, size, angle, scale, translate, hflip, vflip, shear=None):
        """Geometry augmentation genrator

        Args:
            size ([tuple, Number]): size of image
            angle ([Number, tuple]): rotate jitter angle range in degree 
            scale ([Number, tuple]): scale jitter range
            translate ([Number, tuple]): translation jitter, ratio base on size
            hflip ([float]): horizontal flip probability
            vflip ([float]): vertical flip probability
            shear ([Number, shear], optional): shear jitter range
        """
        self.size = size
        self.angle = angle
        self.scale = scale
        self.tranlate = translate
        self.hflip = hflip
        self.vflip = vflip
        self.shear = shear
    
    def __call__(self):
        """Generate augmentation matrix
        Return:
            matrix: geometry augmetation matrix
            hflip:  is hflip invoked
            vflip:  is vflip invoked
        """
        # TODO
        raise NotImplementedError()