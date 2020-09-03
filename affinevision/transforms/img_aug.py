import copy

class ImgAug(object):
    def __call__(self, image):
        raise NotImplementedError

    def __or__(self, aug: ImgAug):
        class _Inherit(aug.__class__):
            def __call__(self, image):
                image = self.__call__(image)
                image = super().__call__(image)
                return aug(image)
        res = copy.copy(self)
        self.__class__ = _Inherit