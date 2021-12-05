from config import LAYER_TYPE_UNIDENTIFIED


class LayerAbstract:

    def __init__(self, *args, **kwargs):
        self.type = LAYER_TYPE_UNIDENTIFIED

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Method "forward" is not implemented in the {self.__class__}')

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f'Method "backward" is not implemented in the {self.__class__}')

