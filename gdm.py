class OptimizerGDM(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., momentum=0.):
        super(OptimizerGDM, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)

        weight_updates = self.momentum * layer.weight_momentums + (
                1.0 - self.momentum) * self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates

        layer.weights -= weight_updates

    def post_update_params(self):
        self.iterations += 1