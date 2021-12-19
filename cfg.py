class OptimizerCGF(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7, max_update=10):
        super(OptimizerCGF, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.max_update = max_update

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        def calc_beta(dweights, dweights_prev):
            assert dweights.shape[0] == dweights.size
            assert dweights_prev.shape[0] == dweights_prev.size
            return dweights.dot(dweights) / (dweights_prev.dot(dweights_prev) + 1)

        if not hasattr(layer, 'prev_dweights'):
            layer.prev_dweights = layer.dweights.copy()
            layer.weight_p = np.zeros_like(-layer.dweights)

        weight_update = self.current_learning_rate * layer.weight_p

        beta_weights = np.array([calc_beta(dweight, dweight_prev) for (dweight, dweight_prev) in
                                 zip(layer.dweights.T, layer.prev_dweights.T)])

        layer.weight_p = -layer.dweights + beta_weights * layer.weight_p

        layer.prev_dweights = layer.dweights.copy()

        layer.weights += weight_update

    def post_update_params(self):
        self.iterations += 1