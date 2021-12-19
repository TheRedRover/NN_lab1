class OptimizerBFGS(OptimizerAbstract):
    def __init__(self, network, learning_rate=0.001, decay=0., epsilon=1e-7):
        super(OptimizerBFGS, self).__init__(network, learning_rate)
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.loss_func = None

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        flat_weights = layer.weights.flatten()
        flat_dweights = layer.dweights.flatten()

        I = np.eye(flat_weights.shape[0])
        if not hasattr(layer, f'prev_weights'):
            layer.H = I
            layer.prev_weights = np.zeros_like(flat_weights)
            layer.prev_dweights = np.zeros_like(flat_dweights)
            layer.Bs = layer.prev_weights.dot(-self.current_learning_rate)

        sk = (flat_weights - layer.prev_weights).reshape((flat_weights.shape[0], 1))
        yk = (flat_dweights - layer.prev_dweights).reshape((flat_dweights.shape[0], 1))

        eps = 1e-1
        ys = yk.T.dot(sk)
        sBs = sk.T.dot(layer.Bs)

        # powell damping
        if ys < eps * sBs:
            theta = ((1 - eps) * sBs) / (sBs - ys)
            yk = (theta * yk.flatten() + (1 - theta) * layer.Bs).T

        rho_inv = sk.flatten() @ yk.flatten()
        if abs(rho_inv) < 0.00001:
            rho = 1000
        else:
            rho = 1 / rho_inv

        A1 = (I - rho * (sk @ yk.T))
        A2 = (I - rho * (yk @ sk.T))
        left = A1 @ layer.H @ A2
        layer.H = left + rho * (sk @ sk.T)

        direction = -layer.H @ flat_dweights

        alpha, fail = weak_wolfe(layer, self.loss_func, direction, flat_dweights, self.current_learning_rate)
        if fail:
            pass
        else:
            pass

        layer.Bs = layer.prev_weights.dot(-alpha)

        weight_update = alpha * direction

        layer.weights += weight_update.reshape((layer.weights.shape[0], layer.weights.shape[1]))

        layer.prev_weights = layer.weights.flatten()
        layer.prev_dweights = layer.dweights.flatten()

    def fit(self, x, y, epochs=10000, loss_function=None):
        self._validate_fit(loss_function)

        for epoch in range(epochs):
            def loss_func():
                predictions = self.network.forward(x)
                loss = self.network.loss_function.calculate(predictions, y)
                self.network.backward(predictions, y)
                return loss, predictions

            self.loss_func = loss_func
            loss, predictions = loss_func()

            if (epoch % 100) == 0:
                print(f'epoch: {epoch}, ' +
                      f'loss: {loss:.3f} ' +
                      f'lr: {self.current_learning_rate}')

                self.losses[epoch] = loss

            self.update_weights()

        return self

    def post_update_params(self):
        self.iterations += 1
