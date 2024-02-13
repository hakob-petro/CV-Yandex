import numpy as np

from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.momentum * parameter - self.lr * parameter_grad
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.where(inputs >= 0, inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * np.where(self.forward_inputs >= 0, 1, 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        norm_input = inputs - np.max(inputs, axis=1)[:, None]
        return np.exp(norm_input) / np.sum(np.exp(norm_input), axis=1)[:, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        n, d = grad_outputs.shape
        grad = np.zeros_like(grad_outputs)
        di = np.diag_indices(d)
        for i in range(n):
            soft_grad = np.zeros((d, d))
            soft_grad[di] = self.forward_outputs[i]
            soft_grad -= np.dot(self.forward_outputs[i][:, None], self.forward_outputs[i][:, None].T)
            grad[i, :] = np.dot(grad_outputs[i], soft_grad)
        return grad
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return np.dot(inputs, self.weights) + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/

        self.weights_grad = np.dot(self.forward_inputs.T, grad_outputs)
        self.biases_grad = np.sum(grad_outputs, axis=0)
        return np.dot(grad_outputs, self.weights.T)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        mask = y_gt == 1
        result = np.sum(np.sum(y_gt[mask] * np.log(1e-8 + y_pred[mask])))
        return -np.array([result / y_gt.shape[0]])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        grads = np.zeros_like(y_gt)
        mask = y_gt == 1
        grads[mask] = (-1 / y_gt.shape[0]) / np.clip(y_pred[mask], eps, None)
        return grads
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optimizer = SGD(lr=3e-3)
    model = Model(loss, optimizer)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(128, (784, )))
    model.add(ReLU())
    model.add(Dense(1024))
    model.add(ReLU())
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=2)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/
    b, d, h, w = inputs.shape
    kb, kd, kh, kw = kernels.shape
    kernels = kernels[:, :, ::-1, ::-1]

    if padding > 0:
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        inputs = np.pad(inputs, pad_width=pad_width)
        b, d, h, w = inputs.shape

    # Based on awesome tutorial by S. Do. : https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3
    # Or we can do fully vectorized variant as in example below, but I didn't figure out how to deal with chans
    batch_stride, channel_stride, rows_stride, columns_stride = inputs.strides

    out_h = int(h - kh + 1)
    out_w = int(w - kw + 1)

    new_shape = (b, d, out_h, out_w, kh, kw)
    new_strides = (batch_stride, channel_stride, rows_stride, columns_stride, rows_stride, columns_stride)
    input_windows = np.lib.stride_tricks.as_strided(inputs, new_shape, new_strides)

    output = np.einsum('bchwkt,fckt->bfhw', input_windows, kernels)
    return output


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        return convolve(inputs, self.kernels, (self.kernel_size - 1) // 2) + self.biases[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        kernels_grad = convolve(
            self.forward_inputs[:, :, ::-1, ::-1].transpose(1, 0, 2, 3),
            grad_outputs.transpose(1, 0, 2, 3),
            padding=(self.kernel_size - 1) // 2
        )
        self.kernels_grad = kernels_grad.transpose(1, 0, 2, 3)

        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        input_grads = convolve(
            grad_outputs,
            self.kernels[:, :, ::-1, ::-1].transpose(1, 0, 2, 3),
            padding=(self.kernel_size - 1) // 2
        )
        return input_grads
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.max_ids = None
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        b, d, _, _ = inputs.shape
        batch_stride, channel_stride, rows_stride, columns_stride = inputs.strides

        new_shape = (b, *self.output_shape, self.pool_size, self.pool_size)
        new_strides = (
            batch_stride,
            channel_stride,
            self.pool_size * rows_stride,
            self.pool_size * columns_stride,
            rows_stride,
            columns_stride)

        input_windows = np.lib.stride_tricks.as_strided(inputs, new_shape, new_strides)

        # Return the result of pooling
        if self.pool_mode == 'max':
            self.max_ids = np.argmax(input_windows.reshape(b, *self.output_shape, self.pool_size ** 2), axis=-1, keepdims=True)
            return input_windows.max(axis=(-1, -2))
        elif self.pool_mode == 'avg':
            return input_windows.mean(axis=(-1, -2))
            # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        p = self.pool_size
        b, d, h, w = self.forward_inputs.shape
        result = grad_outputs[..., None, None]

        if self.pool_mode == "avg":
            result = result * np.ones((b, d, h // p, w // p, p, p))
            result /= p ** 2
            return result.transpose(0, 1, 2, 4, 3, 5).reshape(b, d, h, w)
        elif self.pool_mode == 'max':
            zeros = np.zeros((b, d, h // p, w // p, p * p))
            np.put_along_axis(zeros, self.max_ids, 1, axis=-1)
            result = result * zeros.reshape(b, d, h // p, w // p, p, p)
            return result.transpose(0, 1, 2, 4, 3, 5).reshape(b, d, h, w)

    # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def update_running_params(self, mean, var) -> None:
        self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
        self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3))
            var = np.var(inputs, axis=(0, 2, 3))
            self.update_running_params(mean, var)
            self.forward_inverse_std = 1 / np.sqrt(var + 1e-8)[..., None, None]
            self.forward_centered_inputs = inputs - mean[..., None, None]
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
            return self.gamma[..., None, None] * self.forward_normalized_inputs + self.beta[..., None, None]
        else:
            mean = self.running_mean.copy()
            var = self.running_var.copy()
            inv_std = 1 / np.sqrt(var + 1e-8)[..., None, None]
            centered_input = inputs - mean[..., None, None]
            inputs_norm = centered_input * inv_std
            return self.gamma[..., None, None] * inputs_norm + self.beta[..., None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/
        scaled_grad = self.gamma[..., None, None] * grad_outputs
        b, d, h, w = self.forward_inputs.shape
        scale_coef = 0.5 * b * h * w + 1e-8

        var_grad = -0.5 * self.forward_inverse_std ** 3 * np.sum(scaled_grad * self.forward_centered_inputs, axis=(0, 2, 3))[..., None, None]

        mean_grad = -self.forward_inverse_std * np.sum(scaled_grad, axis=(0, 2, 3))[..., None, None]
        mean_grad -= var_grad * np.sum(self.forward_centered_inputs, axis=(0, 2, 3))[..., None, None] / scale_coef

        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        return scaled_grad * self.forward_inverse_std + \
            var_grad * self.forward_centered_inputs / scale_coef +\
            mean_grad / scale_coef * 2
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(*self.forward_inputs.shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.rand(*inputs.shape) >= self.p
            output = self.forward_mask * inputs
        else:
            output = (1 - self.p) * inputs
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return self.forward_mask * grad_outputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    opt = SGDMomentum(lr=1e-2, momentum=0.9)
    model = Model(loss, opt)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(6, 3, (3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Conv2D(16, 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Conv2D(16, 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Conv2D(8, 3))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2, 'max'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dropout(0.2))
    model.add(Dense(84))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=3, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
