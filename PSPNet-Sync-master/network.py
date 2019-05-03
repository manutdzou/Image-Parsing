import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.is_training = is_training
        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32))
        #return tf.get_variable(name, shape, trainable=self.trainable)

    def get_layer_name(self):
        return layer_name
    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
    @layer
    def zero_padding(self, list_input, paddings, name):
        pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])

        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.pad(list_input[i], paddings=pad_mat, name=name)
            list_output.append(output)
        return list_output

    @layer
    def conv(self,
             list_input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = list_input[0].get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding,data_format=DEFAULT_DATAFORMAT)

        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(name, reuse=(i>0)):
                    with tf.device('/cpu:0'):
                        kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
                    output = convolve(list_input[i], kernel)
                    if biased:
                        with tf.device('/cpu:0'):
                            biases = self.make_var('biases', [c_o])
                        output = tf.nn.bias_add(output, biases)
                    if relu:
                        output = tf.nn.relu(output, name=name)
            list_output.append(output)
        return list_output

    @layer
    def atrous_conv(self,
                    list_input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = list_input[0].get_shape()[-1]
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)

        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(name, reuse=(i>0)):
                    with tf.device('/cpu:0'):
                        kernel = self.make_var('weights', shape=[k_h, k_w, c_i, c_o])
                    output = convolve(list_input[i], kernel)
                    if biased:
                        with tf.device('/cpu:0'):
                            biases = self.make_var('biases', [c_o])
                        output = tf.nn.bias_add(output, biases)
                    if relu:
                        output = tf.nn.relu(output, name=name)
            list_output.append(output)
        return list_output

    @layer
    def relu(self, list_input, name):
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.nn.relu(list_input[i], name=name)
            list_output.append(output)
        return list_output

    @layer
    def max_pool(self, list_input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.nn.max_pool(list_input[i],
                                        ksize=[1, k_h, k_w, 1],
                                        strides=[1, s_h, s_w, 1],
                                        padding=padding,
                                        name=name,
                                        data_format=DEFAULT_DATAFORMAT)
            list_output.append(output)
        return list_output


    @layer
    def avg_pool(self, list_input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.nn.avg_pool(list_input[i],
                                        ksize=[1, k_h, k_w, 1],
                                        strides=[1, s_h, s_w, 1],
                                        padding=padding,
                                        name=name,
                                        data_format=DEFAULT_DATAFORMAT)
            list_output.append(output)
        return list_output

    @layer
    def lrn(self, list_input, radius, alpha, beta, name, bias=1.0):
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.nn.local_response_normalization(list_input[i],
                                                            depth_radius=radius,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            bias=bias,
                                                            name=name)
            list_output.append(output)
        return list_output


    @layer
    def concat(self, list_input, axis, name):
        list_output = []
        for i in range(len(list_input[0])):
            with tf.device('/gpu:%d' % i):
                to_concat = []
                for j in range(len(list_input)):
                    to_concat.append(list_input[j][i])
                output = tf.concat(axis=axis, values=to_concat, name=name)
            list_output.append(output)
        return list_output

    @layer
    def add(self, list_input, name):
        list_output = []
        for i in range(len(list_input[0])):
            with tf.device('/gpu:%d' % i):
                to_add = []
                for j in range(len(list_input)):
                    to_add.append(list_input[j][i])
                output = tf.add_n(to_add, name=name)
            list_output.append(output)
        return list_output

    @layer
    def fc(self, list_input, num_out, name, relu=True):
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(name, reuse=(i>0)):
                    input_shape = list_input[i].get_shape()
                    if input_shape.ndims == 4:
                        # The input is spatial. Vectorize it first.
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(list_input[i], [-1, dim])
                    else:
                        feed_in, dim = (list_input[i], input_shape[-1].value)
                    with tf.device('/cpu:0'):
                        weights = self.make_var('weights', shape=[dim, num_out])
                        biases = self.make_var('biases', [num_out])
                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    output = op(feed_in, weights, biases, name=scope.name)
            list_output.append(output)
        return list_output

    @layer
    def softmax(self, list_input, name):
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                input_shape = map(lambda v: v.value, list_input[i].get_shape())
                if len(input_shape) > 2:
                    # For certain models (like NiN), the singleton spatial dimensions
                    # need to be explicitly squeezed, since they're not broadcast-able
                    # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
                    if input_shape[1] == 1 and input_shape[2] == 1:
                        output = tf.squeeze(list_input[i], squeeze_dims=[1, 2])
                    else:
                        output = tf.nn.softmax(list_input[i], name)
            list_output.append(output)
        return list_output

    @layer
    def batch_normalization(self, list_input, name, relu=False):
        decay = 0.9
        epsilon = 1e-5

        means = []
        square_means = []

        x_shape = list_input[0].get_shape().as_list()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))

        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                batch_mean = tf.reduce_mean(list_input[i], axis)
                batch_square_mean = tf.reduce_mean(tf.square(list_input[i]), axis)
            means.append(batch_mean)
            square_means.append(batch_square_mean)

        with tf.device('/cpu:0'):  # if your GPUs have NVLinks and you've install NCCL2, you can change `/cpu:0` to `/gpu:0`
            #num = reduce(lambda x,y: x * y, x_shape[0:-1]) * len(list_input)
            mean = tf.reduce_mean(means, axis=0)
            variance = tf.reduce_mean(square_means, axis=0) - tf.square(mean)
            #variance *= tf.cast(num, tf.float32) / tf.cast(num-1, tf.float32)  # unbiased variance

            with tf.variable_scope(name):
                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer= tf.zeros_initializer, trainable= False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32, initializer= tf.ones_initializer, trainable= False)

            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay, zero_debias=False)
        
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(name, reuse=(i>0)):
                    if not tf.get_variable_scope().reuse and self.is_training:
                        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                            list_input[i] = tf.identity(list_input[i])
                    with tf.device('/cpu:0'):
                        beta = tf.get_variable('beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
                        gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

                    if self.is_training:
                        output =  tf.nn.batch_normalization(list_input[i], mean, variance, beta, gamma, epsilon)
                    else:
                        output =  tf.nn.batch_normalization(list_input[i], moving_mean, moving_variance, beta, gamma, epsilon)
                    if relu:
                        output = tf.nn.relu(output)
            list_output.append(output)
        return list_output

    @layer
    def dropout(self, list_input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.nn.dropout(list_input[i], keep, name=name)
            list_output.append(output)
        return list_output

    @layer
    def resize_bilinear(self, list_input, size, name):
        list_output = []
        for i in range(len(list_input)):
            with tf.device('/gpu:%d' % i):
                output = tf.image.resize_bilinear(list_input[i], size=size, align_corners=True, name=name)
            list_output.append(output)
        return list_output
