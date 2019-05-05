import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.cuda import dnn
from collections import deque
from difflib import get_close_matches
from inspect import getargspec
from itertools import chain
from warnings import warn
#Some implementations are based on Python module Lasagne


def floatX(arr):
    # numpy array ``theano.config.floatX``.

    return np.asarray(arr, dtype=theano.config.floatX)

def softmax(x):
    #`\\varphi(\\mathbf{x})_j = \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`

    return theano.tensor.nnet.softmax(x)
_rng = np.random
#Get the package-level random number generator.
def get_rng():
    
    return _rng

def expression(input):

    if isinstance(input, theano.gof.Variable):
        return input
    else:
        try:
            return theano.tensor.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s is not a Theano expression and "
                            "cannot be wrapped as a Theano constant (original "
                            "exception: %s)" % (type(input), e))


def sharevars(expressions):

    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]

def newparam(l):
    #Filters duplicates of iterable.

    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list
def Coerce_tuple(x, N, t=None):
    #Coerce a value to a tuple of given length.

    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X

def create_param(spec, shape, name=None):
    
    #create and initialize Theano shared variables for layer parameters initialize.



    import numbers  # to check if argument is a number
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    err_prefix = "cannot initialize parameter %s: " % name
    if callable(spec):
        spec = spec(shape)
        err_prefix += "the %s returned by the provided callable"
    else:
        err_prefix += "the provided %s"

    if isinstance(spec, numbers.Number) or isinstance(spec, np.generic) \
            and spec.dtype.kind in 'biufc':
        spec = np.asarray(spec)

    if isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise ValueError("%s has shape %s, should be %s" %
                             (err_prefix % "numpy array", spec.shape, shape))
        bcast = tuple(s == 1 for s in shape)
        spec = theano.shared(spec, broadcastable=bcast)

    if isinstance(spec, theano.Variable):
        if spec.ndim != len(shape):
            raise ValueError("%s has %d dimensions, should be %d" %
                             (err_prefix % "Theano variable", spec.ndim,
                              len(shape)))
        # We only assign a name if the user hasn't done so already.
        if not spec.name:
            spec.name = name
        return spec

    else:
        if "callable" in err_prefix:
            raise TypeError("%s is not a numpy array or a Theano expression" %
                            (err_prefix % "value"))
        else:
            raise TypeError("%s is not a numpy array, a Theano expression, "
                            "or a callable" % (err_prefix % "spec"))

class SingleLayer(object):
      #a single layer of a neural network

    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

        if any(d is not None and d <= 0 for d in self.input_shape):
            raise ValueError((
                "Cannot create Layer with a non-positive input_shape "
                "dimension. input_shape=%r, self.name=%r") % (
                    self.input_shape, self.name))

    @property
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)
        if any(isinstance(s, T.Variable) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_params(self, unwrap_shared=True, **tags):

        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        if unwrap_shared:
            return sharevars(result)
        else:
            return result

    def get_output_shape_for(self, input_shape):

        return input_shape

    def get_output_for(self, input, **kwargs):

        raise NotImplementedError

    def add_param(self, spec, shape, name=None, **tags):

        # prefix the param name with the layer name if it exists
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)
        # create shared variable, or pass through given variable/expression
        param = create_param(spec, shape, name)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param
class MultipleLayer(SingleLayer):
       #a layer that aggregates input from multiple layers.

    def __init__(self, incomings, name=None):
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

    @SingleLayer.output_shape.getter
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shapes)
        if any(isinstance(s, T.Variable) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_output_shape_for(self, input_shapes):

        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):

        raise NotImplementedError
class InputLayer(SingleLayer):
    #a symbolic variable that represents a network input.
    
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = tuple(shape)
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError((
                "Cannot create InputLayer with a non-positive shape "
                "dimension. shape=%r, self.name=%r") % (
                    self.shape, name))

        ndim = len(shape)
        if input_var is None:
            # create the right TensorType for the given number of dimensions
            input_var_type = T.TensorType(theano.config.floatX, [False] * ndim)
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name
        self.params = OrderedDict()

    @SingleLayer.output_shape.getter
    def output_shape(self):
        return self.shape

class InitParameter(object):
    #Base class for InitParameter.


    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()


class Unidistribution(InitParameter):
    #Sample initial weights from the uniform distribution.


    def __init__(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)

    def sample(self, shape):
        return floatX(get_rng().uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Xavier(InitParameter):
    #Xavier weight InitParameter.


    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)

class XUnidistribution(Xavier):

    def __init__(self, gain=1.0, c01b=False):
        super(XUnidistribution, self).__init__(Unidistribution, gain, c01b)
class Constant(InitParameter):

    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)

def rectify(x):
    #math:`\\varphi(x) = \\max(0, x)`
    return theano.tensor.nnet.relu(x)

def conv_output_length(input_length, filter_size, stride, pad=1):
    #compute the output size of a convolution operation

    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length
class BaseConvLayer(SingleLayer):

    #Convolutional layer base class    
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=1,
                 untie_biases=False,
                 W=XUnidistribution(), b=Constant(0.),
                 nonlinearity=rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = Coerce_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = Coerce_tuple(stride, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = Coerce_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = Coerce_tuple(pad, n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        #Get the shape of the weight matrix `W`.

        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):

        raise NotImplementedError()

def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    
    #Compute the output length of a pooling operator along a single dimension.


    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


def pool_2d(input, **kwargs):

    try:
        return T.signal.pool.pool_2d(input, **kwargs)
    except TypeError:  # pragma: no cover
        # convert from new to old interface
        kwargs['ds'] = kwargs.pop('ws')
        kwargs['st'] = kwargs.pop('stride')
        kwargs['padding'] = kwargs.pop('pad')
        return T.signal.pool.pool_2d(input, **kwargs)


class PoolLayer(SingleLayer):
    #2D pooling layer. Performs 2D mean or max-pooling over the two trailing axes of a 4D input tensor.

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='average_exc_pad', **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)

        self.pool_size = Coerce_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = Coerce_tuple(stride, 2)

        self.pad = Coerce_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = pool_2d(input,
                         ws=self.pool_size,
                         stride=self.stride,
                         ignore_border=self.ignore_border,
                         pad=self.pad,
                         mode=self.mode,
                         )
        return pooled

    
def get_all_params(layer, unwrap_shared=True, **tags):
    
    #Returns a list of Theano shared variables or expressions that parameterize the layer.

    layers = get_all_layers(layer)
    params = chain.from_iterable(l.get_params(
            unwrap_shared=unwrap_shared, **tags) for l in layers)
    return newparam(params)
    
def set_all_param_values(layer, values, **tags):    
    #Given a list of numpy arrays
    params = get_all_params(layer, **tags)
    if len(params) != len(values):
        raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))

    for p, v in zip(params, values):
        if p.get_value().shape != v.shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, v.shape))
        else:
            p.set_value(v)

def get_all_layers(layer, treat_as_input=None):
    #gathers all layers below one or more given
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        layer = queue[0]
        if layer is None:
            queue.popleft()
        elif layer not in seen:
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result
            
def get_output(layer_or_layers, inputs=None, **kwargs):
    
    #Computes the output of the network at one or more given layers.
 
    accepted_kwargs = {'deterministic'}    
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, expression(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = expression(inputs)
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MultipleLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
            try:
                names, _, _, defaults = getargspec(layer.get_output_for)
            except TypeError:
                pass
            else:
                if defaults is not None:
                    accepted_kwargs |= set(names[-len(defaults):])
            accepted_kwargs |= set(layer.get_output_kwargs)
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]
    
class ConvLayer(BaseConvLayer):
    
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=1, untie_biases=False, W=XUnidistribution(),
                 b=Constant(0.), nonlinearity=rectify,
                 flip_filters=False, **kwargs):
        super(ConvLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, n=2, **kwargs)

    def convolve(self, input, **kwargs):
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode
                              )
        return conved
