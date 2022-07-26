# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division

import warnings
import numpy as np
from six import string_types
from .CARE import CARE
from .pretrained import get_registered_models, get_model_details, get_model_instance
import sys
import tensorflow as tf
# if IS_TF_1:
#     import tensorflow as tf
# else:
#     import tensorflow.compat.v1 as tf
#     # tf.disable_v2_behavior()


class UNET(CARE):
    """Standard UNET network for image segmentation based on CARE network.

    Uses a convolutional neural network created by :func:`csbdeep.internals.nets.common_unet`.
   
    Parameters
    ----------
    config : :class:`csbdeep.models.Config` or None
        Valid configuration of CARE network (see :func:`Config.is_valid`).
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.
        Use ``None`` to disable saving (or loading) any data to (or from) disk (regardless of other parameters).

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Example
    -------
    >>> model = UNET(config, 'my_model')

    Attributes
    ----------
    config : :class:`csbdeep.models.Config`
        Configuration of UNET network, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir='.'):
        """See class docstring."""
        super(UNET, self).__init__(config=config, name=name, basedir=basedir)


    def _build(self):
        return nets.common_unet(
            n_dim           = self.config.n_dim,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(self.config.unet_input_shape)

    @classmethod   
    def local_from_pretrained(cls, name_or_alias=None):
          try:
              get_model_details(cls, name_or_alias, verbose=True)
              return get_model_instance(cls, name_or_alias)
          except ValueError:
              if name_or_alias is not None:
                  print("Could not find model with name or alias '%s'" % (name_or_alias), file=sys.stderr)
                  sys.stderr.flush()
              get_registered_models(cls, verbose=True)

    @property
    def _config_class(self):
        return Config
