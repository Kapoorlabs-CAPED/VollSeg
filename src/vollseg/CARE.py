from csbdeep.models import CARE
from csbdeep.internals import train
from .pretrained import (
    get_registered_models,
    get_model_details,
    get_model_instance,
)
import sys

# if IS_TF_1:
#     import tensorflow as tf
# else:
#     import tensorflow.compat.v1 as tf
#     # tf.disable_v2_behavior()


class CARE(CARE):
    """Standard CARE network for image restoration and enhancement.

    Uses a convolutional neural network created by :func:`csbdeep.internals.nets.common_unet`.
    Note that isotropic reconstruction and manifold extraction/projection are not supported here
    (see :class:`csbdeep.models.IsotropicCARE` ).

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
    >>> model = CARE(config, 'my_model')

    Attributes
    ----------
    config : :class:`csbdeep.models.Config`
        Configuration of CARE network, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config, name=None, basedir="."):
        """See class docstring."""
        super().__init__(config=config, name=name, basedir=basedir)

    @classmethod
    def local_from_pretrained(cls, name_or_alias=None):
        try:
            get_model_details(cls, name_or_alias, verbose=True)
            return get_model_instance(cls, name_or_alias)
        except ValueError:
            if name_or_alias is not None:
                print(
                    "Could not find model with name or alias '%s'"
                    % (name_or_alias),
                    file=sys.stderr,
                )
                sys.stderr.flush()
            get_registered_models(cls, verbose=True)

    def train(
        self,
        X,
        Y=None,
        validation_data=None,
        steps_per_epoch=None,
        load_data_sequence=False,
    ):

        epochs = self.config.train_epochs

        if not self._model_prepared:
            self.prepare_for_training()

        if not load_data_sequence:
            # small training data
            steps_per_epoch = len(X) // self.config.train_batch_size
            training_data = train.DataWrapper(
                X,
                Y,
                self.config.train_batch_size,
                length=epochs * steps_per_epoch,
            )

            history = self.keras_model.fit(
                iter(training_data),
                validation_data=validation_data,
                epochs=epochs,
                callbacks=self.callbacks,
                verbose=1,
            )
        else:
            # A sequence object
            history = self.keras_model.fit(
                X,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=self.callbacks,
                verbose=1,
            )
        self._training_finished()

        return history
