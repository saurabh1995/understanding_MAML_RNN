import warnings

from tensorflow.keras.callbacks import Callback


class EarlyStoppingByValue(Callback):
    """Stop training as soon as ``logs[monitor] <= value``.

    (``EarlyStoppingByAccuracy`` in the notebook; it is used on the loss.)
    """

    def __init__(self, monitor='loss', value=1e-4, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            warnings.warn(f"Early stopping requires {self.monitor} available!", RuntimeWarning)
            return
        if current <= self.value:
            if self.verbose > 0:
                print(f"Epoch {epoch:05d}: early stopping, {self.monitor}={current:.3e} <= {self.value:.1e}")
            self.model.stop_training = True


# name used in the notebook
EarlyStoppingByAccuracy = EarlyStoppingByValue
