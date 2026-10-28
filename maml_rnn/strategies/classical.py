"""Classical supervised training (``model2_.fit(...)`` in the notebook)."""

import tensorflow as tf

from ..callbacks import EarlyStoppingByValue
from ..losses import my_loss
from .base import Strategy


class Classical(Strategy):
    name = "classical"

    def fit(self, model, data):
        p = self.params
        model.compile(optimizer=tf.keras.optimizers.Adam(p["lr"]), loss=my_loss)

        callbacks = []
        if p.get("stop_loss"):
            callbacks.append(EarlyStoppingByValue(monitor="loss", value=p["stop_loss"], verbose=1))

        # validation data is only monitored, it does not influence training or stopping
        validation = (data.val_x, data.val_y) if p.get("monitor_validation", True) else None
        history = model.fit(data.train_x, data.train_y, epochs=p["epochs"], batch_size=p["batch_size"],
                            validation_data=validation, callbacks=callbacks, verbose=self.verbose)
        return {k: list(v) for k, v in history.history.items()}
