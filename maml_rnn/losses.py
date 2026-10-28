import tensorflow as tf


def my_loss(y_true, y_pred):
    """MSE averaged over batch and time, kept per output feature (shape ``(features,)``).

    Gradients of a vector loss are those of its sum; Keras' ``compile`` reduces it to the
    mean for reporting.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 0])


def compute_loss(forward, x, y, loss_fn=my_loss):
    """``forward`` is any callable ``x -> prediction`` (a Keras model or a shadow-model closure)."""
    logits = forward(x)
    return loss_fn(y, logits), logits


def loss_norm(loss):
    """Scalar used by the notebook for logging and stopping (``np.linalg.norm(loss)``)."""
    return float(tf.norm(loss))
