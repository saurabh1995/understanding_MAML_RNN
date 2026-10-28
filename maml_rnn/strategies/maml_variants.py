"""The six MAML variants of the notebook, in the order they were developed.

Common setting (unchanged across all variants)
----------------------------------------------
* A *task* is one training sequence (``batch_size = 1``): its support set.
* The *query set* is the fixed held-out test split, shared by all tasks (only v5 differs).
* Inner step:  ``theta' = theta - inner_lr * dL_support(theta)/dtheta``
* Meta-gradient: ``dL_query(theta')/dtheta``, second order: the inner gradient is
  differentiated through, and ``theta'`` is evaluated on a shadow model (see
  ``maml_rnn/functional.py``).
* Outer optimiser: Adam.
* Parameters that a variant does *not* adapt keep their meta-parameter value in the
  query pass, and the direct query-loss path to them is cut. In the notebook this came
  from the copy holding its own variables; here it is done with ``tf.stop_gradient``.
  They still receive a (small) second-order meta-gradient, because the inner gradient of
  the adapted parameters depends on them. This holds in the notebook too, so e.g. the
  Dense layers in v1 are still updated by the outer optimiser.

Evolution
---------
v1  ``MAML``  ->  ``MAML_v1_Basic``
    First working MAML. One inner step. Adapts only the RNN cells' weights
    (``kernel/recurrent_kernel/bias``, i.e. LSTM-style cells). The outer Adam step runs
    after *every task*, so it is closer to an online/SGD meta-update than to the batched
    meta-update of the MAML paper. inner_lr = 0.002.

v2  ``MAML_RLIF``  ->  ``MAML_v2_RLIF``
    Same algorithm, rewritten for the spiking ``RSNN_vthr`` cell. The adapted parameters
    become ``input_weights``, ``recurrent_weights`` and the trainable threshold ``thr``.
    In the notebook it used the global inner_lr = 0.001.

v3  ``MAML_LMU``  ->  ``MAML_v3_MetaBatch``
    Two changes:
    (a) meta-gradients are *accumulated over all tasks* and applied in a single Adam step
        per epoch (proper batched MAML outer loop);
    (b) Dense layers are now adapted as well, in addition to the spiking RNN layers.
    Stopping criterion: ||mean query loss|| < 1e-3.

v4  ``MAML_LMU_multiple``  ->  ``MAML_v4_MultiInner``
    Adds ``inner_epochs`` to take several inner steps. The adaptation branch targeted LSTM
    + Dense layers (the spiking-RNN branch was commented out). Known issue, reproduced
    on purpose: every inner iteration recomputes the gradient on the *unchanged*
    meta-parameters, so k inner steps give exactly the result of one step (k-1 steps are
    wasted). With spiking cells only the Dense layers are adapted.

v5  ``MAML_LMU_multiple_``  ->  ``MAML_v5_TaskQuery``
    Switches adaptation back to spiking RNN + Dense layers and gets a second dataset
    argument (``test_dataset``) to give every task its own query batch instead of the
    whole test split. In the notebook the query batch (``xx, yy``) was never bound, so
    the function could not run. Here the query batches come from cycling through
    ``test_dataset``. The inner-loop issue of v4 is still present.

v6  ``MAML_LMU_multiple_adapt_inner_outer``  ->  ``MAML_v6_AdaptInnerOuter``  (used for the results)
    Fixes the multi-step inner loop: after every inner step the adapted weights are
    written into the model (``set_weights``), so the next step starts from them. The
    meta-parameters are restored before the meta-gradient is taken. Query set is the
    fixed test split again. Other changes:
    * inner_lr = 0.01;
    * the validation loss is evaluated after every outer step;
    * stops when the meta-gradient norm < 1e-4 or the validation loss < 1e-4.
    Caveat: writing the weights back into variables cuts the second-order path through
    the earlier inner steps. With ``inner_steps = 1`` (the setting used for training)
    this is exact second-order MAML. With more steps only the last step is
    differentiated through.
"""

import tensorflow as tf

from ..functional import ShadowModel, param_layers
from ..losses import compute_loss, loss_norm, my_loss
from ..data import make_dataset
from .base import Strategy


class _MAMLBase(Strategy):
    """Shared machinery. Subclasses choose the outer loop and the class-level switches."""

    per_task_update = False     # v1/v2: optimizer step after every task
    write_back = False          # v6: inner steps are chained through the model's variables
    validate = False            # v6: validation loss after every outer step

    # -- setup --------------------------------------------------------------------------
    def _setup(self, model):
        p = self.params
        self.inner_lr = p["inner_lr"]
        self.inner_steps = p.get("inner_steps", 1)
        self.optimizer = tf.keras.optimizers.Adam(p["outer_lr"])
        self.shadow = ShadowModel(self.build_fn)
        self.adapt_mask = [self._is_adapted(layer) for layer in param_layers(model)]
        if not any(self.adapt_mask):
            raise ValueError(f"adapt='{p['adapt']}' selects no parameters of this model")
        if self.verbose:
            n = sum(self.adapt_mask)
            print(f"[{self.name}] adapting {n}/{len(self.adapt_mask)} parameter tensors (adapt='{p['adapt']}')")

    def _is_adapted(self, layer):
        mode = self.params["adapt"]
        is_rnn = isinstance(layer, tf.keras.layers.RNN)
        is_dense = isinstance(layer, tf.keras.layers.Dense)
        if mode == "rnn":
            return is_rnn
        if mode == "all":
            return is_rnn or is_dense
        if mode == "lstm_dense":
            return (is_rnn and isinstance(layer.cell, tf.keras.layers.LSTMCell)) or is_dense
        raise ValueError(f"Unknown adapt mode '{mode}' (rnn | all | lstm_dense)")

    def _task_dataset(self, data):
        p = self.params
        return make_dataset(data.train_x, data.train_y, p["batch_size"], p.get("shuffle_buffer", 1000))

    # -- one task -----------------------------------------------------------------------
    def _inner_step(self, params, grads):
        return [p - self.inner_lr * g if adapt and g is not None else tf.stop_gradient(p)
                for p, g, adapt in zip(params, grads, self.adapt_mask)]

    def _task_meta_gradient(self, model, x, y, xq, yq):
        """Meta-gradient and query loss of one task (support ``x, y``; query ``xq, yq``)."""
        theta = model.trainable_variables
        outer_weights = model.get_weights() if self.write_back else None

        with tf.GradientTape() as outer_tape:
            for step in range(self.inner_steps):
                with tf.GradientTape() as inner_tape:
                    support_loss, _ = compute_loss(model, x, y)
                grads = inner_tape.gradient(support_loss, theta)
                adapted = self._inner_step(theta, grads)
                if self.write_back and step < self.inner_steps - 1:
                    # v6: the next inner step starts from the adapted weights
                    for var, value in zip(theta, adapted):
                        var.assign(value)
            query_loss, _ = compute_loss(lambda inp: self.shadow(adapted, inp), xq, yq)

        if self.write_back:
            model.set_weights(outer_weights)
        return outer_tape.gradient(query_loss, theta), query_loss

    # -- outer update -------------------------------------------------------------------
    def _apply(self, grads, model):
        pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(pairs)

    @staticmethod
    def _accumulate(acc, grads):
        # a parameter whose meta-gradient is None (no path to the query loss) contributes zeros
        return [a + (g if g is not None else tf.zeros_like(a)) for a, g in zip(acc, grads)]

    @staticmethod
    def _global_norm(grads):
        return float(tf.norm(tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)))

    def _validation_loss(self, model, data):
        return float(tf.reduce_mean(my_loss(data.val_y, model(data.val_x))))

    def _log(self, epoch, **values):
        if self.verbose:
            print(f"[{self.name}] epoch {epoch + 1}: " + ", ".join(f"{k}={v:.3e}" for k, v in values.items()))

    # -- outer loop ---------------------------------------------------------------------
    def _query_batches(self, data):
        """Query set per task. Default: the full test split for every task."""
        xq, yq = tf.convert_to_tensor(data.test_x), tf.convert_to_tensor(data.test_y)
        while True:
            yield xq, yq

    def fit(self, model, data):
        p = self.params
        self._setup(model)
        tasks = self._task_dataset(data)
        history = {"query_loss": [], "meta_grad_norm": []}
        if self.validate:
            history["val_loss"] = []

        for epoch in range(p["epochs"]):
            total_loss = 0.
            meta_grads = [tf.zeros_like(v) for v in model.trainable_variables]
            queries = self._query_batches(data)

            for i, (x, y) in enumerate(tasks):
                xq, yq = next(queries)
                grads, query_loss = self._task_meta_gradient(model, x, y, xq, yq)
                if self.per_task_update:
                    self._apply(grads, model)
                meta_grads = self._accumulate(meta_grads, grads)
                total_loss += query_loss

            mean_loss = loss_norm(total_loss / (i + 1.))
            grad_norm = self._global_norm(meta_grads)
            if not self.per_task_update:
                self._apply(meta_grads, model)

            history["query_loss"].append(mean_loss)
            history["meta_grad_norm"].append(grad_norm)
            logs = {"query_loss": mean_loss, "meta_grad_norm": grad_norm}
            if self.validate:
                history["val_loss"].append(self._validation_loss(model, data))
                logs["val_loss"] = history["val_loss"][-1]
            self._log(epoch, **logs)

            if self._should_stop(history):
                if self.verbose:
                    print(f"[{self.name}] stopping criterion reached after epoch {epoch + 1}")
                break
        return history

    def _should_stop(self, history):
        return history["query_loss"][-1] < self.params["stop_loss"]


class MAML_v1_Basic(_MAMLBase):
    name = "maml_v1_basic"
    per_task_update = True


class MAML_v2_RLIF(MAML_v1_Basic):
    # Same algorithm as v1. With RSNN_vthr cells, adapt='rnn' also adapts the threshold.
    name = "maml_v2_rlif"


class MAML_v3_MetaBatch(_MAMLBase):
    name = "maml_v3_meta_batch"


class MAML_v4_MultiInner(MAML_v3_MetaBatch):
    # inner_steps > 1 repeats the same step from the unchanged meta-parameters (see module doc)
    name = "maml_v4_multi_inner"


class MAML_v5_TaskQuery(MAML_v4_MultiInner):
    name = "maml_v5_task_query"

    def _query_batches(self, data):
        p = self.params
        ds = make_dataset(data.test_x, data.test_y, p.get("query_batch_size", 1), p.get("shuffle_buffer", 1000))
        return iter(ds.repeat())


class MAML_v6_AdaptInnerOuter(MAML_v3_MetaBatch):
    name = "maml_v6_adapt_inner_outer"
    write_back = True
    validate = True

    def _should_stop(self, history):
        p = self.params
        return history["meta_grad_norm"][-1] < p["stop_grad_norm"] or history["val_loss"][-1] < p["stop_val_loss"]
