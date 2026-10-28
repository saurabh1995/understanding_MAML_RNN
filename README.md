# understanding_MAML_RNN

Recurrent spiking neural networks (SNNs) as surrogate models for path-dependent
stress-strain behaviour, trained either **classically** (supervised, Adam) or by
**meta-learning** (Model-Agnostic Meta-Learning, MAML). Meta-training is meant to give
pre-trained weights that adapt to a new loading sequence in a few gradient steps.

The code in `maml_rnn/` and the three scripts are a restructured version of the notebook
**`Understanding_MAML_RNN-Copy1.ipynb`**. The notebooks and older modules
(`decoding.py`, `Hybrid_Spiking_Neuron.py`, `recurrent_LIF.py`, ...) are unchanged.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Requirements](#2-requirements)
3. [Quick start](#3-quick-start)
4. [Data](#4-data)
5. [Model](#5-model)
6. [Training strategies](#6-training-strategies)
7. [Configuration reference](#7-configuration-reference)
8. [Scripts](#8-scripts)
9. [Run outputs](#9-run-outputs)
10. [Implementation notes](#10-implementation-notes)
11. [Extending the code](#11-extending-the-code)
12. [Notebook to module mapping](#12-notebook-to-module-mapping)


---

## 1. Repository layout

```
configs/
  default.yaml                 every setting, with the notebook's values
  classical.yaml               preset: classical training
  maml.yaml                    preset: MAML as used in the notebook (variant v6)
  maml_online_adaptation.yaml  preset: continue meta-training from saved weights, 5 inner steps
train.py                       train a model with any strategy
finetune_compare.py            fine-tune pre-trained weights on single validation sequences and compare
tune.py                        keras-tuner Hyperband search over the network size
maml_rnn/
  config.py                    YAML loading, merging, --set overrides
  data.py                      CSV loading, min-max scaling, train/test/validation split
  surrogates.py                spike function with selectable surrogate gradients
  cells.py                     spiking RNN cells: HSNN, RSNN, RSNN_vthr, DSNN
  slmu_cells.py                spiking Legendre Memory Unit cells (reference only)
  models.py                    config-driven model builder
  functional.py                ShadowModel: run the model with parameter tensors (for MAML)
  losses.py                    loss function and helpers
  callbacks.py                 early stopping on a loss threshold
  utils.py                     GPU setup, seeding, weight I/O, run directories, plots
  strategies/
    base.py                    Strategy interface
    classical.py               classical training
    maml_variants.py           the six MAML variants (v1-v6) and their history
```

## 2. Requirements

- Python 3.9 or 3.10
- TensorFlow 2.x with **Keras 2** (tested with TF 2.10). Keras 3 / TF ≥ 2.16 is not
  supported, because the MAML implementation sets Keras layer weight attributes to
  tensors (see [10.1](#101-shadowmodel-differentiating-through-the-inner-update)).
  The weight lookup also recognises Keras 3 variables, but the MAML code has not been
  tested on Keras 3.
- numpy, scikit-learn, PyYAML, matplotlib
- keras-tuner, only for `tune.py`. With TF < 2.16, use `pip install "keras-tuner==1.3.5"`
  (tested). From 1.4 on, keras-tuner depends on the standalone `keras` package and can
  upgrade it.

All commands below are run from the repository root, so `maml_rnn` can be imported.

## 3. Quick start

```bash
# classical pre-training
python train.py --config configs/classical.yaml

# meta-training (MAML variant v6, as in the notebook)
python train.py --config configs/maml.yaml

# compare how quickly both sets of weights adapt to unseen sequences
python finetune_compare.py --config runs/classical_<timestamp>/config.yaml \
    --weights "Klassische Trainiertes Model=runs/classical_<timestamp>/weights.pkl" \
    --weights "Meta Trainiertes Model=runs/maml_v6_adapt_inner_outer_<timestamp>/weights.pkl" \
    --out plots/finetune
```

Any setting can be changed from the command line:

```bash
python train.py --strategy maml_v3_meta_batch --set strategies.maml_v3_meta_batch.epochs=50
python train.py --strategy classical --set model.rnn.cell=LSTM --set model.rnn.units=128
python train.py --config configs/maml.yaml --set seed=0 --set model.dense_units=[128,64]
```

## 4. Data

**Source.** One CSV pair per loading sequence, read from `data.root`:

```
<input_prefix>_<id>.csv    e.g. strain_2111.csv   (network input)
<output_prefix>_<id>.csv   e.g. stress_2111.csv   (network target)
```

`<id>` joins one value from each list in `data.index_grid`. The default
`[[2, 3], [1, 2, 3], [1, 2, 3], [1, 5]]` gives the notebook's four nested loops, which is
2·3·3·2 = **36 sequences**. The files are whitespace-separated (`np.genfromtxt`) and
only the columns in `data.columns` (default `[0, 1, 3, 4, 5]`, so 5 components) are kept.
In the default data each sequence has 62 time steps.

**Scaling.** Inputs and outputs each get their own `MinMaxScaler` with range
`data.feature_range` (default `[-1, 1]`), fitted over all time steps of all sequences.
As in the notebook, the scalers are fitted **before** splitting, so they also see the
test and validation sequences.

**Splits.** Two consecutive `train_test_split` calls, as in the notebook:

| step | split | fraction / seed | default size |
|---|---|---|---|
| 1 | all → rest + **validation** | `validation_size=0.35`, `validation_seed=38` | 13 sequences |
| 2 | rest → **train** + **test** | `test_size=0.3`, `test_seed=36` | 16 train / 7 test |

How the splits are used:

| split | classical | MAML |
|---|---|---|
| train | training data | tasks (support sets), one sequence per task |
| test | not used | query set |
| validation | monitored only (`val_loss`) | monitored (v6); fine-tuning targets in `finetune_compare.py` |

Arrays are `float32` with shape `(n_sequences, n_steps, n_features)`.
`prepare_data()` returns a `SequenceData` object that also holds the fitted scalers
(`x_scaler`, `y_scaler`) for converting predictions back to physical units.

## 5. Model

### 5.1 Architecture

A single builder (`maml_rnn.models.build_model`) is used for every strategy, so weights
can always be moved between runs. The default is the notebook's `model_klassiche`:

```
Input (None, 5)
 → RNN(HSNN(208, thr=1e-7))      ×3   (model.rnn)
 → [RNN(decoder cell)]                (model.decoder, off by default)
 → tanh                               (model.post_rnn_activation)
 → Dense(240, tanh) → Dense(64, tanh) (model.dense_units / dense_activation)
 → Dense(5, tanh)                     (model.output_dim / output_activation)
```

The order of `model.layers` matters for `finetune.freeze_layers`. With the default
config it is:

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| layer | Input | RNN | RNN | RNN | tanh | Dense 240 | Dense 64 | Dense 5 |

The notebook's freezing cell corresponds to `freeze_layers: [1, 2, 3, 4, 5]`, which
trains only the last two Dense layers. A decoder adds one RNN layer after index 3 and
shifts the later indices by one.

### 5.2 Spiking cells (`maml_rnn/cells.py`)

All spiking cells share leaky-integrate dynamics with decay `d = exp(-1/tau)`:

```
i_in = x_t · W_in + r_{t-1} · W_rec          (W_rec has a zero diagonal: no self-connections)
v_t  = d · v_{t-1} + (1 - d) · i_in + reset
z_t  = H((v_t - thr) / thr)                  (Heaviside spike, surrogate gradient)
```

They differ in the recurrent signal `r`, the reset and the output:

| cell | recurrent signal r | reset | output | trainable params | default surrogate | from |
|---|---|---|---|---|---|---|
| `HSNN` | spikes z | −thr·z | z·v | W_in, W_rec | `sigmoid` | `Hybrid_Spiking_Neuron.py` |
| `RSNN` | spikes z | −thr·z | z | W_in, W_rec | `atan_normalized` | `recurrent_LIF.py` |
| `RSNN_vthr` | spikes z | −thr·z | tanh(z·v) | W_in, W_rec, thr | `fast_sigmoid` | notebook cell 6 |
| `DSNN` | voltage v | −thr (constant) | v | W_in, W_rec | `fast_sigmoid` | `decoding.py` |

Weights are initialised as `N(0, 1/n_in)` (input) and `N(0, 1/n)` (recurrent).
The cell arguments `thr`, `tau` (default 20), `dampening_factor` (default 0.3) and
`surrogate` are set via `model.rnn.params`. `RSNN_vthr` ignores `thr`: its scalar threshold is
trainable and initialised uniformly in [−10, 10], as in the notebook.

Keras cells are also supported: `LSTM`, `GRU` and `SimpleRNN`. For these, the spiking-only
arguments (`thr`, `tau`, `dampening_factor`, `surrogate`) are dropped automatically.

### 5.3 Surrogate gradients (`maml_rnn/surrogates.py`)

The forward pass is always the Heaviside step. The backward pass uses one of these
pseudo-derivatives, where `x = (v - thr)/thr` and `d` is the dampening factor:

| name | pseudo-derivative |
|---|---|
| `fast_sigmoid` | d / (1 + \|x\|)² |
| `sigmoid` | d · σ(x) · (1 − σ(x)) |
| `triangle_atan` | atan(max(0, 1 − \|x\|) · d) |
| `atan_normalized` | atan(x · d), then rescaled to unit L2 norm (an all-zero gradient stays zero) |

Each cell keeps the surrogate of the module it came from as its default, so results
match the original code. You can change it with `--set model.rnn.params.surrogate=fast_sigmoid`.

### 5.4 SLMU cells (`maml_rnn/slmu_cells.py`)

The four spiking Legendre Memory Unit cells from notebook cell 5 (`SLMUcell`,
`SLMUcell_thr`, `SLMUcell_thr_V`, `SLMUcell_LIF_thr`) are now one base class plus three
subclasses. The variants differ only in whether the threshold is trainable and in what the cell outputs. The notebook only used them
in commented-out code, so they are not selectable from the config. To use them
directly:

```python
cell  = SLMUcell_LIF_thr(memory_d=10, order=10, theta=784,
                         hidden_cell=tf.keras.layers.SimpleRNNCell(units=208))
layer = tf.keras.layers.RNN(cell, return_sequences=True)
```

The hidden cell can also be an `LSTMCell`. Its extra (carry) state is passed through
unchanged.

### 5.5 Loss

`my_loss(y_true, y_pred) = mean over batch and time of (y_true − y_pred)²`, kept **per
output feature** (shape `(5,)`). Gradients of this vector loss equal those of its sum.
Keras reports its mean, while the MAML strategies log its L2 norm, as the notebook did.
The reported "Final MSE" values and the MAML `val_loss` are the scalar mean.

## 6. Training strategies

| `strategy` | notebook function | summary |
|---|---|---|
| `classical` | `model2_.fit` | supervised training on the train split |
| `maml_v1_basic` | `MAML` | 1 inner step, RNN weights adapted, outer step after every task |
| `maml_v2_rlif` | `MAML_RLIF` | v1 for `RSNN_vthr` cells (threshold adapted as well) |
| `maml_v3_meta_batch` | `MAML_LMU` | meta-gradients summed over all tasks, one outer step per epoch, Dense adapted too |
| `maml_v4_multi_inner` | `MAML_LMU_multiple` | adds `inner_steps`, adapts LSTM + Dense (every step restarts from θ) |
| `maml_v5_task_query` | `MAML_LMU_multiple_` | separate query batch per task from the test split |
| `maml_v6_adapt_inner_outer` | `MAML_LMU_multiple_adapt_inner_outer` | chained inner steps, validation each epoch; used for the results |
| `maml` | — | alias for `maml_v6_adapt_inner_outer` |

### 6.1 Classical

Adam with learning rate `lr`, batch size `batch_size`, for up to `epochs` epochs. Training
stops as soon as the training loss is ≤ `stop_loss` (the notebook's
`EarlyStoppingByAccuracy(monitor='loss', value=1e-4)`). If `monitor_validation` is true,
the validation split is passed to `fit` for logging only. It does not affect training.

### 6.2 MAML: common algorithm

With meta-parameters θ, task support set (x, y) = one training sequence, and query set
(x_q, y_q):

```
for epoch in 1..epochs:
    for each task (x, y):
        θ'  = θ − inner_lr · ∇θ L(θ; x, y)                  # inner step(s)
        g   = ∇θ L(θ'; x_q, y_q)                            # meta-gradient, 2nd order
        v1/v2:  Adam step with g                            # per-task update
        v3-v6:  G += g
    v3-v6:  Adam step with G                                # one meta-update per epoch
    check stopping criterion
```

- The meta-gradient is **second order**: it differentiates through the inner
  gradient computation.
- `adapt` selects which parameters the inner step changes: `rnn` (all RNN layers),
  `all` (RNN + Dense) or `lstm_dense` (RNN layers with an LSTM cell + Dense).
- Parameters that are not adapted keep their meta-value θ in the query pass, and the
  direct query-loss path to them is cut (`tf.stop_gradient`). They still get a small
  second-order meta-gradient, because the inner gradient of the adapted parameters
  depends on them. This matches the notebook.
- Tasks are shuffled every epoch (`shuffle_buffer`, default 1000; set to 0 to disable).

### 6.3 Variant details and history

The variants are kept in the order they were developed, so you can follow and repeat
each step. The module docstring of `maml_rnn/strategies/maml_variants.py` gives the full
account.

| | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|---|
| outer update | per task | per task | per epoch | per epoch | per epoch | per epoch |
| default `adapt` | `rnn` | `rnn` | `all` | `lstm_dense` | `all` | `all` |
| inner_lr | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 | 0.01 |
| multi-step inner loop | – | – | – | repeats step 1 | repeats step 1 | chained |
| query set | test split | test split | test split | test split | 1 test seq. per task | test split |
| validation per epoch | – | – | – | – | – | yes |
| stop when | ‖loss‖ < 1e-4 | ‖loss‖ < 1e-4 | ‖loss‖ < 1e-3 | ‖loss‖ < 2e-4 | ‖loss‖ < 2e-4 | ‖G‖ < 1e-4 or val < 1e-4 |

Here ‖loss‖ is the norm of the epoch-mean query loss and ‖G‖ is the global norm of the
summed meta-gradient. The outer learning rate is 0.001 (Adam) for all variants.

- **v1 → v2:** same algorithm, rewritten for `RSNN_vthr`. With `adapt: rnn` the trainable
  threshold is adapted as well.
- **v2 → v3:** proper batched MAML outer loop (sum over tasks, one step per epoch), and
  the Dense layers are adapted too.
- **v3 → v4:** introduces `inner_steps`. Each inner iteration recomputes the gradient
  from the **unchanged** θ, so k steps give exactly the result of one step. This is
  reproduced on purpose. With spiking cells, `lstm_dense` adapts only the Dense layers.
  The same holds for v1–v3. For v1–v5, a `RuntimeWarning` is raised whenever
  `inner_steps > 1`.
- **v4 → v5:** adaptation goes back to spiking RNN + Dense layers, and each task gets its own
  query batch (`query_batch_size`, default 1) cycled from the test split. In the
  notebook this query batch (`xx, yy`) was never defined, so v5 could not run. The
  version here completes it.
- **v5 → v6:** after each inner step the adapted weights are written back into the model,
  so the next step starts from them. θ is restored before the meta-gradient is taken.
  The validation loss is computed after every outer step, and training stops on a small
  meta-gradient or validation loss. Caveat: writing back into variables cuts the
  second-order path through earlier inner steps. With `inner_steps: 1` (the notebook's
  training setting) this is exact second-order MAML. With more steps, only the last
  step is differentiated through.

## 7. Configuration reference

### 7.1 How configs are combined

1. `configs/default.yaml` is always loaded.
2. The file given with `--config` is merged on top of it, key by key, recursively.
3. `--strategy NAME` sets `strategy` and is applied first among the overrides.
4. `--set key.sub=value` overrides are applied in the order given. Values are parsed as
   YAML (`--set model.dense_units=[128,64]`, `--set model.decoder=null`,
   `--set "model.decoder={cell: DSNN, units: 208, params: {thr: 1.0e-7}}"`).
   A dotted key also works below a section that is `null` by default. For example,
   `--set model.decoder.cell=DSNN --set model.decoder.units=208` creates the section.
5. With `--strategy maml`, settings given under the alias (`--set strategies.maml.epochs=5`)
   are merged into `strategies.maml_v6_adapt_inner_outer`.

The resolved configuration is saved with every run, and a run's `config.yaml` can be
passed back with `--config`.

### 7.2 Keys

**Top level**

| key | default | meaning |
|---|---|---|
| `strategy` | `classical` | strategy name (see §6) |
| `seed` | `null` | seeds Python, NumPy and TF; `null` = unseeded, as in the notebook |
| `output_dir` | `runs` | parent directory of run directories |

**`data`** — see §4

| key | default |
|---|---|
| `root` | `C:\Users\Tandale\sciebo\olddata\Phd\Writing\SLMU_paper\BVPs\Stress-Strain_Data_csv\BVP_1_GeomNL` |
| `input_prefix` / `output_prefix` | `strain` / `stress` |
| `index_grid` | `[[2, 3], [1, 2, 3], [1, 2, 3], [1, 5]]` |
| `columns` | `[0, 1, 3, 4, 5]` |
| `feature_range` | `[-1, 1]` |
| `validation_size` / `validation_seed` | `0.35` / `38` |
| `test_size` / `test_seed` | `0.3` / `36` |

**`model`** — see §5

| key | default | meaning |
|---|---|---|
| `input_dim` / `output_dim` | `5` / `5` | feature dimensions |
| `rnn.cell` | `HSNN` | `HSNN`, `RSNN`, `RSNN_vthr`, `DSNN`, `LSTM`, `GRU`, `SimpleRNN` |
| `rnn.units` | `208` | neurons per RNN layer |
| `rnn.num_layers` | `3` | number of stacked RNN layers |
| `rnn.params` | `{thr: 1.0e-7}` | keyword arguments for the cell |
| `decoder` | `null` | optional extra RNN layer, `{cell, units, params}` |
| `post_rnn_activation` | `tanh` | activation after the RNN stack (`null` = none) |
| `dense_units` | `[240, 64]` | hidden Dense layers |
| `dense_activation` | `tanh` | their activation |
| `output_activation` | `tanh` | activation of the output layer |
| `unroll` | `false` | unroll the RNNs (fixes the input length to the data's sequence length) |

**`training`**

| key | default | meaning |
|---|---|---|
| `init_weights` | `null` | path to a `weights.pkl` to start from, for any strategy |

**`strategies.<name>`** — one section per strategy, with the values from the table in §6.3

| key | used by | meaning |
|---|---|---|
| `epochs` | all | maximum number of epochs (outer iterations for MAML) |
| `batch_size` | all | classical: mini-batch size; MAML: sequences per task (1) |
| `lr` | classical | Adam learning rate |
| `stop_loss` | classical, v1-v5 | loss threshold for stopping |
| `monitor_validation` | classical | log validation loss |
| `inner_lr` / `outer_lr` | MAML | inner SGD step size / Adam learning rate |
| `inner_steps` | MAML | number of inner steps |
| `adapt` | MAML | `rnn`, `all` or `lstm_dense` |
| `query_batch_size` | v5 | test sequences per task query |
| `shuffle_buffer` | MAML | task shuffling buffer, 0 = keep order |
| `stop_grad_norm` / `stop_val_loss` | v6 | stopping thresholds |

**`finetune`** — `finetune_compare.py`

| key | default | meaning |
|---|---|---|
| `num_sequences` | `5` | number of random validation sequences (if no indices are given) |
| `sequence_indices` | `null` | explicit validation indices, e.g. `[0, 2]` |
| `epochs` / `batch_size` / `lr` | `100` / `64` / `0.001` | fine-tuning settings (Adam) |
| `stop_loss` | `null` | optional early-stopping threshold, e.g. `5.0e-5` |
| `freeze_layers` | `[]` | `model.layers` indices to freeze (see §5.1) |
| `formats` | `[pdf, svg]` | plot file formats |

**`tuning`** — `tune.py`

| key | default | meaning |
|---|---|---|
| `directory` / `project_name` | `my_dir` / `hyperband_rnn` | keras-tuner storage |
| `max_epochs` / `factor` / `hyperband_iterations` | `1000` / `3` / `5` | Hyperband settings |
| `batch_size` | `32` | training batch size |
| `rnn_layers` | `[1, 12]` | min/max number of RNN layers |
| `rnn_units` / `dense_units` | `[16, 256, 32]` | min, max, step |
| `dense_layers` | `[1, 4]` | min/max number of Dense layers |

## 8. Scripts

### `train.py`

```
python train.py [--config FILE] [--strategy NAME] [--set KEY=VALUE ...] [--run-name NAME] [--verbose 0|1]
```

The script loads the data, builds the model, optionally loads `training.init_weights`,
and runs the strategy. It then reports the final MSE on the train, test and validation
splits and writes a run directory (§9).

### `finetune_compare.py`

```
python finetune_compare.py --weights LABEL=PATH [--weights LABEL=PATH ...]
                           [--config FILE] [--out DIR] [--set KEY=VALUE ...]
```

For each selected validation sequence and each set of weights:
1. build a fresh model and load the weights;
2. freeze `finetune.freeze_layers`;
3. fine-tune on that single sequence with `model.fit`;
4. plot the loss curves of all weight sets together.

Outputs go to `--out`: `Meta_Klassich_seq_<i>.<fmt>` per sequence, and
`finetune_curves.json` with all curves. The architecture comes from the config, so pass
the `config.yaml` of the run that produced the weights.

### `tune.py`

```
python tune.py [--config FILE] [--set KEY=VALUE ...]
```

Runs keras-tuner Hyperband on classical training, minimising the training loss. It
searches the number and width of the RNN layers (cell type from `model.rnn.cell`) and of
the Dense layers, then prints the best hyperparameters. `model.unroll: true` is
supported.

On Windows, keep `tuning.directory` short (the default `my_dir` is fine). keras-tuner
writes deeply nested checkpoint files. If the full path exceeds the 260-character
limit, every trial fails with "The system cannot find the path specified".

## 9. Run outputs

Each `train.py` run creates `<output_dir>/<run-name or strategy>_<YYYYmmdd_HHMMSS>/`:

| file | content |
|---|---|
| `config.yaml` | fully resolved configuration |
| `weights.pkl` | `model.get_weights()`, pickled (same format as the notebook) |
| `history.json` | per-epoch curves, plus `final_train` / `final_test` / `final_validation` MSE |
| `loss.png` | log-scale plot of the per-epoch curves |

History keys: classical gives `loss` and `val_loss`. MAML gives `query_loss` and
`meta_grad_norm`, plus `val_loss` for v6.

To load weights in your own code:

```python
from maml_rnn.config import load_config
from maml_rnn.models import build_model
from maml_rnn.utils import load_weights

cfg = load_config("runs/<run>/config.yaml")
model = build_model(cfg["model"])
model.set_weights(load_weights("runs/<run>/weights.pkl"))
```

## 10. Implementation notes

### 10.1 ShadowModel: differentiating through the inner update

MAML needs the query loss at θ' = θ − α∇L while keeping θ' a differentiable function
of θ. Assigning θ' to variables would cut that dependency. The notebook built a fresh
copy of the model for every task and overwrote each layer's weight attribute
(`cell.input_weights`, `dense.kernel`, ...) with a tensor, using hand-written index
bookkeeping per layer type.

`maml_rnn.functional.ShadowModel` does the same thing generically:

- It builds the copy once.
- It finds every trainable variable stored as an attribute of a layer or of an RNN layer's
  cell, and maps it to its index in `model.trainable_variables`.
- `shadow(params, x)` writes `params[i]` into those attributes and runs the model.

Cells must therefore read their weights through these attributes in `call`, which all
cells in this package do. Once an attribute has been replaced by a tensor, Keras stops
tracking the variable. The shadow's own `trainable_variables` and `set_weights` must
not be used after that, which is why the index map is computed beforehand.

### 10.2 Differences from the notebook

| topic | notebook | here |
|---|---|---|
| model definitions | `TrainModel` (MAML) and `model_klassiche` (classical), kept in sync by hand | one builder for everything |
| weight substitution | new model copy per task, layer-specific index code | `ShadowModel`, built once |
| optimizer / data | globals | owned by the strategy / passed explicitly |
| v5 query batch | undefined `xx, yy` | cycles through the test split |
| non-adapted parameters in v1/v2 | crash at Dense layers (no `.cell`) | kept at θ, as intended |
| `EarlyStoppingByAccuracy` | used `warnings` without importing it | fixed (`EarlyStoppingByValue`, old name kept as alias) |
| SLMU cells | four ~250-line copies; extra hidden-cell states dropped | one base class + 3 subclasses; hidden states passed through |
| `atan_normalized` surrogate (`recurrent_LIF.py`) | NaN when the gradient is all zeros | stays zero (`divide_no_nan`) |
| post-RNN tanh | `tf.math.tanh` op | `Activation('tanh')` layer, same position in `model.layers` |

## 11. Extending the code

**New strategy**

1. Subclass `maml_rnn.strategies.base.Strategy` and implement
   `fit(self, model, data) -> dict`. `self.params` is the strategy's config section,
   `self.build_fn()` builds a fresh model, and `data` is a `SequenceData`.
2. Register the class in `STRATEGIES` in `maml_rnn/strategies/__init__.py`.
3. Add a `strategies.<name>` section to `configs/default.yaml`.

A new MAML-type method can subclass `_MAMLBase` in `maml_variants.py`. It then inherits
task iteration, inner steps, meta-gradient accumulation and logging, and only needs to
override the class switches (`per_task_update`, `write_back`, `validate`) or methods
(`_query_batches`, `_should_stop`, ...).

**New cell**

Subclass `maml_rnn.cells.SpikingCell` (or any Keras layer with `state_size`) and
implement `call(inputs, states)`. Read all trainable weights from attributes set in
`build` so `ShadowModel` can substitute them. Then add the class to `CELLS` in `cells.py`.

**New surrogate gradient**

Add a pseudo-derivative function and an entry in `SPIKE_FUNCTIONS` in `surrogates.py`.

**Different dataset**

Adjust the `data` section. If the files have another layout, replace `load_sequences`
in `data.py`; everything downstream only needs `(n_seq, n_steps, n_features)` arrays.

## 12. Notebook to module mapping

`Understanding_MAML_RNN-Copy1.ipynb`, by code-cell index:

| cells | content | now in |
|---|---|---|
| 0–1 | imports, GPU memory growth | `utils.setup_gpu` |
| 2–4 | data loading, scaling, splitting | `data.py` |
| 5 | SLMU cells, DSNN | `slmu_cells.py`, `cells.DSNN` |
| 6 | `RSNN_vthr` | `cells.RSNN_vthr` |
| 7 | `my_loss`, `compute_loss` | `losses.py` |
| 9 | `TrainModel`, `copy_model` | `models.py`, `functional.py` |
| 11–14 | Hyperband tuning | `tune.py` |
| 15–16 | `model_klassiche` | `models.py` (default config) |
| 17–18 | early stopping | `callbacks.py` |
| 21–22 | classical training and evaluation | `strategies/classical.py`, `train.py` |
| 26 | MAML datasets and optimizer | `strategies/maml_variants.py` |
| 28 | six MAML functions | `strategies/maml_variants.py` |
| 31 | meta-training call | `configs/maml.yaml` |
| 35 | further meta-training from meta weights | `training.init_weights`, `configs/maml_online_adaptation.yaml` |
| 37–42, 49–59 | fine-tuning comparison, layer freezing, plots | `finetune_compare.py` |
| 45–48 | weight pickling | `utils.save_weights` / `load_weights` |

Not ported: the prediction-vs-data plot (cell 44), the output un-scaling scratch
(cells 60–62) and the unused multi-output model `model2` (cell 24). Use
`data.y_scaler.inverse_transform` to un-scale predictions.

