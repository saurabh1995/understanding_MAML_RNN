# understanding_MAML_RNN

Spiking recurrent networks (HSNN / RSNN / DSNN cells) learning stress-strain sequences,
trained either classically or by meta-learning (MAML).
`Understanding_MAML_RNN-Copy1.ipynb` has been restructured into the `maml_rnn` package
plus three scripts. The notebooks and the older `*.py` modules are left untouched.

```
configs/default.yaml        every setting, with notebook defaults
configs/*.yaml              experiment presets (merged on top of default.yaml)
train.py                    train with any strategy -> runs/<strategy>_<timestamp>/
finetune_compare.py         adapt pre-trained weights to single validation sequences, plot loss curves
tune.py                     keras-tuner Hyperband over network size
maml_rnn/
  data.py                   CSV loading, min-max scaling, train/test/validation split
  cells.py                  HSNN, RSNN, RSNN_vthr, DSNN
  surrogates.py             spike function + surrogate gradients
  slmu_cells.py             spiking LMU cells (reference only)
  models.py                 one config-driven builder (= model_klassiche)
  functional.py             ShadowModel: run the model with adapted parameter tensors
  strategies/classical.py   Keras fit
  strategies/maml_variants.py  the six MAML variants, with their evolution documented
```

## Usage

```bash
python train.py --config configs/classical.yaml
python train.py --config configs/maml.yaml                 # MAML as used in the notebook (v6)
python train.py --strategy maml_v3_meta_batch --set strategies.maml_v3_meta_batch.epochs=50
python train.py --strategy classical --set model.rnn.cell=LSTM --set model.rnn.units=128
python train.py --config configs/maml.yaml --set training.init_weights=runs/<run>/weights.pkl

python finetune_compare.py --config runs/<classical_run>/config.yaml \
    --weights "Klassisch=runs/<classical_run>/weights.pkl" \
    --weights "Meta=runs/<maml_run>/weights.pkl" --out plots/finetune
```

`--set key.sub=value` overrides any config entry (values are parsed as YAML, e.g.
`--set model.dense_units=[128,64]`). Every run stores its resolved `config.yaml`,
`weights.pkl` (pickled `get_weights()`), `history.json` and `loss.png`.

## Strategies

| name | notebook function | in short |
|---|---|---|
| `classical` | `model2_.fit` | supervised training, Adam, early stop at loss 1e-4 |
| `maml_v1_basic` | `MAML` | 1 inner step, RNN weights only, outer step after every task |
| `maml_v2_rlif` | `MAML_RLIF` | v1 for `RSNN_vthr` (threshold adapted too) |
| `maml_v3_meta_batch` | `MAML_LMU` | meta-gradients summed over tasks, one outer step per epoch, Dense adapted too |
| `maml_v4_multi_inner` | `MAML_LMU_multiple` | `inner_steps` added (every step restarts from θ: k steps = 1 step) |
| `maml_v5_task_query` | `MAML_LMU_multiple_` | per-task query batches from the test split |
| `maml_v6_adapt_inner_outer` (`maml`) | `MAML_LMU_multiple_adapt_inner_outer` | chained inner steps, validation per epoch, lr_inner 0.01 |

See the docstring of `maml_rnn/strategies/maml_variants.py` for details. To add a strategy, subclass
`Strategy`, register it in `maml_rnn/strategies/__init__.py` and add a `strategies.<name>`
section to `configs/default.yaml`.

Requirements: TensorFlow 2.x with Keras 2 (tested with TF 2.10), scikit-learn, PyYAML,
matplotlib; keras-tuner for `tune.py`.
