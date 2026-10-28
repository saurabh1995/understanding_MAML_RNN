"""Spiking-RNN surrogate models for stress-strain sequences, trained classically or with MAML.

Package layout
--------------
config      YAML loading / merging / CLI overrides
data        loading of the BVP stress-strain CSVs, scaling and splitting
surrogates  spike function with selectable surrogate (pseudo-)derivatives
cells       spiking RNN cells (HSNN, DSNN, RSNN, RSNN_vthr)
slmu_cells  spiking Legendre-Memory-Unit cells (kept for reference, not wired into the builder)
models      config-driven model builder (one architecture for classical and meta training)
functional  run a Keras model with arbitrary parameter tensors (needed for MAML)
losses      loss functions
callbacks   Keras callbacks
strategies  training strategies (classical + the six MAML variants of the notebook)
utils       GPU setup, seeding, weight I/O, run directories, plotting
"""
