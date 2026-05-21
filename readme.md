# SSM Playground

A small PyTorch playground for learning how state-space models behave on a simple
sequence classification task. The project generates synthetic curves, trains a
binary classifier to distinguish sinusoids from lines, and compares several
hand-written SSM-style architectures against a CPU-friendly Mamba implementation.

## What It Does

- Generates a labeled dataset of one-dimensional curves saved as
  `data/sins_vs_lines.npz`.
- Trains SSM classifiers with mini-batch Adam optimization.
- Tracks train/test loss and accuracy during training.
- Plots generated data, single-model accuracy, and multi-model loss comparisons
  with Plotly.

## Dependencies

Standard Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Conda environment:

```bash
conda env create -f environment.yml
conda activate ssm
```

Core runtime dependencies are Python, NumPy, PyTorch, Plotly, and the packages
needed by the vendored Mamba implementation.

## Code Files

- `data_gen.py` - Builds `data/sins_vs_lines.npz` from random sinusoids and
  lines.
- `engine.py` - Contains the single-model training loop, using `m8` by default.
- `model_comparison.py` - Runs the registered models and plots their loss curves.
- `models.py` - Contains the model implementations, from simple scalar SSMs to a
  Mamba-backed classifier.

## Notes

- Labels are binary: `0` for lines and `1` for sinusoids.
- Most custom SSM models support both recurrent and convolution modes, but their
  default `forward` path uses the convolution-style implementation where
  available.
- `ssm_infographic.png` is useful for understanding the model variants.
- The Mamba SSM implementation is taken from the CPU fork at
  https://github.com/kroggen/mamba-cpu.
