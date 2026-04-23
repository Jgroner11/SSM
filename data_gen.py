import numpy as np
import plotly.graph_objects as go
from numpy import pi


SEED = 31
SAMPLES_PER_CLASS = 4
N_TIME_STEPS = 17
T_RANGE = (-10, 10)

# Sinusoid parameters
A_RANGE = (0, 3)
LOG_P_RANGE = (-.5, 2)
C_RANGE = (-10, 10)


SLOPE_RANGE = (-.5, .5)
INTERCEPT_RANGE = (-7.5, 7.5)

LABEL_LINE = 0
LABEL_SINUSOID = 1

def generate_sinusoids(n_samples, t_values, rng):
    a = rng.uniform(*A_RANGE, size=n_samples)
    log_p = rng.uniform(*LOG_P_RANGE, size=n_samples)
    p = 10 ** log_p
    phi = rng.uniform(0, p)
    c = rng.uniform(*C_RANGE, size=n_samples)

    y_values = (
        a[:, None]
        * np.sin(2 * pi * (t_values[None, :] / p[:, None] + phi[:, None]))
        + c[:, None]
    )

    return y_values


def generate_lines(n_samples, t_values, rng):
    slope = rng.uniform(*SLOPE_RANGE, size=n_samples)
    intercept = rng.uniform(*INTERCEPT_RANGE, size=n_samples)
    y_values = slope[:, None] * t_values[None, :] + intercept[:, None]

    return y_values


def plot_curves(t_values, y_values, title, trace_prefix):
    fig = go.Figure()
    for idx, curve in enumerate(y_values, start=1):
        fig.add_trace(
            go.Scatter(x=t_values, y=curve, mode="lines", name=f"{trace_prefix} {idx}")
        )

    fig.update_layout(
        title=title,
        xaxis_title="t",
        yaxis_title="y",
    )
    return fig


def stack_curves_as_single_trace(t_values, y_values):
    x_segments = []
    y_segments = []

    for curve in y_values:
        x_segments.extend(t_values)
        x_segments.append(None)
        y_segments.extend(curve)
        y_segments.append(None)

    return x_segments, y_segments


def build_labeled_dataset(
    n_samples_per_class=SAMPLES_PER_CLASS,
    n_data_points=N_TIME_STEPS,
    t_range=T_RANGE,
    seed=SEED,
    shuffle=True,
    save_path=None,
):
    rng = np.random.default_rng(seed)
    t_values = np.linspace(*t_range, n_data_points)

    sinusoid_x = generate_sinusoids(n_samples_per_class, t_values, rng)
    line_x = generate_lines(n_samples_per_class, t_values, rng)

    features = np.vstack([sinusoid_x, line_x])
    labels = np.concatenate(
        [
            np.full(n_samples_per_class, LABEL_SINUSOID, dtype=int),
            np.full(n_samples_per_class, LABEL_LINE, dtype=int),
        ]
    )

    if shuffle:
        indices = rng.permutation(features.shape[0])
        features = features[indices]
        labels = labels[indices]

    data = {
        "t": t_values,
        "X": features,
        "y": labels,
    }

    if save_path is not None:
        np.savez(
            save_path,
            t=data["t"],
            X=data["X"],
            y=data["y"],
        )

    return data


def build_demo_figures(
    n_samples=SAMPLES_PER_CLASS,
    n_data_points=N_TIME_STEPS,
    t_range=T_RANGE,
    seed=SEED,
):
    rng = np.random.default_rng(seed)
    t_values = np.linspace(*t_range, n_data_points)

    sinusoid_y = generate_sinusoids(n_samples, t_values, rng)
    line_y = generate_lines(n_samples, t_values, rng)

    sinusoid_x, sinusoid_trace_y = stack_curves_as_single_trace(t_values, sinusoid_y)
    line_x, line_trace_y = stack_curves_as_single_trace(t_values, line_y)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sinusoid_x,
            y=sinusoid_trace_y,
            mode="lines",
            name="sinusoids",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_trace_y,
            mode="lines",
            name="lines",
        )
    )
    fig.update_layout(
        title="Randomly Generated Curves",
        xaxis_title="t",
        yaxis_title="y",
    )
    return fig


if __name__ == "__main__":
    data = build_labeled_dataset(save_path="data/labeled_dataset.npz")
    demo_fig = build_demo_figures()
    demo_fig.show()
