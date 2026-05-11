import plotly.graph_objects as go

from engine import train
from models import m1, m2, m3, m4, m5, m6, m7, m8


MODELS = {
    "m1": m1,
    "m2": m2,
    "m3": lambda: m3(hidden_size=10),
    "m4": m4,
    "m5": lambda: m5(hidden_size=10),
    "m6": m6,
    "m7": lambda: m7(hidden_size=10),
    "m8": lambda: m8(hidden_size=10),
}


def compare_models(n_iters=400, batch_size=10):
    fig = go.Figure()

    for model_name, model_factory in MODELS.items():
        print(f"Training {model_name}")
        history = train(
            n_iters=n_iters,
            batch_size=batch_size,
            model_factory=model_factory,
            plot=False,
        )
        iterations = list(range(1, n_iters + 1))
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=history["train_losses"],
                mode="lines",
                name=model_name,
            )
        )

    fig.update_layout(
        title="Model Loss Comparison",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        hovermode="closest",
    )
    fig.show()


if __name__ == "__main__":
    compare_models()
