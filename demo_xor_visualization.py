import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from neural_network import MLP


RANDOM_SEED = 0
EPOCHS = 350
LEARNING_RATE = 0.05
SNAPSHOT_INTERVAL = 25
GRID_RESOLUTION = 45
ANIMATION_INTERVAL_MS = 220

XOR_INPUTS = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
XOR_TARGETS = [0.0, 1.0, 1.0, 0.0]


def build_model():
    random.seed(RANDOM_SEED)
    return MLP(2, [4, 4, 1])


def predict_scalar(model, inputs):
    return model(inputs).data


def make_grid(resolution):
    axis = np.linspace(-0.25, 1.25, resolution)
    return np.meshgrid(axis, axis)


def evaluate_grid(model, grid_x, grid_y):
    boundary = np.zeros_like(grid_x)
    for row in range(grid_x.shape[0]):
        for col in range(grid_x.shape[1]):
            score = predict_scalar(model, [float(grid_x[row, col]), float(grid_y[row, col])])
            boundary[row, col] = score
    return boundary


def collect_snapshot(model, epoch, loss_value, grid_x, grid_y):
    predictions = [predict_scalar(model, sample) for sample in XOR_INPUTS]
    boundary = evaluate_grid(model, grid_x, grid_y)
    return {
        "epoch": epoch,
        "loss": loss_value,
        "predictions": predictions,
        "boundary": boundary,
    }


def train_with_snapshots():
    model = build_model()
    grid_x, grid_y = make_grid(GRID_RESOLUTION)

    snapshots = []
    loss_history = []

    for epoch in range(EPOCHS + 1):
        predictions = [model(sample) for sample in XOR_INPUTS]
        loss = sum((prediction - target) ** 2 for prediction, target in zip(predictions, XOR_TARGETS))
        loss_history.append(loss.data)

        if epoch == 0 or epoch % SNAPSHOT_INTERVAL == 0 or epoch == EPOCHS:
            snapshots.append(collect_snapshot(model, epoch, loss.data, grid_x, grid_y))

        if epoch == EPOCHS:
            break

        model.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            parameter.data -= LEARNING_RATE * parameter.grad

    return model, snapshots, loss_history, grid_x, grid_y


def build_animation(snapshots, loss_history, grid_x, grid_y):
    figure, (boundary_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5))

    first_frame = snapshots[0]
    boundary_image = boundary_ax.imshow(
        first_frame["boundary"],
        origin="lower",
        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        cmap="RdYlBu",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    figure.colorbar(boundary_image, ax=boundary_ax, fraction=0.046, pad=0.04, label="Network output")

    sample_points = np.array(XOR_INPUTS)
    boundary_ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        c=XOR_TARGETS,
        cmap="bwr",
        edgecolors="black",
        s=140,
        zorder=3,
    )
    for index, (x_value, y_value) in enumerate(XOR_INPUTS):
        boundary_ax.text(
            x_value + 0.04,
            y_value + 0.04,
            f"target={int(XOR_TARGETS[index])}",
            fontsize=9,
            weight="bold",
        )

    boundary_ax.set_title("XOR Decision Boundary")
    boundary_ax.set_xlabel("x1")
    boundary_ax.set_ylabel("x2")
    boundary_ax.set_xlim(grid_x.min(), grid_x.max())
    boundary_ax.set_ylim(grid_y.min(), grid_y.max())

    info_text = boundary_ax.text(
        0.02,
        1.02,
        "",
        transform=boundary_ax.transAxes,
        fontsize=10,
        weight="bold",
    )

    loss_ax.set_title("Training Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Squared error")
    loss_ax.set_xlim(0, EPOCHS)
    loss_ax.set_ylim(0, max(loss_history) * 1.05)
    loss_ax.grid(alpha=0.3)

    loss_line, = loss_ax.plot([], [], color="tab:blue", linewidth=2)
    current_point, = loss_ax.plot([], [], "o", color="tab:red")

    def update(frame_index):
        snapshot = snapshots[frame_index]
        boundary_image.set_data(snapshot["boundary"])

        visible_epochs = list(range(0, snapshot["epoch"] + 1))
        visible_loss = loss_history[: snapshot["epoch"] + 1]
        loss_line.set_data(visible_epochs, visible_loss)
        current_point.set_data([snapshot["epoch"]], [snapshot["loss"]])

        prediction_text = ", ".join(f"{value:.2f}" for value in snapshot["predictions"])
        info_text.set_text(
            f"Epoch {snapshot['epoch']}/{EPOCHS}   Loss {snapshot['loss']:.6f}\n"
            f"Predictions: [{prediction_text}]"
        )

        return boundary_image, loss_line, current_point, info_text

    animation = FuncAnimation(
        figure,
        update,
        frames=len(snapshots),
        interval=ANIMATION_INTERVAL_MS,
        repeat=False,
        blit=False,
    )

    return figure, animation


def main():
    model, snapshots, loss_history, grid_x, grid_y = train_with_snapshots()

    final_predictions = [predict_scalar(model, sample) for sample in XOR_INPUTS]
    print("Final XOR predictions:")
    for sample, prediction, target in zip(XOR_INPUTS, final_predictions, XOR_TARGETS):
        print(f"  input={sample} predicted={prediction:.4f} target={target:.1f}")
    print(f"Final loss: {loss_history[-1]:.8f}")

    backend_name = plt.get_backend().lower()
    if backend_name == "agg":
        print(f"Matplotlib backend '{plt.get_backend()}' is non-interactive; skipping live window.")
        return None

    figure, animation = build_animation(snapshots, loss_history, grid_x, grid_y)
    figure.suptitle("Neural Network Learning XOR", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

    return animation


if __name__ == "__main__":
    _ANIMATION = main()
