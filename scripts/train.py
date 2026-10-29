"""Train the symbol-width regressor end to end."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sw_dlnet.config import load_config  # noqa: E402
from sw_dlnet.data import DatasetManager, set_seed  # noqa: E402
from sw_dlnet.engine import Tester, Trainer  # noqa: E402
from sw_dlnet.models import HybridModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides of the YAML config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config")
    parser.add_argument("--data-root", type=str, default=None, help="Folder of raw CSV files")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run the test split")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint used with --test-only")
    return parser.parse_args()


def main() -> None:
    """Build the dataset and model, then train or evaluate."""
    args = parse_args()
    config = load_config(args.config)

    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["device"] = args.device

    set_seed(config["seed"])
    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    manager = DatasetManager(
        root_dir=config["data"]["root"],
        dataset_path=config["data"]["cache"],
        index_path=config["data"]["indices"],
        stats_path=config["data"]["stats"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    manager.check_split_disjoint()
    train_loader = manager.get_dataloaders()[0]

    # Draw one batch before building the model so that the parameter
    # initialisation consumes the same random stream as in the original code.
    next(iter(train_loader))

    model = HybridModel()

    batch_size = config["batch_size"]
    weights_dir = config["output"]["weights"]
    checkpoint_path = os.path.join(weights_dir, f"best_model_{batch_size}.pth")

    if args.test_only:
        load_weights = args.weights or checkpoint_path
        checkpoint = torch.load(load_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint {load_weights}")
        tester = Tester(model, device)
        tester.evaluate(
            manager.get_dataloaders()[2],
            save_dir=config["output"]["reports"],
        )
        return

    trainer = Trainer(
        model=model,
        device=device,
        dataset_manager=manager,
        stages=config["stages"],
        batch_size=batch_size,
    )

    curve_path = os.path.join(
        config["output"]["figures"], f"loss_curve_{batch_size}.png"
    )
    trainer.fit(checkpoint_path, curve_path, curve_prefix=str(batch_size))

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state"]
    )
    Tester(model, device).evaluate(
        trainer.test_loader,
        save_dir=config["output"]["reports"],
    )


if __name__ == "__main__":
    main()