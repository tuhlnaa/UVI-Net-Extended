from dataclasses import dataclass
import random
from typing import Optional
from argparse import ArgumentParser


@dataclass
class Config:
    """Training configuration parameters."""
    # Basic training parameters
    seed: int
    lr: float
    batch_size: int
    max_epoch: int
    split: Optional[int]
    gpu: Optional[str]
    dataset: str
    
    # Model specific parameters
    weight_cycle: float
    weight_diff: float
    weight_ncc: float
    weight_cha: float
    feature_extract: bool


def get_config() -> Config:
    """Parse command line arguments and return Config object.
    
    Returns:
        Config object containing all parameters
    """
    parser = ArgumentParser()

    # Basic training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cardiac", 
        choices=["cardiac", "lung"]
    )

    # Model specific parameters
    parser.add_argument("--weight_cycle", type=float, default=1.0)
    parser.add_argument("--weight_diff", type=float, default=1.0)
    parser.add_argument("--weight_ncc", type=float, default=1.0)
    parser.add_argument("--weight_cha", type=float, default=1.0)
    parser.add_argument("--feature_extract", action="store_true", default=True)

    args = parser.parse_args()
    return Config(**vars(args))