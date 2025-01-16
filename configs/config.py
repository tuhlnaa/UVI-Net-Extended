from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cardiac", choices=["cardiac", "lung"])

    parser.add_argument("--weight_cycle", type=float, default=1.0)
    parser.add_argument("--weight_diff", type=float, default=1.0)

    parser.add_argument("--weight_ncc", type=float, default=1.0)
    parser.add_argument("--weight_cha", type=float, default=1.0)
    parser.add_argument("--feature_extract", action="store_true", default=True)

    return parser.parse_args()