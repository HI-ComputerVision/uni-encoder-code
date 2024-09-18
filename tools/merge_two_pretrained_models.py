import pickle as pkl
import sys

import torch

"""
Usage:
  # do some conversions first using other scripts to produce pkl files
  # run the merge script
  ./merge_two_pretrained_models.py swin_tiny_patch4_window7_224.pkl r18.pkl swin_tiny_patch4_window7_224_r18.pkl
  # Then, use swin_tiny_patch4_window7_224.pkl with the following changes in config:
MODEL:
  WEIGHTS: "/path/to/swin_tiny_patch4_window7_224_r18.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    model1 = sys.argv[1]
    model2 = sys.argv[2]

    with open(model1, "rb") as f:
        obj1 = pkl.load(f)["model"]
    with open(model2, "rb") as f:
        obj2 = pkl.load(f)["model"]

    res = {"model": None, "__author__": "third_party", "matching_heuristics": True}
    res["model"] = obj1
    res["model"].update(obj2)

    with open(sys.argv[3], "wb") as f:
        pkl.dump(res, f)
