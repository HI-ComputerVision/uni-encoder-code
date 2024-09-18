# Demo

- Pick a model and its config file from. For example, `configs/cityscapes/swin/unified_encoder_cityscapes.yaml`.
- We provide `demo.py` that is able to demo builtin configs.
- You need to specify the `task` token value during inference, The outputs will be saved accordingly in the specified `OUTPUT_DIR`:
  - `panoptic`: Panoptic, Semantic and Instance Predictions when the value of `task` token is `panoptic`.
  - `instance`: Instance Predictions when the value of `task` token is `instance`.
  - `semantic`: Semantic Predictions when the value of `task` token is `semantic`.

```bash
export task=panoptic

python demo.py --config-file ../configs/citysapes/swin/unified_encoder_cityscapes.yaml \
  --input <path-to-images> \
  --output <output-path> \
  --task $task \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS <path-to-checkpoint>
```

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. 