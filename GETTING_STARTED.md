# Getting Started

This document is based on the documentation of OneFormer.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

## Evaluation

- You need to pass the value of `task` token. `task` belongs to [panoptic, semantic, instance].

- The depth evaluation is automatically provided regardless the value of the `task` token.

- To evaluate a model's performance, use:

```bash
python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 8 \
    --config-file configs/citysapes/swin/unified_encoder_cityscapes.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>
```

## Inference Demo

We provide a demo script for inference on images. For more information, please see [demo/README.md](demo/README.md).
