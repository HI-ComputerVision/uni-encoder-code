# *Name of the Repository**

## **Description**  
This repository builds upon the OneFormer framework for unified image segmentation tasks and integrates with Detectron2-based workflows. For more details and a complete understanding of Detectron2 usage, refer to the [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md).

## **RDI Method: EDC(s)**  
*(If applicable, list EDCs here and follow the guidelines provided in the original template. Otherwise, remove this section.)*

| EDCs  | PDE | Internal Article | Current State | Main File | Main Contributor | Corresponding EDC |
|-------|-----|-----------------|---------------|-----------|-----------------|------------------|
| ...   | ... | ...             | ...           | ...       | ...             | ...              |


## **Requirements**  
- Python 3.8+  
- PyTorch and Detectron2 installed (compatible versions listed in the repository's requirements)  
- OneFormer dependencies (refer to OneFormer documentation for details)

## **Installation**  
Install dependencies using the provided `requirements.txt` (if available) or follow standard Detectron2 and OneFormer installation steps.

```
pyhon -m pip install -r /path/to/requirements.txt
```
## **Usage**  
Below are guidelines for evaluating models and running inference demos.

### Evaluation  
You must specify a `task` when evaluating your model. The possible tasks are `panoptic`, `semantic`, or `instance`.  
Note that depth evaluation is automatically included regardless of the chosen task.

**Example Command:**

```
pyhon train_net.py --dist-url 'tcp://127.0.0.1:50164' \
    --num-gpus 1 \
    --config-file configs/citysapes/swin/unified_encoder_cityscapes.yaml \
    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \
    MODEL.TEST.TASK <task>
```
## **Documentation**  
For a detailed walkthrough of using OneFormer and Detectron2 features, refer to their respective documentations:

- [Detectron2 Getting Started](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)  
- [OneFormer Documentation](https://github.com/SHI-Labs/OneFormer)

Additional explanations, feature guides, and detailed methodology are provided in the repository's internal documentation and code comments.

## **Inference Demo**  
A demo script is provided for running inference on images. For instructions, usage examples, and supported input formats, please see the [demo/README.md](demo/README.md).

## **Contributing**  
If you would like to contribute, please:  
- Open issues for bugs or feature suggestions.  
- Submit merge/pull requests for proposed changes.  
- Follow the project's coding standards and contribution guidelines.

## **License**  
This code is for internal use and research purposes only.

## **Contact**  
For any inquiries or further information, please contact the maintainer:  
- Name: Huy-Dung NGUYEN 
- Email: huy-dung.nguyen@capgemini.com

## **Acknowledgements**  
We acknowledge the OneFormer and Detectron2 teams and their extensive documentation that made this integration possible.

## **References**  
- OneFormer: [https://github.com/SHI-Labs/OneFormer](https://github.com/SHI-Labs/OneFormer) 
- Detectron2: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)