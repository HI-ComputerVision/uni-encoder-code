# Prepare Datasets

- A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog) for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
- Detectron2 has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Under this directory, detectron2 will look for datasets in the structure described below, if needed.

  ```text
  $DETECTRON2_DATASETS/
    cityscapes/
  ```

- You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [Cityscapes](https://www.cityscapes-dataset.com/downloads/)

```text
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```

- Login and download the dataset

  ```bash
  wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
  ######## gtFine
  wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
  ######## leftImg8bit
  wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
  ```

- Install cityscapes scripts by:

  ```bash
  pip install git+https://github.com/mcordts/cityscapesScripts.git
  ```

- To create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:

  ```bash
  git clone https://github.com/mcordts/cityscapesScripts.git
  ```

  ```bash
  CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py
  ```

  These files are not needed for instance segmentation.

- To generate Cityscapes panoptic dataset, run cityscapesescript with:

  ```bash
  CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesScripts/cityscapesscripts/preparation/createPanopticImgs.py
  ```

  These files are not needed for semantic and instance segmentation.

- You need to manually crop the images as mention in our paper for the reproducibility.
