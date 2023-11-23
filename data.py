#This will download dataset images from OpenImagesV7 for Training and prepare them for The Yolo model to receive it.

import fiftyone as fo
import fiftyone.zoo as foz
import warnings
from ultralytics.utils import LOGGER, SETTINGS, Path, is_ubuntu, get_ubuntu_version
from ultralytics.utils.checks import check_requirements, check_version

name = 'open-images-v7'
fraction = 0.001  # fraction of full dataset to use

for split in 'train', 'validation':  # 1743042 train, 41620 val images
     train = split == 'train'

      # Load Open Images dataset
     dataset = foz.load_zoo_dataset(name,
                                     split=split,
                                     label_types=['detections'],
                                     classes=['Person'],
                                     dataset_dir=Path(SETTINGS['datasets_dir']) / 'fiftyone' / name,
                                     max_samples=round((1743042 if train else 41620) * fraction))

    

      # Export to YOLO format
     with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")
          dataset.export(export_dir=str(Path(SETTINGS['datasets_dir']) / name), # type: ignore
                         dataset_type=fo.types.YOLOv5Dataset, # type: ignore
                         label_field='ground_truth',
                         split='val' if split == 'validation' else split,
                         classes=['Person'],
                         overwrite=train)