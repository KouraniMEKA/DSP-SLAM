#
# This file is part of https://github.com/JingwenWang95/DSP-SLAM
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import warnings
import cv2
import torch
import numpy as np 
# << ak251007_PyTorch2
# TODO: review and remove legacy MMLab V1 support after full testing is complete.
from packaging.version import parse as parse_version
try:
    # For Python 3.8+
    from importlib import metadata
except ImportError:
    # For Python < 3.8
    import importlib_metadata as metadata

# MMLab V1 vs V2 API compatibility check
try:
    # Modern, preferred method
    mmcv_version_str = metadata.version('mmcv')
except metadata.PackageNotFoundError:
    # Fallback for legacy or editable installs
    import mmcv
    mmcv_version_str = mmcv.__version__

IS_MMCV_V2 = parse_version(mmcv_version_str) >= parse_version('2.0.0')

if IS_MMCV_V2:
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS
    from mmengine import Config  
    # New imports for manual inference pipeline
    from mmdet.utils import register_all_modules
    from mmengine.dataset import Compose
    from mmdet.apis import inference_detector as legacy_inference_detector # Keep old name for else block
else:
    from mmcv import Config as MMCVConfig
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    from mmdet.core import get_classes
    from mmdet.apis import inference_detector
# ak251007_PyTorch2 >>


object_class_table = {"cars": [2], "chairs": [56, 57]}


def get_detector2d(configs):
    return Detector2D(configs)


class Detector2D(object):
    def __init__(self, configs):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = configs.Detector2D.config_path
        checkpoint = configs.Detector2D.weight_path
        if isinstance(config, str):
        # << ak251007_PyTorch2
            if IS_MMCV_V2:
                self.config = Config.fromfile(config)
            else:
                self.config = MMCVConfig.fromfile(config)

        elif not isinstance(config, (MMCVConfig, Config if IS_MMCV_V2 else MMCVConfig)):
        # ak251007_PyTorch2 >>
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        
        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 support after full testing is complete.
        if IS_MMCV_V2:
            # This function will automatically find and register all MMDetection components
            # (models, backbones, etc.) into the MMEngine registry.      
            from mmdet.utils import register_all_modules
            register_all_modules()

            # MMLab V2+ models do not accept 'pretrained' argument.
            # We must remove it from the config dictionary before building the model.
            if 'pretrained' in self.config.model:
                del self.config.model['pretrained']
            self.config.model.train_cfg = None

            # Manually replicate the legacy initialization steps using the new API
            # 1. Build the model
            self.model = MODELS.build(self.config.model)

            # 2. Load the checkpoint
            if checkpoint is not None:
                load_checkpoint(self.model, checkpoint, map_location='cpu')
            
            # In MMDetection v3.x, the data pipeline is defined in test_dataloader.
            # The first step 'LoadImageFromFile' is for loading image from a path.
            # Since we are passing an already-loaded image (np.ndarray), we replace
            # it with a dummy transform that does nothing but ensures the pipeline
            # starts correctly.
            test_pipeline = self.config.test_dataloader.dataset.pipeline
            # Replace 'LoadImageFromFile' with a dummy identity transform
            if test_pipeline[0]['type'] == 'LoadImageFromFile':
                test_pipeline = test_pipeline[1:]
            self.pipeline = Compose(test_pipeline)

        else:
            # Legacy MMLab V1 requires these keys to be explicitly set.
            self.config.model.pretrained = None
            self.config.model.train_cfg = None
            
            self.model = build_detector(self.config.model, test_cfg=self.config.get('test_cfg'))

            if checkpoint is not None:
                checkpoint_data = load_checkpoint(self.model, checkpoint, map_location='cpu')
                if 'CLASSES' in checkpoint_data.get('meta', {}):
                    self.model.CLASSES = checkpoint_data['meta']['CLASSES']
                else:
                    warnings.simplefilter('once')
                    warnings.warn('Class names are not saved in the checkpoint\'s '
                                'meta data, use COCO classes by default.')
                    self.model.CLASSES = get_classes('coco') 

        self.model.cfg = self.config  # save the config in the model for convenience
        # 3. Move model to the correct device
        # self.model.to(device)
        self.model.to(self.device)
        # ak251007_PyTorch2 >>

        self.model.eval()
        self.min_bb_area = configs.min_bb_area
        self.predictions = None

    def make_prediction(self, image, object_class="cars"):
        assert object_class == "chairs" or object_class == "cars"
        
        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 support after full testing is complete.
        if IS_MMCV_V2:
            # 1. Prepare data for the pipeline
            data = dict(img=image, img_id=0, img_shape=image.shape[:2], ori_shape=image.shape[:2])

            # 2. Run the pipeline
            data = self.pipeline(data)
            
            # 3. Collate data into a batch
            from mmengine.dataset import pseudo_collate
            data = pseudo_collate([data])
            data['inputs'][0] = data['inputs'][0].to(self.device)
            data['data_samples'][0] = data['data_samples'][0].to(self.device)
                      
            # 4. Perform inference
            with torch.no_grad():
                results_list = self.model.test_step(data)
            
            # 5. Parse results from the DetDataSample object
            results = results_list[0]
            self.predictions = results
            pred_instances = results.pred_instances
            all_boxes = pred_instances.bboxes
            all_scores = pred_instances.scores
            all_labels = pred_instances.labels
            all_masks = pred_instances.masks.cpu().numpy() if 'masks' in pred_instances else np.zeros((0, *image.shape[:2]))

            boxes_list = []
            masks_list = []
            for class_id in object_class_table[object_class]:
                class_mask_filter = (all_labels == class_id)
                class_boxes = all_boxes[class_mask_filter]
                class_scores = all_scores[class_mask_filter]
                
                boxes_with_scores = torch.cat([class_boxes, class_scores.unsqueeze(1)], dim=1)
                
                boxes_list.append(boxes_with_scores.cpu().numpy())
                masks_list.append(all_masks[class_mask_filter.cpu().numpy()])

            if not boxes_list:
                boxes = np.zeros((0, 5))
                masks = np.zeros((0, 0, 0))
            else:
                boxes = np.concatenate(boxes_list, axis=0)
                masks = np.concatenate(masks_list, axis=0)

        else:
            # Legacy MMLab V1 returns a tuple of (boxes, masks)
            self.predictions = inference_detector(self.model, image)
            boxes = [self.predictions[0][o] for o in object_class_table[object_class]]
            boxes = np.concatenate(boxes, axis=0)
            masks = []
            n_det = 0
            for o in object_class_table[object_class]:
                masks += self.predictions[1][o]
                n_det += len(self.predictions[1][o])

            # In case there are no detections
            if n_det == 0:
                masks = np.zeros((0, 0, 0))
            else:
                masks = np.stack(masks, axis=0)
        # ak251007_PyTorch2 >>

        assert boxes.shape[0] == masks.shape[0]
        return self.get_valid_detections(boxes, masks)

    def visualize_result(self, image, filename):
        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 support after full testing is complete.
        if IS_MMCV_V2:
            # The new API uses a different method signature
            self.model.show_result(image, self.predictions, out_file=filename)
        else:
            # Legacy MMLab V1 show_result method
            self.model.show_result(image, self.predictions, out_file=filename)
        # ak251007_PyTorch2 >>

    def get_valid_detections(self, boxes, masks):
        # Remove those on the margin
        cond1 = (boxes[:, 0] >= 30) & (boxes[:, 1] > 10) & (boxes[:, 2] < 1211) & (boxes[:, 3] < 366)
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # Remove those with too small bounding boxes
        cond2 = (boxes_area > self.min_bb_area)
        scores = boxes[:, -1]
        cond3 = (scores >= 0.70)

        valid_mask = (cond2 & cond3)
        valid_instances = {"pred_boxes": boxes[valid_mask, :4],
                           "pred_masks": masks[valid_mask, ...]}

        return valid_instances

    @staticmethod
    def save_masks(masks):
        mask_imgs = masks.cpu().numpy()
        n = mask_imgs.shape[0]
        for i in range(n):
            cv2.imwrite("mask_%d.png" % i, mask_imgs[i, ...].astype(np.float32) * 255.)
