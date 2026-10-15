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

import torch

# << ak251007_PyTorch2
# TODO: review and remove legacy MMLab V1 API support after full testing is complete.
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
    from mmdet3d.apis import init_model, inference_detector
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import MODELS
    from mmengine import Config  
else:
    from mmcv import Config as MMCVConfig
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from mmdet3d.apis import inference_detector, convert_SyncBN
# ak251007_PyTorch2 >>


def get_detector3d(configs):
    return Detector3D(configs)


class Detector3D(object):
    def __init__(self, configs):
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = configs.Detector3D.config_path
        checkpoint = configs.Detector3D.weight_path

        if isinstance(config, str):
        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 API support after full testing is complete.
            if IS_MMCV_V2:
                config = Config.fromfile(config)
            else:
                config = MMCVConfig.fromfile(config)

        elif not isinstance(config, (MMCVConfig, Config if IS_MMCV_V2 else MMCVConfig)):
        # ak251007_PyTorch2 >>
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')

        
        config.model.train_cfg = None

        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 API support after full testing is complete.
        if IS_MMCV_V2:
            from mmdet3d.utils import register_all_modules as register_all_modules_mmdet3d
            from mmdet.utils import register_all_modules as register_all_modules_mmdet
            from mmengine.config import ConfigDict
            
            register_all_modules_mmdet3d(init_default_scope=True)
            register_all_modules_mmdet(init_default_scope=False)

            if 'bbox_head' in config.model:
                if 'loss_cls' in config.model.bbox_head:
                    config.model.bbox_head.loss_cls['_scope_'] = 'mmdet'
                if 'loss_bbox' in config.model.bbox_head: 
                    config.model.bbox_head.loss_bbox['_scope_'] = 'mmdet'
                if 'loss_dir' in config.model.bbox_head: 
                    config.model.bbox_head.loss_dir['_scope_'] = 'mmdet'

            if 'pretrained' in config.model:
                del config.model['pretrained']
            if 'voxel_layer' in config.model:
                config.model.data_preprocessor = dict(
                    type='Det3DDataPreprocessor',
                    voxel=True,
                    voxel_layer=config.model['voxel_layer']
                )
                del config.model['voxel_layer']

            config.model.train_cfg = None

            self.model = MODELS.build(config.model)

            if checkpoint is not None:
                load_checkpoint(self.model, checkpoint, map_location='cpu', revise_keys=[(r'^pts_', '')])

            config.test_dataloader = ConfigDict(
                dataset=ConfigDict(
                    box_type_3d='LiDAR',
                    pipeline=[
                        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                        dict(type='Pack3DDetInputs', keys=['points'])
                    ]
                )
            )

        else:
            # Legacy MMLab V1 requires these keys to be explicitly set.
            config.model.pretrained = None
            convert_SyncBN(config.model)
            config.model.train_cfg = None
            
            self.model = build_model(config.model, test_cfg=config.get('test_cfg'))
            
            if checkpoint is not None:
                checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
                if 'CLASSES' in checkpoint['meta']:
                    self.model.CLASSES = checkpoint['meta']['CLASSES']
                else:
                    self.model.CLASSES = config.class_names
                if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
                    self.model.PALETTE = checkpoint['meta']['PALETTE']

        self.model.cfg = config  # save the config in the model for convenience
        # 3. Move model to the correct device
        self.model.to(device)
        self.model.eval()
        # ak251007_PyTorch2 >>


    def make_prediction(self, velo_file):
        # << ak251007_PyTorch2
        # TODO: review and remove legacy MMLab V1 API support after full testing is complete.
        if IS_MMCV_V2:
            # MMLab V2+ returns a structured Det3DDataSample object
            results = inference_detector(self.model, velo_file)
            pred_instances = results[0].pred_instances_3d

            # Extract data from the new structured object
            labels = pred_instances.labels_3d
            scores = pred_instances.scores_3d
            boxes = pred_instances.bboxes_3d.tensor

            # Normalize the output format to [x, y, z, w, l, h, yaw]
            # This encapsulates the version-specific logic within the detector.
            # Input format is [x, y, z, l, w, h, yaw]
            # Swap l and w to get [x, y, z, w, l, h, yaw]
            boxes[:, [3, 4]] = boxes[:, [4, 3]]
            boxes[:, 6] = boxes[:, 6] - (torch.pi / 2)
        else:
            # Legacy MMLab V1 returns a tuple of (predictions, data)
            predictions, data = inference_detector(self.model, velo_file)
            labels = predictions[0]["labels_3d"]
            scores = predictions[0]["scores_3d"]
            boxes = predictions[0]["boxes_3d"].tensor
            # Input format is [x, y, z, w, l, h, yaw]
            # Correct the yaw angle (# ak251007 reverse theta sign here instead from inside T_velo_obj computation)
            boxes[:, 6] = -boxes[:, 6]
        # ak251007_PyTorch2 >>

        # Car's label is 0 in KITTI dataset
        valid_mask = (labels == 0) & (scores > 0.0)
        return boxes[valid_mask]
