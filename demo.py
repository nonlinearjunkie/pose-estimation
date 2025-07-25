import os
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import coco
import utils
import RCNN as modellib
import visualize

import torch
import cv2
from dataset import NOCSData
import datetime



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

TRAINED_PATH = 'models/NOCS_Trained_2.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Path to specific image
IMAGE_SPECIFIC = None

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6 # Background plus 6 classes
    OBJ_MODEL_DIR = os.path.join('data','obj_models')


config = InferenceConfig()
config.display()

synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                ]
import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import coco
import utils
import RCNN as modellib
import visualize

import torch
import cv2
from dataset import NOCSData
import datetime



# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")

TRAINED_PATH = 'models/NOCS_Trained_2.pth'

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Path to specific image
IMAGE_SPECIFIC = None

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 6 # Background plus 6 classes
    OBJ_MODEL_DIR = os.path.join('data','obj_models')


config = InferenceConfig()
config.display()

synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]


class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }


config.display()

model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

if config.GPU_COUNT > 0:
    device = torch.device('cuda')
    model.load_state_dict(torch.load(TRAINED_PATH))
else:
    device = torch.device('cpu')
    model.load_state_dict(torch.load(TRAINED_PATH,map_location=torch.device('cpu')))

print("Model to:",device)

model.to(device)

save_dir = 'output_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

now = datetime.datetime.now()

# Whether to use synthetic or real data
use_camera_data = False

# Change this to detect on different img
image_id = 1

if use_camera_data:
    camera_dir = os.path.join('data', 'camera')
    dataset = NOCSData(synset_names,'val')
    dataset.load_camera_scenes(camera_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth=dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data="camera/val"
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # for camera data

else:# for real data
    real_dir = os.path.join('data', 'real')
    dataset = NOCSData(synset_names,'test')
    dataset.load_real_scenes(real_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth=dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data="real/test"
    intrinsics= np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]) # for real data


start_time = datetime.datetime.now()

result = {}
gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
gt_bbox = utils.extract_bboxes(gt_mask)
result['image_id'] = image_id
result['image_path'] = image_path
result['gt_class_ids'] = gt_class_ids
result['gt_bboxes'] = gt_bbox
result['gt_RTs'] = None            
result['gt_scales'] = gt_scales


detect= True
if detect:

    if image.shape[2] == 4:
        image = image[:,:,:3]

    with torch.no_grad():
        # Run detection
        results = model.detect([image])
        # Visualize results
        r = results[0]
        rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

        r['coords'][:,:,:,2]=1-r['coords'][:,:,:,2]

        umeyama = True

        if umeyama:

            result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                                r['masks'], 
                                                                                                r['coords'], 
                                                                                                depth, 
                                                                                                intrinsics, 
                                                                                                synset_names,  image_path)
            draw_rgb = False
            result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
            utils.draw_detections(image, save_dir, data, image_id, intrinsics, synset_names, draw_rgb,
                                                    gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                                    r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'],draw_gt=False)

end_time = datetime.datetime.now()
execution_time = end_time - start_time

print("Time taken for execution:", execution_time)
                    'mug'#6
                    ]


class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }


config.display()

model = modellib.MaskRCNN(config=config, model_dir=MODEL_DIR)

if config.GPU_COUNT > 0:
    device = torch.device('cuda')
    model.load_state_dict(torch.load(TRAINED_PATH))
else:
    device = torch.device('cpu')
    model.load_state_dict(torch.load(TRAINED_PATH,map_location=torch.device('cpu')))

print("Model to:",device)

model.to(device)

save_dir = 'output_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

now = datetime.datetime.now()

# Whether to use synthetic or real data
use_camera_data = False

# Change this to detect on different img
image_id = 1

if use_camera_data:
    camera_dir = os.path.join('data', 'camera')
    dataset = NOCSData(synset_names,'val')
    dataset.load_camera_scenes(camera_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth=dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data="camera/val"
    intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]) # for camera data

else:# for real data
    real_dir = os.path.join('data', 'real')
    dataset = NOCSData(synset_names,'test')
    dataset.load_real_scenes(real_dir)
    dataset.prepare(class_map)

    image = dataset.load_image(image_id)
    depth=dataset.load_depth(image_id)
    image_path = dataset.image_info[image_id]["path"]

    data="real/test"
    intrinsics= np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]) # for real data


start_time = datetime.datetime.now()

result = {}
gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
gt_bbox = utils.extract_bboxes(gt_mask)
result['image_id'] = image_id
result['image_path'] = image_path
result['gt_class_ids'] = gt_class_ids
result['gt_bboxes'] = gt_bbox
result['gt_RTs'] = None            
result['gt_scales'] = gt_scales


detect= True
if detect:

    if image.shape[2] == 4:
        image = image[:,:,:3]

    with torch.no_grad():
        # Run detection
        results = model.detect([image])
        # Visualize results
        r = results[0]
        rois, masks, class_ids, scores, coords = r['rois'], r['masks'], r['class_ids'], r['scores'],r['coords']

        r['coords'][:,:,:,2]=1-r['coords'][:,:,:,2]

        umeyama = True

        if umeyama:

            result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                                r['masks'], 
                                                                                                r['coords'], 
                                                                                                depth, 
                                                                                                intrinsics, 
                                                                                                synset_names,  image_path)
            draw_rgb = False
            result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
            utils.draw_detections(image, save_dir, data, image_id, intrinsics, synset_names, draw_rgb,
                                                    gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                                    r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'],draw_gt=False)

end_time = datetime.datetime.now()
execution_time = end_time - start_time

print("Time taken for execution:", execution_time)