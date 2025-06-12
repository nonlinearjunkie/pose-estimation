import os
import torch
import argparse
import numpy as np
from config import Config
from RCNN import MaskRCNN
from VIT import MaskRCNNViT
from dataset import NOCSData

# root dir of the project
ROOT_DIR = os.getcwd()

# path to save models
RESNET_MODEL_DIR = os.path.join(ROOT_DIR, "resnet_trained_ckpts")
VIT_MODEL_DIR = os.path.join(ROOT_DIR, "vit_trained_ckpts")

#path to coco rcnn checkpoint
COCO_CKPT_PATH = os.path.join(ROOT_DIR,"ckpts", "mask_rcnn_coco.pth")

#logs dir http://download.cs.stanford.edu/orion/nocs/obj_models.zip
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DATASET_DIR = os.path.join(ROOT_DIR, "data")

class RCNNTrainConfig(Config):

    NAME = "NOCS_RCNN_train"
    OBJ_MODEL_DIR = os.path.join(ROOT_DIR,'data','obj_models')
    GPU_COUNT=1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+6 #background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    COORD_USE_BINS = True
    COORD_NUM_BINS = 32
    COORD_SHARE_WEIGHTS = False
    COORD_USE_DELTA = False
    COORD_POOL_SIZE = 14
    COORD_SHAPE = [28, 28]

    USE_MINI_MASK = False

class ViTBaseTrainConfig(RCNNTrainConfig):
    NAME = "NOCS_ViT_Base_train"
    BACKBONE = "vit_base_patch16_224.dino"
    OUT_INDICIES = (3, 6, 9, 11)
    WEIGHT_DECAY = 1e-3
    LEARNING_RATE = 1e-4  

class ViTEva02TrainConfig(RCNNTrainConfig):
    NAME = "NOCS_ViT_Eva02_train"
    BACKBONE = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    OUT_INDICIES = (6, 11, 17, 23)
    WEIGHT_DECAY = 1e-3
    LEARNING_RATE = 1e-4 



def load_resnet_backbone_model(config, rand_weights= False, trained_path=None):
    model = MaskRCNN(config = config, model_dir = RESNET_MODEL_DIR)
    pretrained_state_dict = torch.load(COCO_CKPT_PATH)

    if trained_path:
        model.load_state_dict(torch.load(trained_path))

    elif rand_weights:
        exclude_layers = ["classifier", "mask"]
        filtered_state_dict = {k:v for k,v in pretrained_state_dict.items() if not any(layer in k for layer in exclude_layers)}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        filtered_state_dict = pretrained_state_dict
        mismatches = ["classifier.linear_class.weight","classifier.linear_class.bias",
                      "classifier.linear_bbox.weight","classifier.linear_bbox.bias",
                      "mask.conv5.weight","mask.conv5.bias"]
        
        for i in range(len(mismatches)):

            weights = filtered_state_dict[mismatches[i]]

            if weights.shape[0] == 81 and weights.dim() > 1:
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.vstack((w1,w3,w2))
                pass

            elif weights.shape[0] == 324 and len(weights.shape) > 1:
                weights = torch.reshape(weights, (81,4,1024))

                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.vstack((w1.flatten(end_dim=-2),w3.flatten(end_dim=-2),w2.flatten(end_dim=-2)))

            elif weights.shape[0] == 324:
                weights = torch.reshape(weights, (81,4))
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.cat((w1.flatten(),w3.flatten(),w2.flatten()))
            else:
                w1 = weights[[0,40,46]]
                w2 = weights[[64,42]]
                w3 = torch.zeros_like(w2)

                final_weights = torch.cat((w1,w3,w2))

            filtered_state_dict[mismatches[i]] = final_weights
        model.load_state_dict(filtered_state_dict, strict=False)

        if config.GPU_COUNT>0 and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        return model        



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--backbone', choices=['vit-base',"eva-02", 'resnet'], required=True,
                    help="Choose the backbone architecture: 'vit' or 'resnet'.")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    
    #dataset dirs
    camera_dir = os.path.join(DATASET_DIR, "camera")
    real_dir = os.path.join(DATASET_DIR, "real")

    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
    
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
    
    class_map = {
        "bottle":"bottle",
        "bowl":"bowl",
        "cup":"mug",
        "laptop":"laptop"
    }

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)

    # camera_train_data = NOCSData(synset_names, "train")
    # camera_train_data.load_camera_scenes(camera_dir)
    # camera_train_data.prepare(class_map)

    real_train_data = NOCSData(synset_names, "train", RCNNTrainConfig())
    real_train_data.load_real_scenes(real_dir)
    real_train_data.prepare(class_map)

    val_data = NOCSData(synset_names, "test", RCNNTrainConfig())
    val_data.load_real_scenes(real_dir)
    val_data.prepare(class_map)    

    if args.backbone == "resnet":
        config = RCNNTrainConfig() 
        model = load_resnet_backbone_model(config)

        # Stage - 01 (Training heads only)
        print("Training network heads")
        model.train_model(real_train_data, val_data,
                          learning_rate = config.LEARNING_RATE,
                          epochs =1,
                          layers = "heads")
    
        # Stage - 02 (Finetune ResNet stage 4 and up)
        print("Training ResNet layer 4+")
        model.train_model(real_train_data, val_data,
                          learning_rate = config.LEARNING_RATE/10,
                           epochs =1,
                           layers = "4+")
    
        # Stage - 03 (Finetune all layers)
        print("Finetuning all ResNet layers...")
        model.train_model(real_train_data, val_data,
                          learning_rate = config.LEARNING_RATE/10,
                          epochs = 1,
                          layers = "all")
        
    else:
        if args.backbone == 'vit-base':
            config = ViTBaseTrainConfig()
        elif args.backbone == "eva-02":
            config = ViTEva02TrainConfig()
        else:
            raise NotImplementedError("This backbone is not supported yet.")
        config.display()
        model = MaskRCNNViT(config=config, model_dir=VIT_MODEL_DIR)
        if config.GPU_COUNT > 0 and torch.cuda.is_available():
            device = torch.device('cuda')
        
        else:
            device = torch.device('cpu')

        print("Model to:", device)
        model.to(device)

        # Training - Stage 1
        print("Training network heads")
        model.train_model(real_train_data, val_data,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100,
                    layers='heads')
    
        # Training - Stage 2
        print("Training network all layers")
        model.train_model(real_train_data, val_data,
                    learning_rate=config.LEARNING_RATE/4,
                    epochs=200,
                    layers='all')


    

    