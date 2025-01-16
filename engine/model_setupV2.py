import torch
import voxelmorph as vxm
from torch import optim

from utils import utils
from models.VoxelMorph.model import VoxelMorph
from models.UNet.model import Unet3D, Unet3D_multi
from models.feature_extract.model import FeatureExtract

def setup_models(args, img_size):
    # VoxelMorph configuration
    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
    elif args.dataset == "lung":
        img_size = (128, 128, 128)

    nb_features = [
        [16, 32, 32, 32],             # encoder features
        [32, 32, 32, 32, 32, 16, 16]  # decoder features
    ]
    inshape = [128, 128, 128]

    # Initialize models
    # flow_model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=nb_features,
    #     src_feats=1,    # number of source image features
    #     trg_feats=1,    # number of target image features
    #     bidir=True,     # enable bidirectional registration
    #     int_steps=0,    # disable integration steps to get raw flow fields
    #     int_downsize=1  # prevent downsampling of flow field
    # ).cuda()

    # flow_model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=nb_features,
    #     src_feats=1,
    #     trg_feats=1,
    #     bidir=True,
    #     int_steps=0,              
    #     int_downsize=1,           
    #     nb_unet_conv_per_level=1  
    # ).cuda()

    flow_model = VoxelMorph(img_size).cuda()

    refinement_model = Unet3D_multi(img_size).cuda() if args.feature_extract else Unet3D(img_size).cuda()
    feature_model = FeatureExtract().cuda() if args.feature_extract else None

    # Initialize registration models
    reg_model = utils.register_model(img_size, "nearest").cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear").cuda()

    for param in reg_model.parameters():
        param.requires_grad = False
    for param in reg_model_bilin.parameters():
        param.requires_grad = False

    # Initialize optimizer
    if args.feature_extract:
        optimizer = optim.Adam(
            list(flow_model.parameters()) + 
            list(refinement_model.parameters()) + 
            list(feature_model.parameters()),
            lr=args.lr,
            weight_decay=0,
            amsgrad=True,
        )
    else:
        optimizer = optim.Adam(
            list(flow_model.parameters()) + list(refinement_model.parameters()),
            lr=args.lr,
            weight_decay=0,
            amsgrad=True,
        )

    return flow_model, refinement_model, feature_model, reg_model, reg_model_bilin, optimizer