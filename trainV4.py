import os
os.environ["VXM_BACKEND"] = "pytorch"

import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

import losses
from utils import datasets, utils
from models.UNet.model import Unet3D, Unet3D_multi
from models.VoxelMorph.model import VoxelMorph
from models.feature_extract.model import FeatureExtract
import voxelmorph as vxm

from data.datasets import ACDCHeartDataset, LungDataset
from models.UNet.modelV4 import UNet3D, UNet3DMulti

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main(args):
    set_seed(args.seed)

    if not os.path.exists(f"experiments/{args.dataset}"):
        os.makedirs(f"experiments/{args.dataset}")

    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if args.split is None else args.split
    elif args.dataset == "lung":
        img_size = (128, 128, 128)
        split = 68 if args.split is None else args.split

    """
    Initialize model
    """

    # configure unet features
    nb_features = [
        [16, 32, 32, 32],             # encoder features
        [32, 32, 32, 32, 32, 16, 16]  # decoder features
    ]
    refinement_nb_features = [
        [8, 16, 32],             # encoder features
        [32, 32, 32, 8, 8, 3]    # decoder features
    ]
    inshape = [128, 128, 128]
    additional_dims = [4, 8, 16]

    # build model
    # flow_model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=nb_features,
    #     src_feats=1,    # number of source image features
    #     trg_feats=1,    # number of target image features
    #     bidir=True,     # enable bidirectional registration
    #     int_steps=0,    # disable integration steps to get raw flow fields
    #     int_downsize=1  # prevent downsampling of flow field
    # ).cuda()

    # 原始
    # flow_model = vxm.networks.VxmDense(
    #     inshape=inshape,
    #     nb_unet_features=nb_features,
    #     src_feats=1,
    #     trg_feats=1,
    #     bidir=True,
    #     int_steps=0,              # Disable integration steps
    #     int_downsize=1,           # Disable downsampling
    #     nb_unet_conv_per_level=1  # Match original conv layers per level
    # ).cuda()

    flow_model = VoxelMorph(img_size).cuda()  # Flow calculation model

    if args.feature_extract:
        #refinement_model = Unet3D_multi(img_size).cuda()
        refinement_model = UNet3DMulti(inshape, feature_maps=refinement_nb_features, additional_dims=additional_dims).cuda()
    else:
        #refinement_model = Unet3D(img_size).cuda()
        refinement_model = UNet3D(inshape, encoder_features=refinement_nb_features[0], decoder_features=refinement_nb_features[1]).cuda()

    if args.feature_extract:
        feature_model = FeatureExtract().cuda()

    """
    Initialize spatial transformation function
    """
    reg_model = utils.register_model(img_size, "nearest").cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear").cuda()
    for param in reg_model.parameters():
        param.requires_grad = False
    for param in reg_model_bilin.parameters():
        param.requires_grad = False

    """
    Initialize training
    """
    if args.dataset == "cardiac":
        data_dir = os.path.join("dataset", "ACDC_database", "training")
        train_dataset = datasets.ACDCHeartDataset(data_dir, phase="train", split=split)
        val_dataset = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    elif args.dataset == "lung":
        data_dir = os.path.join("dataset", "4D-Lung_Preprocessed")
        train_dataset = datasets.LungDataset(data_dir, phase="train", split=split)
        val_dataset = datasets.LungDataset(data_dir, phase="test", split=split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # if args.dataset == "cardiac":
    #     data_path = r"./dataset/ACDC_database"
    #     train_dataset = ACDCHeartDataset(
    #         data_path=data_path,
    #         phase="train",
    #         split=90,
    #         image_size=(128, 128, 32)
    #     )
    #     val_dataset = ACDCHeartDataset(
    #         data_path=data_path,
    #         phase="val",
    #         split=90,
    #         image_size=(128, 128, 32)
    #     )

    # elif args.dataset == "lung":
    #     data_path = r"./dataset/4D-Lung_Preprocessed"
    #     train_dataset = LungDataset(
    #         data_path=data_path,
    #         phase="train",
    #         split=68,
    #         image_size=(128, 128, 128)
    #     )
    #     val_dataset = LungDataset(
    #         data_path=data_path,
    #         phase="val",
    #         split=68,
    #         image_size=(128, 128, 128)
    #     )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )


    if args.feature_extract:
        optimizer = optim.Adam(
            list(flow_model.parameters())
            + list(refinement_model.parameters())
            + list(feature_model.parameters()),
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
    criterion_ncc = losses.NCC()
    criterion_cha = losses.CharbonnierLoss
    criterion_reg = losses.Grad3d(penalty="l2")
    criterion_l1n = losses.L1_norm()
    epsilon = 1e-3

    best_ncc = -1
    wandb.init(project="UVI-Net", name=args.dataset, config=args)

    for epoch in trange(args.max_epoch):
        """
        Training
        """
        loss_all = utils.AverageMeter()
        loss_all_full = utils.AverageMeter()
        loss_ncc_all_full = utils.AverageMeter()
        loss_cha_all_full = utils.AverageMeter()
        loss_reg_all_full = utils.AverageMeter()
        loss_all_cycle = utils.AverageMeter()
        loss_diff_all = utils.AverageMeter()

        for idx, data in enumerate(train_loader):
            refinement_model.train()
            flow_model.train()

            data = [t.cuda() for t in data]
            image0 = data[0]
            image1 = data[1]

            alpha1 = random.uniform(-0.5, 0.0)
            alpha2 = random.uniform(0.0, 1.0)
            alpha3 = random.uniform(1.0, 1.5)

            image0_image1 = torch.cat((image0, image1), dim=1)
            source, target = image0_image1[:, 0:1], image0_image1[:, 1:2]

            image_0_1, image_1_0, flow_0_1, flow_1_0 = flow_model(image0_image1)  # Forward pass

            # y_source, y_target, flow_field = flow_model(source, target)
            # # The flow field will be full size and include both directions
            # flow_0_1 = flow_field[:, :3]  # first 3 channels
            # flow_1_0 = -flow_field[:, :3]  # same magnitude, opposite direction
            # i_0_1 = y_source
            # i_1_0 = y_target

            loss_ncc_1 = criterion_ncc(image_0_1, image1) * args.weight_ncc
            loss_cha_1 = criterion_cha(image_0_1, image1, eps=epsilon) * args.weight_cha
            loss_reg_1 = criterion_reg(flow_0_1, None)
            loss_ncc_0 = criterion_ncc(image_1_0, image0) * args.weight_ncc
            loss_cha_0 = criterion_cha(image_1_0, image0, eps=epsilon) * args.weight_cha
            loss_reg_0 = criterion_reg(flow_1_0, None)
            flow_model_loss = (loss_ncc_1 + loss_cha_1 + loss_reg_1 + loss_ncc_0 + loss_cha_0 + loss_reg_0)

            loss_all_full.update(flow_model_loss.item(), image1.numel())
            loss_ncc_all_full.update(loss_ncc_1.item(), image1.numel())
            loss_cha_all_full.update(loss_cha_1.item(), image1.numel())
            loss_reg_all_full.update(loss_reg_1.item(), image1.numel())
            loss_ncc_all_full.update(loss_ncc_0.item(), image1.numel())
            loss_cha_all_full.update(loss_cha_0.item(), image1.numel())
            loss_reg_all_full.update(loss_reg_0.item(), image1.numel())

            if args.weight_cycle == 0:
                optimizer.zero_grad()       # Reset gradients
                flow_model_loss.backward()  # Backward pass
                optimizer.step()            # Update parameters

                loss_all.update(flow_model_loss.item(), image0.numel())

                continue

            """
            First Interpolation
            """
            flow0_t1 = flow_0_1 * alpha1
            image0_t1 = reg_model_bilin([image0, flow0_t1.float()])

            if alpha2 < 0.5:
                flow0_t2 = flow_0_1 * alpha2
                image0_t2 = reg_model_bilin([image0, flow0_t2.float()])
                image_unknown_t2 = image0_t2
            else:
                flow1_t2 = flow_1_0 * (1 - alpha2)
                image1_t2 = reg_model_bilin([image1, flow1_t2.float()])
                image_unknown_t2 = image1_t2

            flow1_t3 = flow_1_0 * (1 - alpha3)
            image1_t3 = reg_model_bilin([image1, flow1_t3.float()])

            """
            Second Interpolation
            """
            image_t1_image_t2 = torch.cat((image0_t1, image_unknown_t2), dim=1)
            source, target = image_t1_image_t2[:, 0:1], image_t1_image_t2[:, 1:2]

            _, _, flow_t1_t2, flow_t2_t1 = flow_model(image_t1_image_t2)

            # y_source, y_target, flow_field = flow_model(source, target)
            # # The flow field will be full size and include both directions
            # flow_a1_a2 = flow_field[:, :3]  # first 3 channels
            # flow_a2_a1 = -flow_field[:, :3]  # same magnitude, opposite direction
            # _ = y_source
            # _ = y_target

            image_t2_image_t3 = torch.cat((image_unknown_t2, image1_t3), dim=1)
            source, target = image_t2_image_t3[:, 0:1], image_t2_image_t3[:, 1:2]

            _, _, flow_t2_t3, flow_t3_t2 = flow_model(image_t2_image_t3)

            # y_source, y_target, flow_field = flow_model(source, target)
            # # The flow field will be full size and include both directions
            # flow_a2_a3 = flow_field[:, :3]  # first 3 channels
            # flow_a3_a2 = -flow_field[:, :3]  # same magnitude, opposite direction
            # _ = y_source
            # _ = y_target


            t1_t2 = (0 - alpha1) / (alpha2 - alpha1)  # max=1, min=0
            t2_t3 = (1 - alpha2) / (alpha3 - alpha2)  # max=2, min=0

            flow_t1_0 = flow_t1_t2 * t1_t2
            flow_t2_0 = flow_t2_t1 * (1 - t1_t2)
            flow_t2_1 = flow_t2_t3 * t2_t3
            flow_t3_1 = flow_t3_t2 * (1 - t2_t3)

            image_t1_0 = reg_model_bilin([image0_t1, flow_t1_0.float()])
            image_t2_0 = reg_model_bilin([image_unknown_t2, flow_t2_0.float()])
            image_t2_1 = reg_model_bilin([image_unknown_t2, flow_t2_1.float()])
            image_t3_1 = reg_model_bilin([image1_t3, flow_t3_1.float()])

            image0_combined = (1 - t1_t2) * image_t1_0 + t1_t2 * image_t2_0
            image1_combined = (1 - t2_t3) * image_t2_1 + t2_t3 * image_t3_1

            if args.feature_extract:
                x_feat_t1_list = feature_model(image0_t1)
                x_feat_t2_list = feature_model(image_unknown_t2)
                x_feat_t3_list = feature_model(image1_t3)
                (
                    x_feat_t1_0_list,
                    x_feat_t2_0_list,
                    x_feat_t2_1_list,
                    x_feat_t3_1_list,
                ) = ([], [], [], [])

                for feat_idx in range(len(x_feat_t1_list)):
                    reg_model_feat = utils.register_model(
                        tuple([x // (2**feat_idx) for x in img_size])
                    )
                    x_feat_t1_0_list.append(
                        reg_model_feat(
                            [
                                x_feat_t1_list[feat_idx],
                                F.interpolate(
                                    flow_t1_0 * (0.5 ** (feat_idx)),
                                    scale_factor=0.5 ** (feat_idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_t2_0_list.append(
                        reg_model_feat(
                            [
                                x_feat_t2_list[feat_idx],
                                F.interpolate(
                                    flow_t2_0 * (0.5 ** (feat_idx)),
                                    scale_factor=0.5 ** (feat_idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_t2_1_list.append(
                        reg_model_feat(
                            [
                                x_feat_t2_list[feat_idx],
                                F.interpolate(
                                    flow_t2_1 * (0.5 ** (feat_idx)),
                                    scale_factor=0.5 ** (feat_idx),
                                ).float(),
                            ]
                        )
                    )
                    x_feat_t3_1_list.append(
                        reg_model_feat(
                            [
                                x_feat_t3_list[feat_idx],
                                F.interpolate(
                                    flow_t3_1 * (0.5 ** (feat_idx)),
                                    scale_factor=0.5 ** (feat_idx),
                                ).float(),
                            ]
                        )
                    )

                image0_out_diff = refinement_model(
                    image0_combined, x_feat_t1_0_list, x_feat_t2_0_list
                )
                image1_out_diff = refinement_model(
                    image1_combined, x_feat_t2_1_list, x_feat_t3_1_list
                )
                
            else:
                image0_out_diff = refinement_model(image0_combined)
                image1_out_diff = refinement_model(image1_combined)

            image0_out = image0_combined + image0_out_diff
            image1_out = image1_combined + image1_out_diff
            loss_diff_0 = criterion_l1n(image0_out_diff)
            loss_diff_1 = criterion_l1n(image1_out_diff)
            loss_diff = (loss_diff_0 + loss_diff_1) * args.weight_diff

            loss_cyc_ncc_0 = criterion_ncc(image0_out, image0) * args.weight_ncc
            loss_cyc_cha_0 = criterion_cha(image0_out, image0, eps=epsilon) * args.weight_cha
            loss_cyc_ncc_1 = criterion_ncc(image1_out, image1) * args.weight_ncc
            loss_cyc_cha_1 = criterion_cha(image1_out, image1, eps=epsilon) * args.weight_cha

            loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
            loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
            loss_cycle = (loss_cycle_0 + loss_cycle_1) * args.weight_cycle

            loss_diff_all.update(loss_diff_0.item(), image1.numel())
            loss_diff_all.update(loss_diff_1.item(), image1.numel())
            loss_all_cycle.update(loss_cycle_0.item(), image1.numel())
            loss_all_cycle.update(loss_cycle_1.item(), image1.numel())

            loss = flow_model_loss + loss_cycle + loss_diff

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), image0.numel())

        wandb.log({"Loss_all/train": loss_all.avg}, step=epoch)
        wandb.log({"Loss_full/train_all": loss_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_ncc": loss_ncc_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_cha": loss_cha_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_reg": loss_reg_all_full.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_all": loss_all_cycle.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_diff": loss_diff_all.avg}, step=epoch)

        """
        Validation
        """
        #if (epoch == 0) or ((epoch + 1) % 50 == 0):
        eval_ncc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                flow_model.eval()
                refinement_model.eval()
                data = [t.cuda() for t in data]
                image0 = data[0]
                image1 = data[1]

                image0_image1 = torch.cat((image0, image1), dim=1)
                source, target = image0_image1[:, 0:1], image0_image1[:, 1:2]
                
                _, _, flow_0_1, _ = flow_model(image0_image1)

                # y_source, y_target, flow_field = flow_model(source, target)
                # # The flow field will be full size and include both directions
                # flow_0_1 = flow_field[:, :3]  # first 3 channels
                # _ = -flow_field[:, :3]  # same magnitude, opposite direction
                # _ = y_source
                # _ = y_target

                image_0_1 = reg_model_bilin([image0, flow_0_1.float()])

                ncc = -1 * criterion_ncc(image_0_1, image1)
                eval_ncc.update(ncc.item(), image0.size(0))

        print("Epoch {}, NCC {:.5f}\n".format(epoch, eval_ncc.avg), flush=True)
        wandb.log({"Validate/NCC": eval_ncc.avg}, step=epoch)

        if eval_ncc.avg > best_ncc:
            best_ncc = eval_ncc.avg

        
        torch.save(
            {
                "epoch": epoch + 1,
                "flow_model_state_dict": flow_model.state_dict(),
                "model_state_dict": refinement_model.state_dict(),
                "feature_model_state_dict": feature_model.state_dict() if args.feature_extract else None,
                "best_ncc": best_ncc,
                "optimizer": optimizer.state_dict(),
            },
            "experiments/{}/epoch{}_ncc{:.4f}.ckpt".format(args.dataset, epoch + 1, eval_ncc.avg),
        )

            

        loss_all.reset()
        loss_all_full.reset()
        loss_ncc_all_full.reset()
        loss_cha_all_full.reset()
        loss_reg_all_full.reset()
        loss_all_cycle.reset()
        loss_diff_all.reset()

    print("best_ncc {}".format(best_ncc), flush=True)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Basic training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cardiac", choices=["cardiac", "lung"])

    # Model specific parameters
    parser.add_argument("--weight_cycle", type=float, default=1.0)
    parser.add_argument("--weight_diff", type=float, default=1.0)
    parser.add_argument("--weight_ncc", type=float, default=1.0)
    parser.add_argument("--weight_cha", type=float, default=1.0)
    parser.add_argument("--feature_extract", action="store_true", default=True)
    args = parser.parse_args()

    """
    GPU configuration
    """
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))

    main(args)
