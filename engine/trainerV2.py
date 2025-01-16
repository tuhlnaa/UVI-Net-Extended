import torch
import torch.nn.functional as F
import wandb
from tqdm import trange
import random

from utils import utils
import losses

class Trainer:
    def __init__(self, flow_model, refinement_model, feature_model, reg_model, reg_model_bilin, 
                 optimizer, args, img_size):
        self.flow_model = flow_model
        self.refinement_model = refinement_model
        self.feature_model = feature_model
        self.reg_model = reg_model
        self.reg_model_bilin = reg_model_bilin
        self.optimizer = optimizer
        self.args = args
        self.img_size = img_size
        
        self.criterion_ncc = losses.NCC()
        self.criterion_cha = losses.CharbonnierLoss
        self.criterion_reg = losses.Grad3d(penalty="l2")
        self.criterion_l1n = losses.L1_norm()
        self.epsilon = 1e-3
        

    def train_epoch(self, train_loader, epoch):
        loss_all = utils.AverageMeter()
        loss_all_full = utils.AverageMeter()
        loss_ncc_all_full = utils.AverageMeter()
        loss_cha_all_full = utils.AverageMeter()
        loss_reg_all_full = utils.AverageMeter()
        loss_all_cycle = utils.AverageMeter()
        loss_diff_all = utils.AverageMeter()
        
        for idx, data in enumerate(train_loader):
            self.refinement_model.train()
            self.flow_model.train()
            
            loss = self._train_step(data, loss_all_full, loss_ncc_all_full, 
                                  loss_cha_all_full, loss_reg_all_full, 
                                  loss_all_cycle, loss_diff_all)
            
            loss_all.update(loss.item(), data[0].numel())
            
        self._log_training_metrics(epoch, loss_all, loss_all_full, loss_ncc_all_full,
                                 loss_cha_all_full, loss_reg_all_full, loss_all_cycle,
                                 loss_diff_all)
        
        return loss_all
        

    def validate(self, val_loader):
        eval_ncc = utils.AverageMeter()
        
        with torch.no_grad():
            for data in val_loader:
                self.flow_model.eval()
                self.refinement_model.eval()
                
                ncc = self._validate_step(data)
                eval_ncc.update(ncc.item(), data[0].size(0))
                
        return eval_ncc
    

    def _train_step(self, data, loss_all_full, loss_ncc_all_full, loss_cha_all_full,
                   loss_reg_all_full, loss_all_cycle, loss_diff_all):
        data = [t.cuda() for t in data]
        i0, i1 = data[0], data[1]
        
        # Generate random interpolation points
        alpha1 = random.uniform(-0.5, 0.0)
        alpha2 = random.uniform(0.0, 1.0)
        alpha3 = random.uniform(1.0, 1.5)
        
        # First registration pass
        loss_full = self._compute_registration_loss(i0, i1, loss_all_full, loss_ncc_all_full,
                                                  loss_cha_all_full, loss_reg_all_full)
        
        if self.args.weight_cycle == 0:
            self.optimizer.zero_grad()
            loss_full.backward()
            self.optimizer.step()
            return loss_full
            
        # Compute cycle consistency loss
        loss_cycle, loss_diff = self._compute_cycle_loss(i0, i1, alpha1, alpha2, alpha3,
                                                       loss_all_cycle, loss_diff_all)
        
        loss = loss_full + loss_cycle + loss_diff
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    

    def _validate_step(self, data):
        i0, i1 = data[0].cuda(), data[1].cuda()
        i0_i1 = torch.cat((i0, i1), dim=1)
        source, target = i0_i1[:, 0:1], i0_i1[:, 1:2]
        
        # _, _, flow_field = self.flow_model(source, target)
        # flow_0_1 = flow_field[:, :3]
        
        _, _, flow_0_1, flow_1_0 = self.flow_model(i0_i1)

        i_0_1 = self.reg_model_bilin([i0, flow_0_1.float()])
        return -1 * self.criterion_ncc(i_0_1, i1)
    

    # Helper methods for loss computation
    def _compute_registration_loss(self, i0, i1, loss_all_full, loss_ncc_all_full,
                                 loss_cha_all_full, loss_reg_all_full):
        i0_i1 = torch.cat((i0, i1), dim=1)
        source, target = i0_i1[:, 0:1], i0_i1[:, 1:2]
        
        # y_source, y_target, flow_field = self.flow_model(source, target)
        # flow_0_1 = flow_field[:, :3]
        # flow_1_0 = -flow_field[:, :3]
        # i_0_1 = y_source
        # i_1_0 = y_target

        i_0_1, i_1_0, flow_0_1, flow_1_0 = self.flow_model(i0_i1) 

        # Compute losses
        loss_ncc_1 = self.criterion_ncc(i_0_1, i1) * self.args.weight_ncc
        loss_cha_1 = self.criterion_cha(i_0_1, i1, eps=self.epsilon) * self.args.weight_cha
        loss_reg_1 = self.criterion_reg(flow_0_1, None)
        loss_ncc_0 = self.criterion_ncc(i_1_0, i0) * self.args.weight_ncc
        loss_cha_0 = self.criterion_cha(i_1_0, i0, eps=self.epsilon) * self.args.weight_cha
        loss_reg_0 = self.criterion_reg(flow_1_0, None)
        
        loss_full = (loss_ncc_1 + loss_cha_1 + loss_reg_1 + 
                    loss_ncc_0 + loss_cha_0 + loss_reg_0)
        
        # Update meters
        self._update_registration_meters(loss_full, loss_ncc_1, loss_cha_1, loss_reg_1,
                                      loss_ncc_0, loss_cha_0, loss_reg_0, i1.numel(),
                                      loss_all_full, loss_ncc_all_full, loss_cha_all_full,
                                      loss_reg_all_full)
        
        return loss_full
    

    def _compute_cycle_loss(self, i0, i1, alpha1, alpha2, alpha3, loss_all_cycle, loss_diff_all):
        # First Interpolation
        flow_0_a1 = self._get_flow_for_frame(i0, i1, alpha1)
        i_0_a1 = self.reg_model_bilin([i0, flow_0_a1.float()])

        if alpha2 < 0.5:
            flow_0_a2 = self._get_flow_for_frame(i0, i1, alpha2)
            i_0_a2 = self.reg_model_bilin([i0, flow_0_a2.float()])
            i_unknown_a2 = i_0_a2
        else:
            flow_1_a2 = self._get_flow_for_frame(i1, i0, 1 - alpha2)
            i_1_a2 = self.reg_model_bilin([i1, flow_1_a2.float()])
            i_unknown_a2 = i_1_a2

        flow_1_a3 = self._get_flow_for_frame(i1, i0, 1 - alpha3)
        i_1_a3 = self.reg_model_bilin([i1, flow_1_a3.float()])

        # Second Interpolation
        ia1_ia2 = torch.cat((i_0_a1, i_unknown_a2), dim=1)
        ia2_ia3 = torch.cat((i_unknown_a2, i_1_a3), dim=1)

        # Get flows between intermediate frames
        # _, _, flow_field_a1_a2 = self.flow_model(ia1_ia2[:, 0:1], ia1_ia2[:, 1:2])
        # flow_a1_a2 = flow_field_a1_a2[:, :3]
        # flow_a2_a1 = -flow_field_a1_a2[:, :3]

        # _, _, flow_field_a2_a3 = self.flow_model(ia2_ia3[:, 0:1], ia2_ia3[:, 1:2])
        # flow_a2_a3 = flow_field_a2_a3[:, :3]
        # flow_a3_a2 = -flow_field_a2_a3[:, :3]

        _, _, flow_a1_a2, flow_a2_a1 = self.flow_model(ia1_ia2)
        _, _, flow_a2_a3, flow_a3_a2 = self.flow_model(ia2_ia3)

        # Calculate interpolation weights
        alpha12 = (0 - alpha1) / (alpha2 - alpha1)
        alpha23 = (1 - alpha2) / (alpha3 - alpha2)

        # Calculate flows to original frames
        flow_a1_0 = flow_a1_a2 * alpha12
        flow_a2_0 = flow_a2_a1 * (1 - alpha12)
        flow_a2_1 = flow_a2_a3 * alpha23
        flow_a3_1 = flow_a3_a2 * (1 - alpha23)

        # Interpolate back to original frames
        i_a1_0 = self.reg_model_bilin([i_0_a1, flow_a1_0.float()])
        i_a2_0 = self.reg_model_bilin([i_unknown_a2, flow_a2_0.float()])
        i_a2_1 = self.reg_model_bilin([i_unknown_a2, flow_a2_1.float()])
        i_a3_1 = self.reg_model_bilin([i_1_a3, flow_a3_1.float()])

        # Combine interpolations
        i0_combined = (1 - alpha12) * i_a1_0 + alpha12 * i_a2_0
        i1_combined = (1 - alpha23) * i_a2_1 + alpha23 * i_a3_1

        # Apply refinement with feature extraction if enabled
        if self.args.feature_extract:
            i0_out, i1_out, i0_out_diff, i1_out_diff = self._apply_feature_refinement(
                i0_combined, i1_combined, i_0_a1, i_unknown_a2, i_1_a3,
                flow_a1_0, flow_a2_0, flow_a2_1, flow_a3_1
            )
        else:
            i0_out_diff = self.refinement_model(i0_combined)
            i1_out_diff = self.refinement_model(i1_combined)
            i0_out = i0_combined + i0_out_diff
            i1_out = i1_combined + i1_out_diff

        # Calculate losses
        loss_diff_0 = self.criterion_l1n(i0_out_diff)
        loss_diff_1 = self.criterion_l1n(i1_out_diff)
        loss_diff = (loss_diff_0 + loss_diff_1) * self.args.weight_diff

        loss_cyc_ncc_0 = self.criterion_ncc(i0_out, i0) * self.args.weight_ncc
        loss_cyc_cha_0 = self.criterion_cha(i0_out, i0, eps=self.epsilon) * self.args.weight_cha
        loss_cyc_ncc_1 = self.criterion_ncc(i1_out, i1) * self.args.weight_ncc
        loss_cyc_cha_1 = self.criterion_cha(i1_out, i1, eps=self.epsilon) * self.args.weight_cha

        loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
        loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
        loss_cycle = (loss_cycle_0 + loss_cycle_1) * self.args.weight_cycle

        # Update loss meters
        loss_diff_all.update(loss_diff_0.item(), i1.numel())
        loss_diff_all.update(loss_diff_1.item(), i1.numel())
        loss_all_cycle.update(loss_cycle_0.item(), i1.numel())
        loss_all_cycle.update(loss_cycle_1.item(), i1.numel())

        return loss_cycle, loss_diff


    def _get_flow_for_frame(self, source, target, alpha):
        combined = torch.cat((source, target), dim=1)
        # _, _, flow_field = self.flow_model(combined[:, 0:1], combined[:, 1:2])
        # return flow_field[:, :3] * alpha
    
        _, _, flow_0_1, flow_1_0 = self.flow_model(combined)
        return flow_0_1 * alpha


    def _apply_feature_refinement(self, i0_combined, i1_combined, i_0_a1, i_unknown_a2, i_1_a3,
                                flow_a1_0, flow_a2_0, flow_a2_1, flow_a3_1):
        # Extract features
        x_feat_a1_list = self.feature_model(i_0_a1)
        x_feat_a2_list = self.feature_model(i_unknown_a2)
        x_feat_a3_list = self.feature_model(i_1_a3)

        # Initialize feature lists
        x_feat_a1_0_list, x_feat_a2_0_list = [], []
        x_feat_a2_1_list, x_feat_a3_1_list = [], []

        # Process features at each scale
        for feat_idx in range(len(x_feat_a1_list)):
            scale_factor = 2 ** feat_idx
            img_size_at_scale = tuple([x // scale_factor for x in self.img_size])
            reg_model_feat = utils.register_model(img_size_at_scale)

            # Transform features
            x_feat_a1_0_list.append(
                reg_model_feat([
                    x_feat_a1_list[feat_idx],
                    F.interpolate(
                        flow_a1_0 * (0.5 ** feat_idx),
                        scale_factor=0.5 ** feat_idx
                    ).float()
                ])
            )
            x_feat_a2_0_list.append(
                reg_model_feat([
                    x_feat_a2_list[feat_idx],
                    F.interpolate(
                        flow_a2_0 * (0.5 ** feat_idx),
                        scale_factor=0.5 ** feat_idx
                    ).float()
                ])
            )
            x_feat_a2_1_list.append(
                reg_model_feat([
                    x_feat_a2_list[feat_idx],
                    F.interpolate(
                        flow_a2_1 * (0.5 ** feat_idx),
                        scale_factor=0.5 ** feat_idx
                    ).float()
                ])
            )
            x_feat_a3_1_list.append(
                reg_model_feat([
                    x_feat_a3_list[feat_idx],
                    F.interpolate(
                        flow_a3_1 * (0.5 ** feat_idx),
                        scale_factor=0.5 ** feat_idx
                    ).float()
                ])
            )

        # Apply refinement with features
        i0_out_diff = self.refinement_model(i0_combined, x_feat_a1_0_list, x_feat_a2_0_list)
        i1_out_diff = self.refinement_model(i1_combined, x_feat_a2_1_list, x_feat_a3_1_list)

        return (i0_combined + i0_out_diff, i1_combined + i1_out_diff,
                i0_out_diff, i1_out_diff)
    

    def _update_registration_meters(self, loss_full, loss_ncc_1, loss_cha_1, loss_reg_1,
                                  loss_ncc_0, loss_cha_0, loss_reg_0, numel,
                                  loss_all_full, loss_ncc_all_full, loss_cha_all_full,
                                  loss_reg_all_full):
        loss_all_full.update(loss_full.item(), numel)
        loss_ncc_all_full.update(loss_ncc_1.item(), numel)
        loss_cha_all_full.update(loss_cha_1.item(), numel)
        loss_reg_all_full.update(loss_reg_1.item(), numel)
        loss_ncc_all_full.update(loss_ncc_0.item(), numel)
        loss_cha_all_full.update(loss_cha_0.item(), numel)
        loss_reg_all_full.update(loss_reg_0.item(), numel)
    

    def _log_training_metrics(self, epoch, loss_all, loss_all_full, loss_ncc_all_full,
                            loss_cha_all_full, loss_reg_all_full, loss_all_cycle,
                            loss_diff_all):
        wandb.log({"Loss_all/train": loss_all.avg}, step=epoch)
        wandb.log({"Loss_full/train_all": loss_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_ncc": loss_ncc_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_img_cha": loss_cha_all_full.avg}, step=epoch)
        wandb.log({"Loss_full/train_reg": loss_reg_all_full.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_all": loss_all_cycle.avg}, step=epoch)
        wandb.log({"Loss_cycle/train_diff": loss_diff_all.avg}, step=epoch)