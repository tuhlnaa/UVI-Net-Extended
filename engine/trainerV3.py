from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from utils.utils import AverageMeter, register_model


class Trainer:
    def __init__(
        self,
        flow_model: nn.Module,
        refinement_model: nn.Module,
        feature_model: Optional[nn.Module],
        optimizer: optim.Optimizer,
        criterion_ncc: nn.Module,
        criterion_cha: nn.Module,
        criterion_reg: nn.Module,
        criterion_l1n: nn.Module,
        reg_model: nn.Module,
        reg_model_bilin: nn.Module,
        img_size: Tuple[int, int, int],
        config: Dict,
    ):
        """Initialize the trainer with models, optimizer and loss functions.
        
        Args:
            flow_model: VoxelMorph model for flow field prediction
            refinement_model: UNet model for refinement
            feature_model: Optional feature extraction model
            optimizer: Optimizer for training
            criterion_*: Various loss functions
            reg_model*: Registration models
            img_size: Input image dimensions
            config: Training configuration/hyperparameters
        """
        self.flow_model = flow_model
        self.refinement_model = refinement_model
        self.feature_model = feature_model
        self.optimizer = optimizer
        self.criterion_ncc = criterion_ncc
        self.criterion_cha = criterion_cha
        self.criterion_reg = criterion_reg
        self.criterion_l1n = criterion_l1n
        self.reg_model = reg_model
        self.reg_model_bilin = reg_model_bilin
        self.img_size = img_size
        self.config = config
        self.epsilon = 1e-3

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dict containing loss metrics
        """
        meters = {
            'loss_all': AverageMeter(),
            'loss_all_full': AverageMeter(),
            'loss_ncc_all_full': AverageMeter(),
            'loss_cha_all_full': AverageMeter(), 
            'loss_reg_all_full': AverageMeter(),
            'loss_all_cycle': AverageMeter(),
            'loss_diff_all': AverageMeter()
        }

        for data in train_loader:
            self.refinement_model.train()
            self.flow_model.train()
            if self.feature_model:
                self.feature_model.train()

            loss = self._train_step(data, meters)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Log metrics
        metrics = {
            "Loss_all/train": meters['loss_all'].avg,
            "Loss_full/train_all": meters['loss_all_full'].avg,
            "Loss_full/train_img_ncc": meters['loss_ncc_all_full'].avg,
            "Loss_full/train_img_cha": meters['loss_cha_all_full'].avg,
            "Loss_full/train_reg": meters['loss_reg_all_full'].avg,
            "Loss_cycle/train_all": meters['loss_all_cycle'].avg,
            "Loss_cycle/train_diff": meters['loss_diff_all'].avg
        }
        wandb.log(metrics, step=epoch)
        
        return metrics

    def _train_step(self, data: list, meters: Dict[str, AverageMeter]) -> torch.Tensor:
        """Perform a single training step.

        Args:
            data: List of input tensors
            meters: Dict of AverageMeters for tracking losses

        Returns:
            Total loss for the step
        """
        data = [t.cuda() for t in data]
        i0, i1 = data[0], data[1]

        # Initial flow prediction
        i0_i1 = torch.cat((i0, i1), dim=1)
        i_0_1, i_1_0, flow_0_1, flow_1_0 = self.flow_model(i0_i1)

        # Calculate full losses
        loss_full = self._compute_full_loss(i0, i1, i_0_1, i_1_0, flow_0_1, flow_1_0, meters)
        
        if self.config.weight_cycle == 0:
            meters['loss_all'].update(loss_full.item(), i0.numel())
            return loss_full

        # Random interpolation points
        alphas = self._generate_alphas()
        
        # Compute intermediate frames and flows
        interp_results = self._compute_interpolations(i0, i1, flow_0_1, flow_1_0, alphas)
        
        # Refinement and cycle consistency
        loss_cycle, loss_diff = self._compute_cycle_loss(i0, i1, interp_results, meters)

        total_loss = loss_full + loss_cycle + loss_diff
        meters['loss_all'].update(total_loss.item(), i0.numel())
        
        return total_loss

    def _compute_full_loss(self, i0, i1, i_0_1, i_1_0, flow_0_1, flow_1_0, meters):
        """Compute the full registration loss."""
        # Forward direction
        loss_ncc_1 = self.criterion_ncc(i_0_1, i1) * self.config.weight_ncc
        loss_cha_1 = self.criterion_cha(i_0_1, i1, eps=self.epsilon) * self.config.weight_cha
        loss_reg_1 = self.criterion_reg(flow_0_1, None)
        
        # Backward direction
        loss_ncc_0 = self.criterion_ncc(i_1_0, i0) * self.config.weight_ncc
        loss_cha_0 = self.criterion_cha(i_1_0, i0, eps=self.epsilon) * self.config.weight_cha
        loss_reg_0 = self.criterion_reg(flow_1_0, None)

        loss_full = loss_ncc_1 + loss_cha_1 + loss_reg_1 + loss_ncc_0 + loss_cha_0 + loss_reg_0

        # Update meters
        self._update_full_loss_meters(meters, loss_full, loss_ncc_1, loss_cha_1, loss_reg_1,
                                    loss_ncc_0, loss_cha_0, loss_reg_0, i1.numel())
        
        return loss_full

    def _update_full_loss_meters(self, meters, loss_full, loss_ncc_1, loss_cha_1, loss_reg_1,
                            loss_ncc_0, loss_cha_0, loss_reg_0, numel):
        """Update loss meters for full registration loss."""
        meters['loss_all_full'].update(loss_full.item(), numel)
        meters['loss_ncc_all_full'].update(loss_ncc_1.item(), numel)
        meters['loss_cha_all_full'].update(loss_cha_1.item(), numel)
        meters['loss_reg_all_full'].update(loss_reg_1.item(), numel)
        meters['loss_ncc_all_full'].update(loss_ncc_0.item(), numel)
        meters['loss_cha_all_full'].update(loss_cha_0.item(), numel)
        meters['loss_reg_all_full'].update(loss_reg_0.item(), numel)
        

    @staticmethod
    def _generate_alphas() -> Tuple[float, float, float]:
        """Generate random interpolation points."""
        return (
            torch.FloatTensor(1).uniform_(-0.5, 0.0).item(),
            torch.FloatTensor(1).uniform_(0.0, 1.0).item(),
            torch.FloatTensor(1).uniform_(1.0, 1.5).item()
        )

    def _compute_interpolations(self, i0, i1, flow_0_1, flow_1_0, alphas):
        """Compute interpolated frames and flows."""
        alpha1, alpha2, alpha3 = alphas
        
        # First interpolation
        flow_0_a1 = flow_0_1 * alpha1
        i_0_a1 = self.reg_model_bilin([i0, flow_0_a1.float()])

        if alpha2 < 0.5:
            flow_0_a2 = flow_0_1 * alpha2
            i_unknown_a2 = self.reg_model_bilin([i0, flow_0_a2.float()])
        else:
            flow_1_a2 = flow_1_0 * (1 - alpha2)
            i_unknown_a2 = self.reg_model_bilin([i1, flow_1_a2.float()])

        flow_1_a3 = flow_1_0 * (1 - alpha3)
        i_1_a3 = self.reg_model_bilin([i1, flow_1_a3.float()])

        return {
            'i_0_a1': i_0_a1,
            'i_unknown_a2': i_unknown_a2,
            'i_1_a3': i_1_a3,
            'alpha1': alpha1,
            'alpha2': alpha2,
            'alpha3': alpha3
        }

    def _compute_cycle_loss(self, i0, i1, interp_results, meters):
        """Compute cycle consistency and refinement losses.
        
        Args:
            i0, i1: Source and target images
            interp_results: Dict containing interpolation results
            meters: Dict of AverageMeters for tracking losses
            
        Returns:
            Tuple of (cycle loss, difference loss)
        """
        i_0_a1 = interp_results['i_0_a1']
        i_unknown_a2 = interp_results['i_unknown_a2']
        i_1_a3 = interp_results['i_1_a3']
        
        # Second stage flow computation
        ia1_ia2 = torch.cat((i_0_a1, i_unknown_a2), dim=1)
        ia2_ia3 = torch.cat((i_unknown_a2, i_1_a3), dim=1)
        
        # Get flows between intermediate frames
        _, _, flow_a1_a2, flow_a2_a1 = self.flow_model(ia1_ia2)
        _, _, flow_a2_a3, flow_a3_a2 = self.flow_model(ia2_ia3)

        # Compute interpolation coefficients
        alpha12 = (0 - interp_results['alpha1']) / (interp_results['alpha2'] - interp_results['alpha1'])
        alpha23 = (1 - interp_results['alpha2']) / (interp_results['alpha3'] - interp_results['alpha2'])

        # Scale flows by interpolation coefficients
        flow_a1_0 = flow_a1_a2 * alpha12
        flow_a2_0 = flow_a2_a1 * (1 - alpha12)
        flow_a2_1 = flow_a2_a3 * alpha23
        flow_a3_1 = flow_a3_a2 * (1 - alpha23)

        # Apply flows to get warped images
        i_a1_0 = self.reg_model_bilin([i_0_a1, flow_a1_0.float()])
        i_a2_0 = self.reg_model_bilin([i_unknown_a2, flow_a2_0.float()])
        i_a2_1 = self.reg_model_bilin([i_unknown_a2, flow_a2_1.float()])
        i_a3_1 = self.reg_model_bilin([i_1_a3, flow_a3_1.float()])

        # Combine warped images
        i0_combined = (1 - alpha12) * i_a1_0 + alpha12 * i_a2_0
        i1_combined = (1 - alpha23) * i_a2_1 + alpha23 * i_a3_1

        # Apply refinement
        if self.feature_model is not None:
            i0_out_diff, i1_out_diff = self._apply_feature_refinement(
                i0_combined, i1_combined,
                i_0_a1, i_unknown_a2, i_1_a3,
                flow_a1_0, flow_a2_0, flow_a2_1, flow_a3_1
            )
        else:
            i0_out_diff = self.refinement_model(i0_combined)
            i1_out_diff = self.refinement_model(i1_combined)

        # Compute refinement outputs
        i0_out = i0_combined + i0_out_diff
        i1_out = i1_combined + i1_out_diff

        # Compute losses
        loss_diff_0 = self.criterion_l1n(i0_out_diff)
        loss_diff_1 = self.criterion_l1n(i1_out_diff)
        loss_diff = (loss_diff_0 + loss_diff_1) * self.config.weight_diff

        loss_cyc_ncc_0 = self.criterion_ncc(i0_out, i0) * self.config.weight_ncc
        loss_cyc_cha_0 = self.criterion_cha(i0_out, i0, eps=self.epsilon) * self.config.weight_cha
        loss_cyc_ncc_1 = self.criterion_ncc(i1_out, i1) * self.config.weight_ncc
        loss_cyc_cha_1 = self.criterion_cha(i1_out, i1, eps=self.epsilon) * self.config.weight_cha

        loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
        loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
        loss_cycle = (loss_cycle_0 + loss_cycle_1) * self.config.weight_cycle

        # Update meters
        meters['loss_diff_all'].update(loss_diff_0.item(), i1.numel())
        meters['loss_diff_all'].update(loss_diff_1.item(), i1.numel())
        meters['loss_all_cycle'].update(loss_cycle_0.item(), i1.numel())
        meters['loss_all_cycle'].update(loss_cycle_1.item(), i1.numel())

        return loss_cycle, loss_diff

    def _apply_feature_refinement(self, i0_combined, i1_combined, 
                                i_0_a1, i_unknown_a2, i_1_a3,
                                flow_a1_0, flow_a2_0, flow_a2_1, flow_a3_1):
        """Apply feature-based refinement.
        
        Args:
            i0_combined, i1_combined: Combined intermediate images
            i_0_a1, i_unknown_a2, i_1_a3: Interpolated images
            flow_*: Flow fields between frames
            
        Returns:
            Tuple of refinement differences for both directions
        """
        # Extract features for each frame
        x_feat_a1_list = self.feature_model(i_0_a1)
        x_feat_a2_list = self.feature_model(i_unknown_a2)
        x_feat_a3_list = self.feature_model(i_1_a3)

        # Lists to store warped features
        x_feat_a1_0_list, x_feat_a2_0_list = [], []
        x_feat_a2_1_list, x_feat_a3_1_list = [], []

        # Warp features at each scale
        for feat_idx, feats in enumerate(zip(x_feat_a1_list, x_feat_a2_list, x_feat_a3_list)):
            feat_a1, feat_a2, feat_a3 = feats
            scale_factor = 2 ** feat_idx
            scaled_size = tuple([x // scale_factor for x in self.img_size])
            
            reg_model_feat = register_model(scaled_size)
            flow_scale = 0.5 ** feat_idx

            # Scale and apply flows
            def scale_and_warp(feat, flow):
                scaled_flow = F.interpolate(
                    flow * flow_scale,
                    scale_factor=1/scale_factor
                ).float()
                return reg_model_feat([feat, scaled_flow])

            x_feat_a1_0_list.append(scale_and_warp(feat_a1, flow_a1_0))
            x_feat_a2_0_list.append(scale_and_warp(feat_a2, flow_a2_0))
            x_feat_a2_1_list.append(scale_and_warp(feat_a2, flow_a2_1))
            x_feat_a3_1_list.append(scale_and_warp(feat_a3, flow_a3_1))

        # Apply refinement with features
        i0_out_diff = self.refinement_model(i0_combined, x_feat_a1_0_list, x_feat_a2_0_list)
        i1_out_diff = self.refinement_model(i1_combined, x_feat_a2_1_list, x_feat_a3_1_list)

        return i0_out_diff, i1_out_diff