import random
import wandb
import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from utils import utils
from utils import losses

class Trainer:
    """Trainer class for managing the training process of the medical image interpolation model."""
    def __init__(
        self,
        flow_model: nn.Module,
        refinement_model: nn.Module,
        feature_model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        reg_model: nn.Module,
        reg_model_bilin: nn.Module,
        criterion_ncc: nn.Module,
        criterion_cha: nn.Module,
        criterion_reg: nn.Module,
        criterion_l1n: nn.Module,
        config: Dict,
        img_size: Tuple[int, int, int],
        device: torch.device,
        ):
        """Initialize trainer with models, optimizer and loss functions."""
        self.flow_model = flow_model
        self.refinement_model = refinement_model
        self.feature_model = feature_model
        self.optimizer = optimizer
        self.reg_model = reg_model
        self.reg_model_bilin = reg_model_bilin
        self.criterion_ncc = criterion_ncc
        self.criterion_cha = criterion_cha
        self.criterion_reg = criterion_reg
        self.criterion_l1n = criterion_l1n
        self.config = config
        self.img_size = img_size
        self.device = device
        self.epsilon = 1e-3


    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        meters = {
            'loss_all': utils.AverageMeter(),
            'loss_all_full': utils.AverageMeter(),
            'loss_ncc_all_full': utils.AverageMeter(),
            'loss_cha_all_full': utils.AverageMeter(),
            'loss_reg_all_full': utils.AverageMeter(),
            'loss_all_cycle': utils.AverageMeter(),
            'loss_diff_all': utils.AverageMeter()
        }

        self.flow_model.train()
        self.refinement_model.train()
        if self.feature_model is not None:
            self.feature_model.train()

        for data in train_loader:
            data = [t.to(self.device) for t in data]
            image0, image1 = data[0], data[1]

            # Generate random interpolation points
            alphas = self._generate_interpolation_points()
            
            # Calculate initial flows between images
            loss_dict = self._compute_flow_loss(image0, image1)
            flow_model_loss = loss_dict['flow_model_loss']
            
            # Update meters for flow losses
            self._update_flow_meters(meters, loss_dict, image0, image1)

            if self.config.weight_cycle == 0:
                self._optimize_step(flow_model_loss)
                meters['loss_all'].update(flow_model_loss.item(), image0.numel())
                continue

            # Compute cycle consistency losses
            cycle_loss_dict = self._compute_cycle_loss(image0, image1, alphas)
            
            # Compute final loss and optimize
            total_loss = (flow_model_loss + cycle_loss_dict['loss_cycle'] + cycle_loss_dict['loss_diff'])
            self._optimize_step(total_loss)
            
            # Update remaining meters
            meters['loss_all'].update(total_loss.item(), image0.numel())
            self._update_cycle_meters(meters, cycle_loss_dict, image1)

        # Log metrics
        self._log_metrics(meters, epoch)
        
        return {name: meter.avg for name, meter in meters.items()}


    def _generate_interpolation_points(self) -> Tuple[float, float, float]:
        """Generate random interpolation points."""
        return (
            random.uniform(-0.5, 0.0),  # alpha1
            random.uniform(0.0, 1.0),   # alpha2
            random.uniform(1.0, 1.5)    # alpha3
        )


    def _compute_flow_loss(self, image0: torch.Tensor, image1: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute losses for the flow model."""
        image0_image1 = torch.cat((image0, image1), dim=1)
        image_0_1, image_1_0, flow_0_1, flow_1_0 = self.flow_model(image0_image1)

        loss_ncc_1 = self.criterion_ncc(image_0_1, image1) * self.config.weight_ncc
        loss_cha_1 = self.criterion_cha(image_0_1, image1, eps=self.epsilon) * self.config.weight_cha
        loss_reg_1 = self.criterion_reg(flow_0_1, None)
        
        loss_ncc_0 = self.criterion_ncc(image_1_0, image0) * self.config.weight_ncc
        loss_cha_0 = self.criterion_cha(image_1_0, image0, eps=self.epsilon) * self.config.weight_cha
        loss_reg_0 = self.criterion_reg(flow_1_0, None)

        flow_model_loss = (loss_ncc_1 + loss_cha_1 + loss_reg_1 + 
                          loss_ncc_0 + loss_cha_0 + loss_reg_0)

        return {
            'flow_model_loss': flow_model_loss,
            'loss_ncc_1': loss_ncc_1,
            'loss_cha_1': loss_cha_1,
            'loss_reg_1': loss_reg_1,
            'loss_ncc_0': loss_ncc_0,
            'loss_cha_0': loss_cha_0,
            'loss_reg_0': loss_reg_0,
            'flows': (flow_0_1, flow_1_0)
        }


    def _compute_cycle_loss(self, image0: torch.Tensor, image1: torch.Tensor, alphas: Tuple[float, float, float]) -> Dict[str, torch.Tensor]:
        """Compute cycle consistency losses."""
        alpha1, alpha2, alpha3 = alphas
        
        # First interpolation
        interpolated_images = self._compute_first_interpolation(
            image0, image1, alpha1, alpha2, alpha3
        )
        
        # Second interpolation
        refined_outputs = self._compute_second_interpolation(
            interpolated_images, image0, image1, alpha1, alpha2, alpha3
        )
        
        # Compute losses
        loss_diff = (refined_outputs['loss_diff_0'] + refined_outputs['loss_diff_1']) * self.config.weight_diff
        loss_cycle = (refined_outputs['loss_cycle_0'] + refined_outputs['loss_cycle_1']) * self.config.weight_cycle

        return {
            'loss_cycle': loss_cycle,
            'loss_diff': loss_diff,
            'loss_diff_0': refined_outputs['loss_diff_0'],
            'loss_diff_1': refined_outputs['loss_diff_1'],
            'loss_cycle_0': refined_outputs['loss_cycle_0'],
            'loss_cycle_1': refined_outputs['loss_cycle_1']
        }


    def _compute_first_interpolation(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        ) -> Dict[str, torch.Tensor]:
        """
        Compute the first interpolation step between image pairs.
        
        Args:
            image0: Source image tensor
            image1: Target image tensor
            alpha1: First interpolation point (< 0)
            alpha2: Second interpolation point (between 0 and 1)
            alpha3: Third interpolation point (> 1)
            
        Returns:
            Dictionary containing interpolated images and flow fields
        """
        # Compute initial flows between images
        image0_image1 = torch.cat((image0, image1), dim=1)
        _, _, flow_0_1, flow_1_0 = self.flow_model(image0_image1)
        
        # Compute first timepoint (t1)
        flow0_t1 = flow_0_1 * alpha1
        image0_t1 = self.reg_model_bilin([image0, flow0_t1.float()])
        
        # Compute second timepoint (t2)
        if alpha2 < 0.5:
            flow0_t2 = flow_0_1 * alpha2
            image0_t2 = self.reg_model_bilin([image0, flow0_t2.float()])
            image_unknown_t2 = image0_t2
        else:
            flow1_t2 = flow_1_0 * (1 - alpha2)
            image1_t2 = self.reg_model_bilin([image1, flow1_t2.float()])
            image_unknown_t2 = image1_t2
        
        # Compute third timepoint (t3)
        flow1_t3 = flow_1_0 * (1 - alpha3)
        image1_t3 = self.reg_model_bilin([image1, flow1_t3.float()])
        
        return {
            'image0_t1': image0_t1,
            'image_unknown_t2': image_unknown_t2,
            'image1_t3': image1_t3,
            'flows': (flow_0_1, flow_1_0)
        }


    def _compute_second_interpolation(
        self,
        interpolated_images: Dict[str, torch.Tensor],
        image0: torch.Tensor,
        image1: torch.Tensor,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        ) -> Dict[str, torch.Tensor]:
        """
        Compute the second interpolation step and associated losses.
        
        Args:
            interpolated_images: Dictionary containing first interpolation results
            image0: Original source image
            image1: Original target image
            
        Returns:
            Dictionary containing refined outputs and associated losses
        """
        image0_t1 = interpolated_images['image0_t1']
        image_unknown_t2 = interpolated_images['image_unknown_t2']
        image1_t3 = interpolated_images['image1_t3']
        
        # Compute flows between t1 and t2
        image_t1_image_t2 = torch.cat((image0_t1, image_unknown_t2), dim=1)
        _, _, flow_t1_t2, flow_t2_t1 = self.flow_model(image_t1_image_t2)
        
        # Compute flows between t2 and t3
        image_t2_image_t3 = torch.cat((image_unknown_t2, image1_t3), dim=1)
        _, _, flow_t2_t3, flow_t3_t2 = self.flow_model(image_t2_image_t3)
        
        # Calculate temporal weights
        t1_t2 = (0 - alpha1) / (alpha2 - alpha1)  # max=1, min=0
        t2_t3 = (1 - alpha2) / (alpha3 - alpha2)  # max=2, min=0
    
        # Compute flows to original timepoints
        flow_t1_0 = flow_t1_t2 * t1_t2
        flow_t2_0 = flow_t2_t1 * (1 - t1_t2)
        flow_t2_1 = flow_t2_t3 * t2_t3
        flow_t3_1 = flow_t3_t2 * (1 - t2_t3)
        
        # Warp images to original timepoints
        image_t1_0 = self.reg_model_bilin([image0_t1, flow_t1_0.float()])
        image_t2_0 = self.reg_model_bilin([image_unknown_t2, flow_t2_0.float()])
        image_t2_1 = self.reg_model_bilin([image_unknown_t2, flow_t2_1.float()])
        image_t3_1 = self.reg_model_bilin([image1_t3, flow_t3_1.float()])
        
        # Combine warped images
        image0_combined = (1 - t1_t2) * image_t1_0 + t1_t2 * image_t2_0
        image1_combined = (1 - t2_t3) * image_t2_1 + t2_t3 * image_t3_1
        
        # Apply refinement
        if self.feature_model is not None:
            # Extract features
            x_feat_t1_list = self.feature_model(image0_t1)
            x_feat_t2_list = self.feature_model(image_unknown_t2)
            x_feat_t3_list = self.feature_model(image1_t3)
            
            # Initialize feature lists
            x_feat_t1_0_list, x_feat_t2_0_list = [], []
            x_feat_t2_1_list, x_feat_t3_1_list = [], []
            
            # Compute features at each scale
            for feat_idx in range(len(x_feat_t1_list)):
                scale_factor = 2 ** feat_idx
                reg_model_feat = utils.register_model(tuple([x // scale_factor for x in self.img_size])).to(self.device)
                
                # Warp features for first image
                x_feat_t1_0_list.append(
                    reg_model_feat([x_feat_t1_list[feat_idx],
                                    F.interpolate(flow_t1_0 * (0.5 ** feat_idx), scale_factor=0.5 ** feat_idx).float()])
                )
                x_feat_t2_0_list.append(
                    reg_model_feat([x_feat_t2_list[feat_idx],
                                    F.interpolate(flow_t2_0 * (0.5 ** feat_idx), scale_factor=0.5 ** feat_idx).float()])
                )
                
                # Warp features for second image
                x_feat_t2_1_list.append(
                    reg_model_feat([x_feat_t2_list[feat_idx],
                                    F.interpolate(flow_t2_1 * (0.5 ** feat_idx), scale_factor=0.5 ** feat_idx).float()])
                )
                x_feat_t3_1_list.append(
                    reg_model_feat([x_feat_t3_list[feat_idx],
                                    F.interpolate(flow_t3_1 * (0.5 ** feat_idx), scale_factor=0.5 ** feat_idx).float()])
                )
            
            # Apply refinement with features
            image0_out_diff = self.refinement_model(image0_combined, x_feat_t1_0_list, x_feat_t2_0_list)
            image1_out_diff = self.refinement_model(image1_combined, x_feat_t2_1_list, x_feat_t3_1_list)

        else:
            # Apply refinement without features
            image0_out_diff = self.refinement_model(image0_combined)
            image1_out_diff = self.refinement_model(image1_combined)
        
        # Compute final outputs
        image0_out = image0_combined + image0_out_diff
        image1_out = image1_combined + image1_out_diff
        
        # Compute losses
        loss_diff_0 = self.criterion_l1n(image0_out_diff)
        loss_diff_1 = self.criterion_l1n(image1_out_diff)
        
        loss_cyc_ncc_0 = self.criterion_ncc(image0_out, image0) * self.config.weight_ncc
        loss_cyc_cha_0 = self.criterion_cha(image0_out, image0, eps=self.epsilon) * self.config.weight_cha
        loss_cyc_ncc_1 = self.criterion_ncc(image1_out, image1) * self.config.weight_ncc
        loss_cyc_cha_1 = self.criterion_cha(image1_out, image1, eps=self.epsilon) * self.config.weight_cha
        
        loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
        loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
        
        return {
            'loss_diff_0': loss_diff_0,
            'loss_diff_1': loss_diff_1,
            'loss_cycle_0': loss_cycle_0,
            'loss_cycle_1': loss_cycle_1,
            'image0_out': image0_out,
            'image1_out': image1_out
        }


    def _optimize_step(self, loss: torch.Tensor):
        """Perform optimization step."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def _update_flow_meters(
        self, 
        meters: Dict[str, utils.AverageMeter],
        loss_dict: Dict[str, torch.Tensor],
        image0: torch.Tensor,
        image1: torch.Tensor
        ):
        """Update meters related to flow losses."""
        numel = image1.numel()
        meters['loss_all_full'].update(loss_dict['flow_model_loss'].item(), numel)
        meters['loss_ncc_all_full'].update(loss_dict['loss_ncc_1'].item(), numel)
        meters['loss_cha_all_full'].update(loss_dict['loss_cha_1'].item(), numel)
        meters['loss_reg_all_full'].update(loss_dict['loss_reg_1'].item(), numel)
        meters['loss_ncc_all_full'].update(loss_dict['loss_ncc_0'].item(), numel)
        meters['loss_cha_all_full'].update(loss_dict['loss_cha_0'].item(), numel)
        meters['loss_reg_all_full'].update(loss_dict['loss_reg_0'].item(), numel)


    def _update_cycle_meters(
        self,
        meters: Dict[str, utils.AverageMeter],
        cycle_loss_dict: Dict[str, torch.Tensor],
        image1: torch.Tensor
        ):
        """Update meters related to cycle consistency losses."""
        numel = image1.numel()
        meters['loss_diff_all'].update(cycle_loss_dict['loss_diff_0'].item(), numel)
        meters['loss_diff_all'].update(cycle_loss_dict['loss_diff_1'].item(), numel)
        meters['loss_all_cycle'].update(cycle_loss_dict['loss_cycle_0'].item(), numel)
        meters['loss_all_cycle'].update(cycle_loss_dict['loss_cycle_1'].item(), numel)


    def _log_metrics(self, meters: Dict[str, utils.AverageMeter], epoch: int):
        """Log metrics to wandb."""
        wandb.log({
            "Loss_all/train": meters['loss_all'].avg,
            "Loss_full/train_all": meters['loss_all_full'].avg,
            "Loss_full/train_img_ncc": meters['loss_ncc_all_full'].avg,
            "Loss_full/train_img_cha": meters['loss_cha_all_full'].avg,
            "Loss_full/train_reg": meters['loss_reg_all_full'].avg,
            "Loss_cycle/train_all": meters['loss_all_cycle'].avg,
            "Loss_cycle/train_diff": meters['loss_diff_all'].avg
        }, step=epoch)