from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from utils.utils import AverageMeter


class Validator:
    def __init__(
        self,
        flow_model: nn.Module,
        refinement_model: nn.Module,
        feature_model: Optional[nn.Module],
        criterion_ncc: nn.Module,
        reg_model_bilin: nn.Module,
    ):
        """Initialize the validator.
        
        Args:
            flow_model: VoxelMorph model for flow field prediction
            refinement_model: UNet model for refinement
            feature_model: Optional feature extraction model
            criterion_ncc: NCC loss function
            reg_model_bilin: Bilinear registration model
        """
        self.flow_model = flow_model
        self.refinement_model = refinement_model
        self.feature_model = feature_model
        self.criterion_ncc = criterion_ncc
        self.reg_model_bilin = reg_model_bilin

    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Average NCC score
        """
        eval_ncc = AverageMeter()

        with torch.no_grad():
            for data in val_loader:
                self.flow_model.eval()
                self.refinement_model.eval()
                if self.feature_model:
                    self.feature_model.eval()

                ncc = self._validate_step(data)
                eval_ncc.update(ncc.item(), data[0].size(0))

        print(f"Epoch {epoch}, NCC {eval_ncc.avg:.5f}\n", flush=True)
        wandb.log({"Validate/NCC": eval_ncc.avg}, step=epoch)

        return eval_ncc.avg

    def _validate_step(self, data: list) -> torch.Tensor:
        """Perform a single validation step.

        Args:
            data: List of input tensors
            
        Returns:
            NCC score for the step
        """
        data = [t.cuda() for t in data]
        i0, i1 = data[0], data[1]

        # Forward pass
        i0_i1 = torch.cat((i0, i1), dim=1)
        _, _, flow_0_1, _ = self.flow_model(i0_i1)

        # Apply registration
        i_0_1 = self.reg_model_bilin([i0, flow_0_1.float()])

        # Calculate NCC
        ncc = -1 * self.criterion_ncc(i_0_1, i1)
        
        return ncc