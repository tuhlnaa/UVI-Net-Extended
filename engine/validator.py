import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

#from utils.meters import AverageMeter
from utils import utils

class Validator:
    """Validator class for evaluating the medical image interpolation model."""
    
    def __init__(
        self,
        flow_model: nn.Module,
        refinement_model: nn.Module,
        reg_model_bilin: nn.Module,
        criterion_ncc: nn.Module,
        device: torch.device
    ):
        """Initialize validator with models and evaluation metrics."""
        self.flow_model = flow_model
        self.refinement_model = refinement_model
        self.reg_model_bilin = reg_model_bilin
        self.criterion_ncc = criterion_ncc
        self.device = device


    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validate the model on the validation dataset.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            float: Average NCC score
        """
        eval_ncc = utils.AverageMeter()
        
        self.flow_model.eval()
        self.refinement_model.eval()
        
        with torch.no_grad():
            for data in val_loader:
                data = [t.to(self.device) for t in data]
                image0, image1 = data[0], data[1]

                ncc_score = self._compute_ncc_score(image0, image1) # Forward pass
                eval_ncc.update(ncc_score.item(), image0.size(0))

        # Log validation metrics
        wandb.log({"Validate/NCC": eval_ncc.avg}, step=epoch)
        
        return eval_ncc.avg


    def _compute_ncc_score(
        self, 
        image0: torch.Tensor, 
        image1: torch.Tensor
    ) -> torch.Tensor:
        """Compute NCC score between warped image0 and image1."""
        image0_image1 = torch.cat((image0, image1), dim=1)
        _, _, flow_0_1, _ = self.flow_model(image0_image1)
        
        # Warp image0 to image1's space
        image_0_1 = self.reg_model_bilin([image0, flow_0_1.float()])
        
        # Compute negative NCC (higher is better)
        return -1 * self.criterion_ncc(image_0_1, image1)