import torch
import torch.nn as nn
import torch.nn.functional as F

class LossManagerDTAD(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(LossManagerDTAD, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def cosine_loss(self, student_logits, teacher_logits):
        """
        Compute cosine similarity loss between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: Cosine similarity loss.
        """
        
        # Normalize logits
        student_norm = F.normalize(student_logits, p=2, dim=1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=1)
        
        # Compute cosine similarity loss
        cosine_loss = 1 - F.cosine_similarity(student_norm, teacher_norm).mean()
        return cosine_loss

    def rmse_loss(self, student_logits, teacher_logits):
        """
        Compute Root Mean Square Error (RMSE) between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: RMSE loss.
        """
        
        rmse = torch.sqrt(F.mse_loss(student_logits, teacher_logits))
        return rmse

    def hard_loss(self, student_logits, outputs):
        """
        Compute hard loss (cross-entropy) between student logits and true labels.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        
        return F.cross_entropy(student_logits, outputs)

    def soft_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute knowledge distillation loss with dynamic temperature.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: Knowledge distillation loss.
        """
        soft_targets = F.softmax(teacher_logits / self.current_temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.current_temperature, dim=1)
        
        loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        return loss
    
    def mse_loss(self, student_logits, teacher_logits):
        """
        Compute Mean Squared Error (MSE) between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: MSE loss.
        """
        
        return F.mse_loss(student_logits, teacher_logits)

    def combined_loss(self, student_logits, teacher_logits, outputs):
        """
        Compute combined loss including soft distillation, hard, cosine, and RMSE losses.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Combined loss.
        """
        
        # Soft loss
        soft_loss = self.alpha * self.soft_distillation_loss(student_logits, teacher_logits)

        # Hard loss
        hard_loss = (1 - self.alpha) * self.hard_loss(student_logits, outputs)
        
        # Cosine loss
        cosine_loss = self.beta * self.cosine_loss(student_logits, teacher_logits)
        
        # RMSE loss
        rmse_loss = self.gamma * self.rmse_loss(student_logits, teacher_logits)
        
        return hard_loss + soft_loss + cosine_loss + rmse_loss
    
    # def forward(self, image, target, **kwargs):
    #     """
    #     Forward pass to compute the loss based on the specified loss type.

    #     Args:
    #         student_logits (torch.Tensor): Logits from student model.
    #         teacher_logits (torch.Tensor): Logits from teacher model.
    #         outputs (torch.Tensor): True labels.

    #     Returns:
    #         torch.Tensor: Computed loss.
    #     """
    #     student_logits, _ = self.student(image)
    #     teacher_logits, _ = self.teacher(image)

    #     outputs = target

    #     combined_loss = self.combined_loss(student_logits, teacher_logits, outputs)

    #     loss = combined_loss
    #     losses_dict = {
    #         "loss": loss,
    #     }

    #     teacher_loss = F.cross_entropy(teacher_logits, outputs)

    #     with torch.no_grad():
    #         self.update_temperature(
    #             kwargs["epoch"], 
    #             teacher_loss=teacher_loss.item(), 
    #             student_loss=loss
    #         )

    #     return student_logits, losses_dict
