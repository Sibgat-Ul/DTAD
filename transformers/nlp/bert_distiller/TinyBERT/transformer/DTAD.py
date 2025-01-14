import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicTemperatureScheduler(nn.Module):
    """
    Dynamic Temperature Scheduler for Knowledge Distillation.

    Args:
        initial_temperature (float): Starting temperature value.
        min_temperature (float): Minimum allowable temperature.
        max_temperature (float): Maximum allowable temperature.
        schedule_type (str): Type of temperature scheduling strategy.
        loss_type (str): Type of loss to use (combined or general KD).
        alpha (float): Importance for soft loss, 1-alpha for hard loss.
        beta (float): Importance of cosine loss.
        gamma (float): Importance for RMSE loss.
    """
    def __init__(
        self, 
        initial_temperature=8.0, 
        min_temperature=1.0, 
        max_temperature=8,
        max_epoch=50,
        warmup=20,
        loss_type="combined_loss",
        alpha=0.5,
        beta=0.5,
        gamma=0.1,
    ):
        super(DynamicTemperatureScheduler, self).__init__()

        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.loss_type = loss_type
        self.max_epoch = max_epoch
        self.warmup = warmup
        
        # Tracking training dynamics
        self.loss_history = []
        self.student_loss = []

        # Constants for importance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def update_temperature(self, current_epoch, loss_divergence):
        """
        Dynamically update temperature based on training dynamics.

        Args:
            current_epoch (int): Current training epoch.
            total_epochs (int): Total number of training epochs.
            teacher_loss (float): Loss of teacher model.
            student_loss (float): Loss of student model.
        """
        total_epochs = self.max_epoch
        
        # Cosine annealing with adaptive scaling
        progress = current_epoch / total_epochs
        scale_factor = 0.5 + torch.cos(
            torch.pi * torch.tensor(
                progress * 0.7, 
                device="cuda"
            )
        )

        adaptive_scale = 0.5 + 0.7 * loss_divergence
        
        # Update temperature
        self.current_temperature = max(
            self.min_temperature, 
            min(
                self.max_temperature*torch.exp(torch.tensor(-progress*.4, device="cuda")), 
                0.1 + self.initial_temperature * scale_factor * adaptive_scale * torch.exp(torch.tensor(-progress*.75, device="cuda"))
            )
        )
        
    def get_temperature(self):
        """
        Retrieve current temperature value.

        Returns:
            float: Current dynamic temperature.
        """
        
        return self.current_temperature

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
        
    def mae_loss(self, student_logits, teacher_logits):
        """
        Compute Root Mean Square Error (RMSE) between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: RMSE loss.
        """
        
        rmse = torch.nn.L1Loss()(student_logits, teacher_logits)
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
        
        return torch.nn.CrossEntropyLoss()(student_logits, outputs)
    
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

    def combined_loss(self, student_logits, teacher_logits, outputs):
        """Only include the additional losses (cosine and RMSE) here"""
        # Cosine loss
        cosine_loss = self.beta * self.cosine_loss(student_logits, teacher_logits)
        # RMSE loss
        rmse_loss = self.gamma * self.rmse(student_logits, teacher_logits)
        return cosine_loss + rmse_loss

    def forward(self, epoch, student_logits, teacher_logits, outputs):
        temp_ratio = (self.current_temperature - 1.0) / (3.0 - 1.0)
        temp_ratio = max(0, min(1, temp_ratio))
        
        # Base losses (always present)
        soft_loss = self.soft_distillation_loss(student_logits, teacher_logits)
        
        hard_loss = self.hard_loss(student_logits, outputs)
        teacher_loss = self.hard_loss(teacher_logits, outputs)
        
        loss_divergence = teacher_loss - hard_loss
        log_loss_divergence = torch.log(1 + torch.abs(loss_divergence))
        
        # Temperature-dependent weighting for soft vs hard
        if self.current_temperature > 1:
            soft_weight = self.alpha * temp_ratio + 0.15 * (1 - temp_ratio)
            hard_weight = (1 - self.alpha) * temp_ratio + 0.85 * (1 - temp_ratio)
        else:
            soft_weight = 0.15
            hard_weight = 0.85
            
        # Additional losses only when temperature is higher
        additional_losses = temp_ratio * self.combined_loss(student_logits, teacher_logits, outputs)
        
        with torch.no_grad():
            self.update_temperature(
                    current_epoch = epoch, 
                    loss_divergence = log_loss_divergence 
                )
            
        warmup = 1 if self.warmup == None else min(epoch / self.warmup, 1.0)
        
        total_loss = (
            soft_weight * soft_loss + 
            hard_weight * hard_loss + 
            additional_losses
        ) + log_loss_divergence
        
        return  warmup * total_loss