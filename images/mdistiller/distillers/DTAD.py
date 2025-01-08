import torch
from torch import nn
import torch.nn.functional as F
from ._base import Distiller

class DTAD(Distiller):
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
        student, teacher, cfg,
        initial_temperature=8.0, 
        min_temperature=1.0, 
        max_temperature=8,
        alpha=0.5,
        beta=0.9,
        gamma=0.5,
    ):
        super(DTAD, self).__init__(student, teacher)

        self.cfg = cfg            
        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_epoch = cfg.SOLVER.EPOCHS
        self.warmup = cfg.DTAD.WARMUP,
        self.loss_history = []
        
        # Constants for importance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def update_temperature(self, current_epoch, teacher_loss, student_loss):
        """
        Dynamically update temperature based on training dynamics.

        Args:
            current_epoch (int): Current training epoch.
            total_epochs (int): Total number of training epochs.
            teacher_loss (float): Loss of teacher model.
            student_loss (float): Loss of student model.
        """
        # Store loss and compute gradient information
        progress = current_epoch / self.max_epoch
        scale_factor = 0.5 + torch.cos(
            torch.pi * torch.tensor(
                progress * 0.72, 
                device="cuda"
            )
        )

        self.loss_history.append((teacher_loss, student_loss))

        window_size = 5
        self.loss_history = self.loss_history[-window_size:]
        recent_losses = self.loss_history
        teacher_losses = torch.tensor([loss[0] for loss in recent_losses], device='cuda')
        student_losses = torch.tensor([loss[1] for loss in recent_losses], device='cuda')

        teacher_loss = teacher_losses.mean()
        student_loss = student_losses.mean()
        
        # Dynamic scaling based on loss divergence
        loss_divergence = torch.abs(teacher_loss - student_loss)
        adaptive_scale = 0.8 + torch.log(1 + loss_divergence)
        
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
        
    def forward_train(self, image, target, **kwargs):
        """
        Forward pass to compute the loss based on the specified loss type.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        student_logits, _ = self.student(image)
        teacher_logits, _ = self.teacher(image)

        outputs = target

        combined_loss = self.combined_loss(student_logits, teacher_logits, outputs)
        warmup = min(kwargs["epoch"] / 20, 1.0)

        loss = warmup * combined_loss
        losses_dict = {
            "loss": loss,
        }

        teacher_loss = F.cross_entropy(teacher_logits, outputs)

        with torch.no_grad():
            self.update_temperature(
                kwargs["epoch"], 
                teacher_loss=teacher_loss.item(), 
                student_loss=loss
            )

        return student_logits, losses_dict

    def get_extra_parameters(self):
        """
        Retrieve temperature value for logging purposes.

        Returns:
            dict: Extra parameters to log.
        """

        return {"temperature": self.get_temperature()}