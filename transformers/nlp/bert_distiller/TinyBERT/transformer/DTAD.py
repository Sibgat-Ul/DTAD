import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

class LossManager:
    def __init__(
            self,
            initial_temperature,
            min_temperature
    ):
        self.current_temperature = initial_temperature
        self.min_temperature = min_temperature

    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)

        return (logit - mean) / (1e-7 + stdv)

    def kd_loss(self, logits_student_in, logits_teacher_in, logit_stand=True):
        temperature = self.current_temperature

        logits_student = self.normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = self.normalize(logits_teacher_in) if logit_stand else logits_teacher_in
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)

        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature * temperature

        return loss_kd

class DynamicTemperatureScheduler(nn.Module):
    def __init__(
            self,
            initial_temperature=8.0,
            min_temperature=4.0,
            max_temperature=8,
            max_epoch=50,
            warmup=20,
    ):
        super(DynamicTemperatureScheduler, self).__init__()

        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.logit_stand = False
        self.loss_type = "kd"

        # Constants for importance
        self.loss_manager = LossManager(
            initial_temperature,
            min_temperature
        )

    def update_temperature(self, current_epoch, loss_divergence):
        progress = torch.tensor(current_epoch / self.max_epoch)
        cosine_factor = 0.5 * (1 + torch.cos(0.65*torch.pi * progress))
        # log_loss = torch.log(torch.tensor(loss_divergence))
        adaptive_scale = loss_divergence / (loss_divergence + 1)

        # if adaptive_scale > 1:
        #     target_temperature = self.initial_temperature * cosine_factor * (adaptive_scale)
        # else:
        #     target_temperature = self.initial_temperature * cosine_factor
        target_temperature = self.initial_temperature * cosine_factor

        target_temperature = torch.clamp(
            target_temperature,
            self.min_temperature,
            self.max_temperature
        )

        momentum = 0.9
        self.current_temperature = momentum * self.current_temperature + (1 - momentum) * target_temperature
        self.loss_manager.current_temperature = self.current_temperature

    def get_temperature(self):
        """
        Retrieve current temperature value.

        Returns:
            float: Current dynamic temperature.
        """

        return self.current_temperature

    def forward(self, epoch, student_logits, teacher_logits, outputs, loss_type="kd"):
        """
        Forward pass to compute the loss based on the specified loss type.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss_type = self.loss_type

        if loss_type == "kd":
            logits_student = student_logits
            logits_teacher = teacher_logits
            target = outputs

            softmax = nn.Softmax(dim=-1)
            teacher_loss = F.cross_entropy(softmax(logits_teacher), target)
            student_loss = F.cross_entropy(softmax(logits_student), target)

            with torch.no_grad():
                loss_divergence = teacher_loss.item() - student_loss.item()

            loss_ce = 0.1 * student_loss

            loss_kd = 0.9 * self.loss_manager.kd_loss(
                logits_student, logits_teacher, self.logit_stand
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }

            with torch.no_grad():
                self.update_temperature(epoch, loss_divergence)

            self.loss_manager.current_temperature = self.current_temperature
            return sum([l.mean() for l in losses_dict.values()])

        else:
            logits_student = student_logits
            logits_teacher = teacher_logits
            target = outputs

            student_loss = F.cross_entropy(logits_student, target)
            teacher_loss = F.cross_entropy(logits_teacher, target)

            with torch.no_grad():
                loss_divergence = teacher_loss.item() - student_loss.item()

            # losses
            loss_ce = 1.0 * student_loss

            loss_dkd = min(epoch / self.warmup, 1.0) * dkd_loss(
                logits_student,
                logits_teacher,
                target,
                9.0 if self.logit_stand else 1.0,
                18.0 if self.logit_stand else 8.0,
                self.current_temperature,
                self.logit_stand,
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_dkd,
            }

            with torch.no_grad():
                self.update_temperature(epoch, loss_divergence)

            return sum([l.mean() for l in losses_dict.values()])

