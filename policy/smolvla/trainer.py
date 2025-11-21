import transformers
import torch 
from policy.trainer import BaseTrainer
from transformers.optimization import get_cosine_schedule_with_warmup

def create_smolvla_scheduler(optimizer, num_training_steps):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    # SmolVLA 参数
    num_warmup_steps = 1_000
    num_decay_steps = 30_000
    peak_lr = 1e-4
    decay_lr = 2.5e-6
    
    # 如果训练步数少于衰减步数，自动缩放
    actual_warmup_steps = num_warmup_steps
    actual_decay_steps = num_decay_steps
    
    if num_training_steps < num_decay_steps:
        scale_factor = num_training_steps / num_decay_steps
        actual_warmup_steps = int(num_warmup_steps * scale_factor)
        actual_decay_steps = num_training_steps
        print(f"Auto-scaling LR scheduler: warmup={actual_warmup_steps}, decay={actual_decay_steps}")
    
    def lr_lambda(current_step):
        # Warmup 阶段
        if current_step < actual_warmup_steps:
            if current_step <= 0:
                return 1 / (actual_warmup_steps + 1)
            frac = 1 - current_step / actual_warmup_steps
            return (1 / (actual_warmup_steps + 1) - 1) * frac + 1
        
        # Cosine Decay 阶段
        step = min(current_step, actual_decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
        alpha = decay_lr / peak_lr
        decayed = (1 - alpha) * cosine_decay + alpha
        return decayed
    
    return LambdaLR(optimizer, lr_lambda, -1)

class Trainer(BaseTrainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = create_smolvla_scheduler(
                self.optimizer if optimizer is None else optimizer,
                num_training_steps
            )
        return self.lr_scheduler