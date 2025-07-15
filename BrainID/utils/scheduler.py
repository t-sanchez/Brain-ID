from torch.optim.lr_scheduler import LambdaLR

class PolynomialWarmupDecayLR(LambdaLR):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, power: float = 2.0):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch+1) / float(max(1, self.warmup_epochs))
            else:
                decay_epochs = self.total_epochs - self.warmup_epochs
                decay_epoch = epoch - self.warmup_epochs
                if decay_epoch > decay_epochs:
                    return 0.0
                return (1 - decay_epoch / decay_epochs) ** self.power

        super().__init__(optimizer, lr_lambda)
