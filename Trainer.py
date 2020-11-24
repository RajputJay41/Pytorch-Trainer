import torch.nn as nn
import torch

class mode(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
    
    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)

    def fetch_optimizer(self, *args, **kwargs):
        return

    def fetch_scheduler(self, *args, **kwargs):
        return

    
    def train_one_step(data, device):
        self.optimizer.zero_grad()
        for k, v in data.items():
            data[k] = v.to(device)
        _, loss = self(**data)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def train_one_epoch(self,data_loader, device):
        self.train()
        epoch_loss = 0
        for data in data_loader:
            loss = self.train_one_step(data, device)
            epoch_loss += loss
        return epoch_loss / len(data_loader)

            

    def fit(self, train_dataset, batch_size, epochs):
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size = batch_size,
                shuffle = True

            )
        if next(self.parameters().device) != device:
            self.to(device)

        self.optimizer = self.fetch_optimizer()
        self.scheduler = self.fetch_scheduler()


        for _ in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader, device)

class MyModel(Model):
    super().__init__()
    def __init__(self,num_classes):
        # define network , layers
        self.out = nn.linear(128, num_classes)
    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer)
        return sch

    def fetch_optimizer(self):
        # params = self.parameters
        # define opt here
        opt = torch.optim.Adam(self.parameters)
        return opt

    def forward(self, features, targets=None):
        # x = self.something(forward)
        # x  = ..............(self)
        loss = self.loss(out, targets)
        return out, loss


m = MyModel(....)
m.fit(train_dataset, batch_size=16, device="cuda")




