import torch.nn.functional as F
import torch.optim as optim 
from jumpml import eval
import numpy as np
import matplotlib.pyplot as plt

class TrainModel:
  def __init__(self, model, train_loader, val_loader, device="cpu", learning_rate=0.1):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    
    self.device = device
    self.learning_rate = 0.1
    self.log_frequency = 100
    
    self.train_losses = []
    self.val_accuracy = []
    self.train_len = len(self.train_loader.dataset)

    # Setup loss function, optimizer (TBD: read these from a config file)
    self.lossFn = F.nll_loss
    self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    # Apply decaying Learning Rate
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.25)

    # Setup evaluation object
    self.eval = eval.evalModel(self.model, self.lossFn, self.device)
    
  def run_training_loop(self, n_epochs=1):
    acc,_ = self.eval.evalClass(self.val_loader)
    self.val_accuracy.append(acc)
    for epoch in range(n_epochs):
      print(f'Epoch-{epoch + 1} lr: {self.optimizer.param_groups[0]["lr"]}')
      self.train_epoch()
      acc,_ = self.eval.evalClass(self.val_loader)
      self.val_accuracy.append(acc)

  def train_epoch(self):
    self.model.train()
    for batch_idx, (X_train, y_train) in enumerate(self.train_loader):
      # Move to device
      X_train, y_train = X_train.to(self.device), y_train.to(self.device)
      # Initialize gradients
      self.optimizer.zero_grad()
      # Predict on current batch of data
      y_hat = self.model(X_train)
      # Calculate Average Loss
      loss = self.lossFn(y_hat, y_train)
      # Calculate Gradients
      loss.backward()
      # Update model parameters using SGD
      self.optimizer.step()
      if batch_idx % self.log_frequency == 0:
        print(f'Train  {batch_idx * len(X_train)}/{self.train_len} Loss:{loss.item()}\n')
        self.train_losses.append(loss.item())
  
    self.scheduler.step() # update learning rate for next call to train()

  def plot_loss(self):
    plt.figure(figsize=(10, 8), dpi=60)
    bs = self.train_loader.batch_size
    x = range(0,self.log_frequency*len(self.train_losses)*bs,self.log_frequency*bs)
    plt.plot(x, self.train_losses, color='blue')
    x = range(0, self.train_len*len(self.eval.losses), self.train_len)
    plt.plot(x,self.eval.losses,'r*')
    plt.plot(x,np.array(self.val_accuracy)/100.0,'g*-')
    plt.legend(['Train Loss', 'Val. Loss', 'Val. Acc.'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')





