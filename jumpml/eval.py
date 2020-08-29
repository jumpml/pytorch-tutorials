
# © 2020 JumpML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import numpy as np
import time

class evalModel():
  def __init__(self, model, lossFn, device):
    self.model =  model
    self.lossFn = lossFn
    self.device = device
    self.losses = []

  def predict(self, X):
    self.model.eval()
    with torch.no_grad():
      X = X.to(self.device)
      y_hat = self.model(X)
    
    return(y_hat)

  def predictClass(self, X):
    y_hat = self.predict(X)
    conf, pred = y_hat.data.max(1) 
    return(pred.numpy()[0], np.exp(conf.numpy()[0]))

  def evalClass(self, dataloader):
    self.model.eval()
    start_time = time.time()
    loss = 0
    correct = 0
    predictions = torch.tensor([])
    labels = torch.tensor([])
    with torch.no_grad():
      for X_test, y_test in dataloader:
        labels = torch.cat((labels, y_test),dim=0)
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)
        y_hat = self.model(X_test)
        loss += self.lossFn(y_hat, y_test).item()
        pred = y_hat.data.max(1, keepdim=True)[1]
        predictions = torch.cat((predictions, pred.cpu()),dim=0)
        correct += pred.eq(y_test.data.view_as(pred)).sum()

    elapsed_time = time.time() - start_time
    loss /= len(dataloader.dataset)
    self.losses.append(loss)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f'\nAvg. loss: {loss:.4f}, Accuracy: {accuracy} %  Elapsed Time={elapsed_time}\n')
    cm = confusion_matrix(predictions.numpy(), labels.numpy())
    return(accuracy, cm)
