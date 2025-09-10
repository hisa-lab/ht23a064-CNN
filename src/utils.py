import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix




def set_seed(seed: int = 42):
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)




def accuracy(output, target):
with torch.no_grad():
pred = output.argmax(dim=1)
correct = pred.eq(target).sum().item()
return correct / target.size(0)




def compute_confusion_matrix(y_true, y_pred, labels=(0, 1)):
return confusion_matrix(y_true, y_pred, labels=labels)