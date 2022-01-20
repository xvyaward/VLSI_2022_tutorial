import torch
import torch.nn as nn

__all__ = ['MLP']

class MLP_class(nn.Module):
  def __init__(self, hidden_size):
    super(MLP_class, self).__init__()

    self.numclass = 10
    self.linear = '''fill in here'''                    # input image size is 32x32 in this example
    self.relu = '''fill in here'''
    self.fc = '''fill in here'''

  def forward(self, x):                                 # model structure is FC-ReLU-FC
    
    '''
        fill in here
    '''
    
    return x

def MLP(hidden_size=128):
  return MLP_class(hidden_size)
