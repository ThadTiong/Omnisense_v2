import torch.nn as nn
import torch.nn.functional as F

class Pix2NormalNet(nn.Module):
    '''
    A neural network model for predicting surface normals from RGB images.

    Attributes
    ----------
    input_size : int -
        The number of inputs - 5 ((R,G,B), (pixel_x, pixel_y))
    hidden_size : int -
        The number of neurons in hidden layers - 64
    output_size : int -
        The number of predictions - 2 (normal_x, normal_y)
    probability : float -
        The dropout probability - 0.05
    layers : torch.nn.ModuleList -
        List of linear layers
    dropout : torch.nn.Dropout -
        Dropout layer with the specified probability
    '''
    def __init__(self):
        super(Pix2NormalNet, self).__init__()
        
        self.input_size     = 5
        self.hidden_size    = 64
        self.output_size    = 2
        self.probability    = 0.05
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)
        ])
        self.dropout = nn.Dropout(self.probability)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.layers[-1](x)
        return x