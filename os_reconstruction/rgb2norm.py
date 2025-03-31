import torch.nn as nn
import torch.nn.functional as F

class RGB2NormalNet(nn.Module):
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
    fc1 : torch.nn.Linear -
        The first fully connected layer
    fc2 : torch.nn.Linear -
        The second fully connected layer
    fc3 : torch.nn.Linear -
        The third fully connected layer
    fc4 : torch.nn.Linear -
        The fourth fully connected layer
    dropout : torch.nn.Dropout -
        Dropout layer with the specified probability
    '''
    def __init__(self):
        super(RGB2NormalNet, self).__init__()
        
        self.input_size     = 5
        self.hidden_size    = 64
        self.output_size    = 2
        self.probability    = 0.05
        
        self.fc1        = nn.Linear(self.input_size, self.hidden_size)
        self.fc2        = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3        = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4        = nn.Linear(self.hidden_size, self.output_size)
        self.dropout    = nn.Dropout(self.probability)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x