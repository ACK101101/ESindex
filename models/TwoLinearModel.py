from torch import nn

class TwoLayerLinearRegressionModel(nn.Module):
    '''
    TODO:
    - Custom Loss Function for penalizing outside the range and
    incorrect prediction (maybe more depending on length of string)
    '''
    def __init__(self, input_size):
        super(TwoLayerLinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)