from torch import nn

class LinearRegressionModel(nn.Module):
    '''
    TODO:
    - Custom Loss Function for penalizing outside the range and
    incorrect prediction (maybe more depending on length of string)
    '''
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)