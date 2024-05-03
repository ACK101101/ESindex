from torch import nn

class NonLinearModel(nn.Module):
    '''
    TODO:
    - Custom Loss Function for penalizing outside the range and
    incorrect prediction (maybe more depending on length of string)
    '''
    def __init__(self, input_size):
        super(NonLinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, 1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.relu(x)
        return self.linear3(x)