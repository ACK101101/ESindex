from abc import ABC, abstractmethod
from torch import no_grad

class IndexPredictor(ABC):
    def __init__(self, data, model):
        self.data = data
        self.model = model

    @abstractmethod
    def predict(self, data) -> int:
        pass

    def find_max_error(self):
        keys = self.data.get_series()
        self.model.eval()
        with no_grad():
            max_error = 0
            for key in keys:
                
                prediction = self.model(value)
                error = abs(value - prediction)
                if error > max_error:
                    max_error = error
        pass