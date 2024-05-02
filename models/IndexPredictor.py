from abc import ABC, abstractmethod

class IndexPredictor(ABC):
    def __init__(self, data, model):
        pass

    @abstractmethod
    def predict(self, data) -> int:
        pass

    def find_max_error():
        pass