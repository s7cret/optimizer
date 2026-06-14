from abc import ABC, abstractmethod


class SearchAlgorithm(ABC):
    @abstractmethod
    def generate(self, space, config): ...
