from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def init_run(self, *a, **k):
        ...

    @abstractmethod
    def save_trial(self, trial):
        ...

    @abstractmethod
    def load_trials(self):
        ...

    def close(self):
        return None
