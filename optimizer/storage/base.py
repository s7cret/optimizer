class StorageBackend:
    def init_run(self, *a, **k):
        raise NotImplementedError

    def save_trial(self, trial):
        raise NotImplementedError

    def load_trials(self):
        raise NotImplementedError

    def close(self):
        pass
