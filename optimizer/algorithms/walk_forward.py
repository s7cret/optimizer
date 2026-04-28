from optimizer.errors import UnsupportedFeatureError
def run(*a, **k): raise UnsupportedFeatureError('walk-forward placeholder requires range-aware runner adapter')
