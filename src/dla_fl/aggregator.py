import numpy as np

class FogAggregator:
    def __init__(self, cfg):
        self.region = cfg['region']

    def aggregate(self, updates):
        if not updates:
            return None
        weights = [u['weight'] for u in updates]
        params = [u['params'] for u in updates]
        # Pad params to the same length
        max_len = max(len(p) for p in params)
        padded_params = [p + [0.0] * (max_len - len(p)) for p in params]
        avg = np.average(padded_params, axis=0, weights=weights)
        return {'params': avg, 'weight': sum(weights)}

class CloudAggregator:
    def __init__(self, cfg):
        self.global_history = []

    def aggregate(self, fog_update):
        if fog_update is None:
            return None
        self.global_history.append(fog_update)
        return fog_update
