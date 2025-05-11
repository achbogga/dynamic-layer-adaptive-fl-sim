import random

class TaskScheduler:
    def __init__(self, cfg, partitions):
        self.partitions = partitions
        self.idle_threshold = cfg.get('idle_threshold', 0.2)
        self.device_profiles = cfg['devices']
        self.global_model = None

    def dispatch_tasks(self):
        updates = []
        for device, profile in self.device_profiles.items():
            if profile['idle_fraction'] >= self.idle_threshold:
                part = random.choice(self.partitions)
                update = part.train_on_device(device)
                updates.append(update)
        return updates

    def update_global_model(self, global_update):
        self.global_model = global_update