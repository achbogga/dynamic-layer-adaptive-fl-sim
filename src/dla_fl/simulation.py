import logging
import numpy as np
from dla_fl.scheduler import TaskScheduler
from dla_fl.aggregator import FogAggregator, CloudAggregator
from dla_fl.model_partition import ModelPartitioner
from dla_fl.utils import setup_logging, load_config

class SimulationRunner:
    def __init__(self, config: dict):
        setup_logging()
        self.config = config
        self.partitions = ModelPartitioner(self.config['model']).create_partitions()
        self.scheduler = TaskScheduler(self.config['scheduler'], self.partitions)
        self.fog_agg = FogAggregator(self.config['fog'])
        self.cloud_agg = CloudAggregator(self.config['cloud'])
        self.history = []

    def run(self):
        rounds = self.config['simulation']['rounds']
        for idx in range(rounds):
            logging.info(f"Round {idx+1}/{rounds}")
            updates = self.scheduler.dispatch_tasks()
            fog_upd = self.fog_agg.aggregate(updates)
            global_upd = self.cloud_agg.aggregate(fog_upd)
            self.scheduler.update_global_model(global_upd)
            avg = None
            if global_upd and 'params' in global_upd:
                avg = float(np.mean(global_upd['params']))
            self.history.append({'round': idx+1, 'avg_param': avg})
        metrics = {
            'total_rounds': rounds,
            'final_avg_param': self.history[-1]['avg_param']
        }
        return metrics, self.history

if __name__ == '__main__':
    cfg = load_config()
    runner = SimulationRunner(cfg)
    metrics, _ = runner.run()
    print(f"Final avg param: {metrics['final_avg_param']}")