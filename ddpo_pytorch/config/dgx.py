import ml_collections
from ddpo_pytorch.config import base

def get_config():
    config = base.get_config()

    config.mixed_precision = "bf16"
    config.allow_tf32 = True

    config.train.batch_size = 8
    config.train.gradient_accumulation_steps = 4

    # sampling
    config.sample.num_steps = 50
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.per_prompt_stat_tracking = None

    return config