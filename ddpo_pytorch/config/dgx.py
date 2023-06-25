import ml_collections
from ddpo_pytorch.config import base

def get_config():
    config = base.get_config()

    config.mixed_precision = "no"
    config.allow_tf32 = True
    config.use_lora = False

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 8
    config.train.learning_rate = 1e-5
    config.train.clip_range = 1.0

    # sampling
    config.sample.num_steps = 50
    config.sample.batch_size = 16
    config.sample.num_batches_per_epoch = 2

    config.per_prompt_stat_tracking = None

    return config