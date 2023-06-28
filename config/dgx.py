import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config():
    config = base.get_config()

    config.pretrained.model = "runwayml/stable-diffusion-v1-5"

    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.use_lora = False

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    config.train.learning_rate = 3e-5
    config.train.clip_range = 1e-4

    # sampling
    config.sample.num_steps = 50
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config
