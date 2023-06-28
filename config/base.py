import ml_collections

def get_config():

    config = ml_collections.ConfigDict()

    # misc
    config.seed = 42
    config.logdir = "logs"
    config.num_epochs = 100
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.use_lora = True

    # pretrained model initialization
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "runwayml/stable-diffusion-v1-5"
    pretrained.revision = "main"

    # training
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 1e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    train.cfg = True
    train.adv_clip_max = 10
    train.clip_range = 1e-4
    train.timestep_fraction = 1.0

    # sampling
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 10
    sample.eta = 1.0
    sample.guidance_scale = 5.0
    sample.batch_size = 1
    sample.num_batches_per_epoch = 2

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = ml_collections.ConfigDict()
    config.per_prompt_stat_tracking.buffer_size = 16
    config.per_prompt_stat_tracking.min_count = 16

    return config