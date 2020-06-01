import yaml

class Config:
    def __init__(self):
        self.load_model_epochs = None
        self.debug = None
        self.n_epochs = None
        self.load_g_model_score = None
        self.load_d_model_score = None
        self.model_no = None
        self.batch_size = None
        self.n_split = None
        self.max_lr = None
        self.min_lr = None
        self.lambda1 = None
        self.lambda2 = None
        self.seed = None
        self.dataloader_seed = None
        self.device = None
        self.size = None
        self.load()

    def load(self):
        with open('config/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            self.load_model_epochs = config.get('LOAD_MODEL_EPOCH')
            self.debug = config.get('DEBUG')
            self.n_epochs = config.get('N_EPOCHS')
            self.load_g_model_score = config.get('LOAD_G_MODEL_SCORE')
            self.load_d_model_score = config.get('LOAD_D_MODEL_SCORE')
            self.model_no = config.get('MODEL_NO')
            self.batch_size = config.get('BATCH_SIZE')
            self.n_split = config.get('N_SPLIT')
            self.max_lr = config.get('MAX_LR')
            self.min_lr = config.get('MIN_LR')
            self.lambda1 = config.get('LAMBDA1')
            self.lambda2 = config.get('LAMBDA2')
            self.seed = config.get('SEED')
            self.dataloader_seed = config.get('DATALOADER_SEED')
            self.device = config.get('DEVICE')
            self.size = config.get('SIZE')


class TestConfig:
    def __init__(self):
        self.load_model_epochs = None
        self.debug = None
        self.load_g_model_score = None
        self.batch_size = None
        self.seed = None
        self.device = None
        self.size = None
        self.load()

    def load(self):
        with open('config/test_config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            self.load_model_epochs = config.get('LOAD_MODEL_EPOCH')
            self.debug = config.get('DEBUG')
            self.load_g_model_score = config.get('LOAD_G_MODEL_SCORE')
            self.batch_size = config.get('BATCH_SIZE')
            self.seed = config.get('SEED')
            self.device = config.get('DEVICE')
            self.size = config.get('SIZE')