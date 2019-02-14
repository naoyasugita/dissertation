class Config:
    def __init__(self, **args):
        self.model = ModelConfig()
        self.model_api = ModelAPIConfig()
        self.training = TrainingConfig()

class ModelConfig:
    def __init__(self, **args):
        pass

class ModelAPIConfig:
    def __init__(self, **args):
        pass

class TrainingConfig:
    def __init__(self, **args):
        pass