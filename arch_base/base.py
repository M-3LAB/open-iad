class Base():
    def __init__(self, config, device, file_path):
        self.config = config
        self.device = device
        self.file_path = file_path
    
    def train_epoch(self, train_loaders, inf=''):
        pass


    def prediction(self, valid_loader):
        pass