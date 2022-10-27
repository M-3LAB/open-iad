class ModelBase():
    def __init__(self, config, device, file_path, net, optimizer, scheduler):
        self.config = config
        self.device = device
        self.file_path = file_path
        self.net = net
    
    def train_epoch(self, train_loader, task_id, inf=''):
        pass


    def prediction(self, valid_loader, task_id=None):
        pass