from configuration.config import parse_arguments_centralized
from arch_centralized.centralized_learning import CentralizedTrain


def centralized_training():
    args = parse_arguments_centralized()
    
    for key_arg in ['dataset', 'model']:
        if not vars(args)[key_arg]:
            raise ValueError('Parameter {} Must Be Refered!'.format(key_arg))

    work = CentralizedTrain(args=args)
    work.run_work_flow()



if __name__ == '__main__':
    centralized_training()