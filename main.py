from configuration.config import parse_arguments_main
from paradigms.centralized.centralized_learning_2d import CentralizedAD2D
from paradigms.centralized.centralized_learning_3d import CentralizedAD3D


def main(args):
    # centralized learning for 2d anomaly detection
    if args.paradigm == 'c2d':
        work = CentralizedAD2D(args=args)
        work.run_work_flow()
        
    # centralized learning for 3d anomaly detection
    if args.paradigm == 'c3d':
        work = CentralizedAD3D(args=args)
        work.run_work_flow()

    # federated learning for 2d anomaly detection
    if args.paradigm == 'f2d':
        # TODO
        pass

if __name__ == '__main__':
    args = parse_arguments_main()    
    main(args)