from configuration.config import parse_arguments_main
from architecture.centralized.centralized_learning_2d import CentralizedAD2D
from architecture.centralized.centralized_learning_3d import CentralizedAD3D


def main(args):
    
    if args.paradigm == 'c2d':
        work = CentralizedAD2D(args=args)
        work.run_work_flow()
        
    if args.paradigm == 'c3d':
        work = CentralizedAD3D(args=args)
        work.run_work_flow()


if __name__ == '__main__':
    args = parse_arguments_main()    
    main(args)