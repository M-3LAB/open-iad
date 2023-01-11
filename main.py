from configuration.architecture.config import parse_arguments_main
from architecture.centralized.centralized_learning_2d import Centralized2DTrain
from architecture.centralized.centralized_learning_3d import Centralized3DTrain


def main(args):
    
    if args.paradigm == 'c2d':
        work = Centralized2DTrain(args=args)
        work.run_work_flow()
        
    if args.paradigm == 'c3d':
        work = Centralized3DTrain(args=args)
        work.run_work_flow()


if __name__ == '__main__':
    args = parse_arguments_main()    
    main(args)