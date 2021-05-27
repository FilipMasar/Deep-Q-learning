import argparse

from src.agent import Agent


def argument_parser():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Agent")

    parser.add_argument("action", type=str, choices=["train", "test"], help='Agent actions')
    parser.add_argument("--environment", default="CartPole-v1", choices=["CartPole-v1", "LunarLander-v2"], type=str,
                        help="Environment to be in")

    parser.add_argument("--episodes", default=150, type=int, help="Number of training episodes")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor")
    parser.add_argument("--epsilon_start", default=0.5, type=float, help="Initial exploration factor")
    parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor")
    parser.add_argument("--epsilon_final_at", default=500, type=int, help="Episode when epsilon should be final")
    parser.add_argument("--target_update_freq", default=0, type=int, help="Update frequency of target network")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer")

    return parser


def perform_training(params):
    agent = Agent(**params)
    agent.train()
    agent.plot_training_stats()
    agent.save()


def perform_testing(environment):
    agent = Agent(environment=environment)
    agent.load()
    returns = agent.test(3, render=True)
    print(f'ℹ️ Average return over 3 episodes was {returns}')


if __name__ == "__main__":
    argument_parser = argument_parser()
    args = argument_parser.parse_args()

    if args.action == "train":
        args = vars(args)
        del args["action"]
        perform_training(args)
    elif args.action == "test":
        perform_testing(args.environment)
    else:
        argument_parser.print_help()
