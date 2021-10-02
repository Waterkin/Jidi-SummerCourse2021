import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--game_name', default='snake1v1')
    parser.add_argument('--algo', default='dqn', help='dqn')

    # trainer
    parser.add_argument('--max_episodes', default=80000, type=int)  # 最大episode
    parser.add_argument('--episode_length', default=1024, type=int)  # 每个episode采样多少transition
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--model_episode', default=0, type=int)
    parser.add_argument('--train_redo', default=False, type=bool)
    parser.add_argument('--run_redo', default=None, type=int)

    # algo
    parser.add_argument('--output_activation', default='softmax', type=str, help='tanh/softmax')
    parser.add_argument('--buffer_size', default=int(1e6), type=int)  # buffer大小
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--lr_a', default=0.0001, type=float)  # 好像没用到
    parser.add_argument('--lr_c', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)  # 好像没用到
    parser.add_argument('--epsilon_end', default=0.05, type=float)
    parser.add_argument('--hidden_size', default=128, type=int)  # 隐层宽度
    parser.add_argument('--target_replace', default=500, type=int)

    # seed
    parser.add_argument('--seed_nn', default=1, type=int)
    parser.add_argument('--seed_np', default=1, type=int)
    parser.add_argument('--seed_random', default=1, type=int)

    # evaluation 好像没用到
    parser.add_argument('--evaluate_rate', default=50)

    args = parser.parse_args()

    return args