import gym
from gail import gail


if __name__ == '__main__':
    # * make the performance improve evidently
    env = gym.make('CartPole-v0')
    file = open('./traj/cartpole.pkl', 'rb')
    test = gail(
        env=env,
        episode=10000000,
        capacity=1000,
        gamma=0.99,
        lam=0.95,
        is_disc=True,
        value_learning_rate=3e-4,
        policy_learning_rate=3e-4,
        discriminator_learning_rate=3e-4,
        batch_size=64,
        file=file,
        policy_iter=1,
        disc_iter=10,
        value_iter=1,
        epsilon=0.2,
        entropy_weight=1e-4,
        train_iter=500,
        clip_grad=40,
        render=False
    )
    test.run()