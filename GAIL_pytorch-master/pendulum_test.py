import gym
from gail import gail


if __name__ == '__main__':
    # * the GAIL doesn't perform well in continuous case
    # * (maybe only in this case under these hyperparameters)
    # * exist ocillation phenomenon and can't converge
    env = gym.make('Pendulum-v0')
    file = open('./traj/pendulum.pkl', 'rb')
    test = gail(
        env=env,
        episode=10000000,
        capacity=1000,
        gamma=0.99,
        lam=0.95,
        is_disc=False,
        value_learning_rate=1e-4,
        policy_learning_rate=1e-4,
        discriminator_learning_rate=3e-4,
        batch_size=64,
        file=file,
        policy_iter=3,
        disc_iter=10,
        value_iter=3,
        epsilon=0.05,
        entropy_weight=0,
        train_iter=600,
        clip_grad=0.2,
        render=False
    )
    test.run()