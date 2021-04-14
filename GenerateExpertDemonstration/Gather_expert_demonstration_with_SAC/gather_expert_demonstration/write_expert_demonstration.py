import numpy as np

# env_name = ["InvertedDoublePendulum-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"]
env_name = ["HalfCheetah-v2"]
for env in env_name:
    expert_data_path = "./expert_demonstration_data" \
                       "/model_guide/deterministic_SAC_{}(6000)_johnny.npz".format(env)
    expert_data = np.load(expert_data_path)
    returns = expert_data["returns"]
    mean = np.mean(returns)
    std = np.std(returns)
    print("env:", env)
    print("mean:{}\nstd:{}".format(mean,std))