#!/usr/bin/env bash

#cd gather_expert_demonstration/

# generate Hopper-v2 expert demonstration
#python -m gather_expert_demonstration.generate_expert_trajectory \
#       --env_name "Hopper-v2" \
#       --actor_path "./models/Hopper/sac_actor_Hopper-v2_" \
#       --critic_path  "./models/Hopper/sac_critic_Hopper-v2_"
## Average length: 1000.0
## Average return: 3450.0

#generate Hopper-v2 expert demonstration return:2100
python -m gather_expert_demonstration.generate_expert_trajectory \
       --env_name "Hopper-v2" \
       --actor_path "./models/Hopper/sac_actor_Hopper-v2_(2100)" \
       --critic_path  "./models/Hopper/sac_critic_Hopper-v2_(2100)"
# Average length: 1000.0
# Average return: 3450.0


## generate ANt-v2 expert demonstration
#python -m gather_expert_demonstration.generate_expert_trajectory \
#       --env_name "Ant-v2" \
#       --actor_path "./models/Ant/sac_actor_Ant" \
#       --critic_path  "./models/Ant/sac_critic_Ant"
##Average length: 1000.0
##Average return: 5688.6615694237635


# generate HalfCheetah-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "HalfCheetah-v2" \
#        --actor_path "./models/HalfCheetah/sac_actor_HalfCheetah " \
#        --critic_path  "./models/HalfCheetah/sac_critic_HalfCheetah"
##Average length: 1000.0
##Average return: 11588.00914054808


#sac_actor_HalfCheetah(6000)
# generate HalfCheetah-v2 expert demonstration Average return:6000
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "HalfCheetah-v2" \
#        --actor_path "./models/HalfCheetah/sac_actor_HalfCheetah(6000)" \
#        --critic_path  "./models/HalfCheetah/sac_critic_HalfCheetah(6000)"


## generate Humanoid-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "Humanoid-v2" \
#        --actor_path "./models/Humanoid/sac_actor_Humanoid" \
#        --critic_path  "./models/Humanoid/sac_critic_Humanoid"




## generate Walker2d-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "Walker2d-v2" \
#        --actor_path "./models/Walker2d/sac_actor_Walker2d" \
#        --critic_path "./models/Walker2d/sac_critic_Walker2d"
##Average length: 1000.0
##Average return: 4913.904440835768

## generate Walker2d-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "MountainCarContinuous-v0" \
#        --actor_path "./models/MountainCarContinuous/sac_actor_MountainCarContinuous" \
#        --critic_path "./models/MountainCarContinuous/sac_critic_MountainCarContinuous"

##generate Walker2d-v2 expert demonstration
#python  -m gather_expert_demonstration.generate_expert_trajectory \
#        --env_name "InvertedPendulum-v2" \
#        --actor_path "./models/InvertedPendulum/sac_actor_InvertedPendulum" \
#        --critic_path "./models/InvertedPendulum/sac_critic_InvertedPendulum"