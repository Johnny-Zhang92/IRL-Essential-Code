#!/usr/bin/env bash

#python main.py --env_name  "Ant-v2" --actor_path  "./models/Ant/sac_actor_Ant" \
#--critic_path  "./models/Ant/sac_critic_Ant"
#
#python  main.py --env_name "HalfCheetah-v2" --actor_path "./models/HalfCheetah/sac_actor_HalfCheetah" \
#--critic_path  "./models/HalfCheetah/sac_critic_HalfCheetah"
#
##
#python  main.py --env_name "HalfCheetah-v2" --actor_path "./models/HalfCheetah/sac_actor_HalfCheetah(6000)" \
#--critic_path  "./models/HalfCheetah/sac_critic_HalfCheetah(6000)"
#
#
#python  main.py --env_name "Humanoid-v2" --actor_path "./models/Humanoid/sac_actor_Humanoid" \
#--critic_path  "./models/Humanoid/sac_critic_Humanoid"
#
#python  main.py --env_name "Walker2d-v2" --actor_path "./models/Walker2d/sac_actor_Walker2d" \
#--critic_path "./models/Walker2d/sac_critic_Walker2d"
#
#
#python  main.py --env_name "MountainCarContinuous-v0" --actor_path "./models/MountainCarContinuous/sac_actor_MountainCarContinuous" \
#--critic_path "./models/MountainCarContinuous/sac_critic_MountainCarContinuous"
#
#python  main.py --env_name "InvertedDoublePendulum-v2" --actor_path "./models/InvertedDoublePendulum/sac_actor_InvertedDoublePendulum" \
#--critic_path  "./models/InvertedDoublePendulum/sac_critic_InvertedDoublePendulum"
#
#python  main.py --env_name "Reacher-v2" --actor_path "./models/Reacher/sac_actor_Reacher" \
#--critic_path  "./models/Reacher/sac_critic_Reacher"
#
#python  main.py --env_name "InvertedPendulum-v2" --actor_path "./models/InvertedPendulum/sac_actor_InvertedPendulum" \
#--critic_path  "./models/InvertedPendulum/sac_critic_InvertedPendulum"

python  main.py --env_name "Hopper-v2" --actor_path "./models/Hopper/sac_actor_Hopper(2000)" \
--critic_path  "./models/Hopper/sac_critic_Hopper(2000)"