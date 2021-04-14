# Introduction

- Use SAC algorithm as agent , the file name is  [pythorch-SAC] :(./pythorch-SAC/)
- Agent model actor and critic store in the folder named models
- Write expert demonstration data , the file name is [write_expert_demonstration]: (./write_expert_demonstration)
- Generate expert episode/trajectory  data, the file name is [generate_expert_trajectory]:(./generate_expert_trajectory)
- The folder save  expert demonstration data , [expert_demonstration_data]:(./expert_demonstration_data)

# Process

1. Use SAC algorithm to train actor and critic network

   the file is `main.py` , save actor and critic network in `models/`

2. Load actor and critic network with file`generate_expert_trajectory.py` in `/Gather_expert_demonstration` fold.

   this file could generate and save expert demonstration as `.npz` format in path `/Gather_expert_demonstration/expert_demonstration_data`

    





































