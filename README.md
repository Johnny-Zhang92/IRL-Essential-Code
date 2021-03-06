# Description

**This is a GAIL baselines what belong to Inverse Reinforcement Learning (IRL) methods.** 

**As we all know GAN and GAIL are fragile, even the baseline code what is written by OpenAI is hard to train. Therefore,  I write a GAIL code which is PyTorch edition. Besides, Because of the fragility of GAIL, I add some trick in code, what is inevitable, and the tricks are as flows:**

# Ｍy_GAIL_PyThorch

## Requirements

- mujoco-py==2.0.2.13
- PyTorch==1.7.1
- See more details in requirement.txt
## Trick:

1. **Memory:** add a replay buffer to train generator, 
2. **Batch Normal:** using batch normal trick to transform  state , action and next state , note: this trick is used for train generator net ，instead of  discriminator net.
3. **Reward Function:** if generator accuracy less than 0.5,  then this indicates that the generator can not identify the generated data and exert data, thus the reward is optimal reward.  Conversely the reward equals to reward function generated by discriminator.
4. **Add noise :** add noise to discriminator

## Note:

1. The key to train GAIL is that balancing the discriminator and generator performance, a strong discriminator is not allowed, the discriminator should waiting for the generator.



# Usage

```bash
python main.py  --env_name=Hopper-v2
```

**note:** By this way, you can only change the ==environment name==, the other parameters  only can be changed in their ==yaml file==, the file path is =="./env_parser/"==.

# Runs



1. **Hopper-v2** (expert return = 3500)

![image-20210408143157754](README.assets/image-20210408143157754.png)



![image-20210414091849909](README.assets/image-20210414091849909.png)

2. **HalfCheetah-v2**(expert return = 6000)
 ![image-20210409142601820](README.assets/image-20210409142601820.png)
 
3. **Ant-v2** (expert return =5500 )

    ![image-20210412100841604](README.assets/image-20210412100841604.png)

4. **Walker2d-v2** (expert return = 4900)

![image-20210414102348203](README.assets/image-20210414102348203.png)

5. InvertedPendulum((expert return = 1000)

   ![image-20210413092110327](README.assets/image-20210413092110327.png)

6. InvertedDoublePendulum((expert return = 9359)
   ![image-20210414101914232](README.assets/image-20210414101914232.png)

# Generate Expert Demonstrations
This package can be used to generate expert demonstrations.

You can also download expert demonstration via link:
[Expert Demonstration](https://drive.google.com/drive/folders/1oMfjTrmIy3tPdnPrjEU7YfuxmBQeW3oz?usp=sharing)

# Reference

**[SAC(pytorch-soft-actor-critic-master)]**: https://github.com/pranz24/pytorch-soft-actor-critic

The websites of Four GAIL editions are as flows:

**[gail-pytorch]**:https://github.com/hcnoh/gail-pytorch.git

**[PyTorch-RL]**:https://github.com/Khrylx/PyTorch-RL.git

**[imitation]**:https://github.com/openai/imitation.git

**[GAIL]**:https://github.com/JiangengDong/GAIL.git

