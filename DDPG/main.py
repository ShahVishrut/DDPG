#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
import gym_carla
import carla
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

def main():
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 4000,  # connection port
        'town': 'Town03',  # which town to simulate
        'max_time_episode': 50,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
    }

  # Set gym-carla environment
    experiment= 'carla-v0' #specify environments here
    env= gym.make(experiment, params=params)
    steps= params['max_time_episode'] #steps per episode
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = 327680
    num_actions = env.action_space.shape[0]    
    print("Number of States:", num_states)
    print("Number of Actions:", num_actions)
    print("Number of Steps per episode:", steps)
    #saving reward:
    reward_st = np.array([0])
      
    
    for i in range(episodes):
        print("==== Starting episode no:",i,"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            #rendering environmet (optional)            
            #env.render()
            x = np.reshape(observation, [num_states])
            action = agent.evaluate_actor(np.reshape(x, [1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            print("Action at step", t ," :",action,"\n")
            
            observation,reward,done,info=env.step(action)
            observation = np.reshape(observation, [num_states])
            
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done or (t == steps-1)):
                print('EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
                print("Printing reward to file")
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print('\n\n')
                break
    total_reward+=reward_per_episode            
    print("Average reward per episode {}".format(total_reward / episodes))


if __name__ == '__main__':
    main()    
