#from smac.env import StarCraft2Env
from starcraft2 import StarCraft2Env
import numpy as np
from Neural_Network import DQNAgent,DoubleQAgent,A2C,REINFORCE
from memory import Memory
from run_ai import run_3s_vs_3z
from torch.distributions import Categorical
import matplotlib.pyplot as plt

def main(): #DQN,Qtable,AC,DoubleQ,REINFORCE,A2C
    AI1 = ["A2C", "REINFORCE"]#,"REINFORCE"
    N_episodes = 200
    for AI in AI1:
        
        avg_reward, best_avg_reward,samp_rewards = run_3s_vs_3z(N_episodes, AI) 
        #print(avg_reward)
        t = np.arange(1,len(samp_rewards)+1,1) 
        plt.figure('Episode Rwards of '+ AI)
        plt.plot(t, samp_rewards)
        plt.xlabel('time_step')
        plt.ylabel('Episode Rwards of' + AI)
        plt.title(AI)
        plt.show()


    
if __name__ == "__main__":
    main()
    
    
#test dimension 
# 1. sprase rewards vs reward per kill
