#from smac.env import StarCraft2Env
from starcraft2 import StarCraft2Env
import numpy as np
from Neural_Network import DQNAgent,DoubleQAgent,REINFORCE,A2C
from memory import Memory
from torch.distributions import Categorical

def create_ai (type_of_ai,map_name,n_agents,state_size,n_actions):
    agents = {}
    if map_name == "3s_vs_3z" and state_size == 36 and type_of_ai == "DQN":
        for i in range(n_agents):
            agents[i] = DQNAgent(state_size,n_actions)
        return agents #agent0, agent1, agent2
    elif map_name == "3s_vs_3z" and state_size == 36 and type_of_ai == "DoubleQ":
        for i in range(n_agents):
            agents[i] = DoubleQAgent(state_size,n_actions)
        return agents #agent0, agent1, agent2
    elif map_name == "3s_vs_3z" and state_size == 36 and type_of_ai == "REINFORCE":
        for i in range(n_agents):
            agents[i] = REINFORCE(state_size,n_actions)
        return agents #agent0, agent1, agent2
    elif map_name == "3s_vs_3z" and state_size == 36 and type_of_ai == "A2C":
        for i in range(n_agents):
            agents[i] = A2C(state_size,n_actions)
        return agents #agent0, agent1, agent2
    elif map_name == "3s_vs_3z" and state_size == 36 and type_of_ai == "AC":
        for i in range(n_agents):
            agents[i] = ActorCritic()
        return agents#agent0, agent1, agent2
    
        
def get_state_each_agent(env,n_agents):
    avail_actions = {}
    avail_actions_ind = {}
    states = {}
    for agent_id in range(n_agents):
        states[agent_id] = np.array([(env.get_obs_agent(agent_id))])
        avail_actions = env.get_avail_agent_actions(agent_id)
        avail_actions_ind[agent_id] = np.nonzero(avail_actions)[0]
    return states, avail_actions_ind 

def get_action_for_each_agent(type_of_ai,agents,n_agents,states_s0,avail_actions_ind):
    actions = []
    if type_of_ai == "DQN":
        for agent_id in range(n_agents):
            action = []
            #print(agents[agent_id])
            action = agents[agent_id].act(states_s0[agent_id])
            if action not in avail_actions_ind[agent_id]:
                action = np.random.choice(avail_actions_ind[agent_id])
            actions.append(action)
    elif type_of_ai == "DoubleQ":
        for agent_id in range(n_agents):
            action = []
            #print(agents[agent_id])
            action = agents[agent_id].act(states_s0[agent_id])
            if action not in avail_actions_ind[agent_id]:
                action = np.random.choice(avail_actions_ind[agent_id])
            actions.append(action)
    elif type_of_ai == "REINFORCE":
        for agent_id in range(n_agents):
            action = []
            #print(agents[agent_id])
            action = agents[agent_id].select_action(states_s0[agent_id])
            actions.append(action)
            """
            try:
                action = agents[agent_id].select_action(states_s0[agent_id])
            except:
                pass"""
            
            """
            if action not in avail_actions_ind[agent_id]:
                action = np.random.choice(avail_actions_ind[agent_id])
            actions.append(action)
            """
    elif type_of_ai == "A2C":
        for agent_id in range(n_agents):
            action = []
            #print(agents[agent_id])
            #print(avail_actions_ind[agent_id])
            action = agents[agent_id].select_action(states_s0[agent_id])
            #print(action,"++++")
            actions.append(action)
            """
            try:
                action = agents[agent_id].select_action(states_s0[agent_id])
            except:
                pass"""
            """
            if action not in avail_actions_ind[agent_id]:
                action = np.random.choice(avail_actions_ind[agent_id])
            actions.append(action)
            """
    elif type_of_ai == "Qtable":
        pass

    return actions

def get_rewards(reward,n_agents):
    rewards = {}
    for agent_id in range(n_agents):
        rewards[agent_id] = reward
    return rewards

def save_to_experience_replay(agents,n_agents,states_s0,actions,rewards,states_s1,terminated):
    for agent_id in range(n_agents):
        agents[agent_id].remember(states_s0[agent_id],actions[agent_id],rewards[agent_id],states_s1[agent_id],terminated)
        
def learn(type_of_ai,n_agents,agents,memory):
    if type_of_ai == "DQN":
        for agent_id in range(n_agents):
            agents[agent_id].replay(30)
    elif type_of_ai == "DoubleQ":
        for agent_id in range(n_agents):
            agents[agent_id].replay(30)
    elif type_of_ai == "REINFORCE":
        for agent_id in range(n_agents):
            agents[agent_id].train()
    elif type_of_ai == "A2C":
        for agent_id in range(n_agents):
            agents[agent_id].train()
    elif type_of_ai == "Qtable":
        pass
 
def run_3s_vs_3z(number_of_episodes, AI="DQN"):
    #setup environment
    map_name= "3s_vs_3z"
    memory = Memory(max_size=300)
    reward_only_positive = True
    env = StarCraft2Env(map_name=map_name,reward_sparse = False,reward_only_positive = reward_only_positive, move_amount = 2, continuing_episode = False, obs_terrain_height = False, episode_limit = 750) #8m
    env_info = env.get_env_info()
    n_episodes = number_of_episodes
    #observation from environment
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    #n_map_limit = env_info["limit"]
    #print("!!!!!!!!!!!!!!!!!!!!!n_map_limit =", env_info)
    #print(n_actions)
    #print(n_agents)
    n_episodes = number_of_episodes
    #initial parameters
    state_size = 36
    best_avg_reward = -np.inf
    samp_rewards = []
    avg_rewards = []
    avg_reward = -np.inf
    #create AI
    AI_type = AI #DQN,Qtable
    agents = create_ai(AI_type,map_name,n_agents,state_size,n_actions)

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        Running_away_from_battle = 0
        while not terminated:
            #obs = env.get_obs() # global state havn't use it yet
            #state = env.get_state()
            
            #get state s0 --> model(NN) --> get action --> environment --> get state s1, reward
            states_s0, available_action_s0 = get_state_each_agent(env,n_agents)
            
            actions = get_action_for_each_agent(AI_type,agents,n_agents,states_s0,available_action_s0)
            #print(actions,"---")
            reward, terminated, _, Running_away_from_battle = env.step(actions) #actions in list
            #print(Running_away_from_battle)
            episode_reward += reward
            #print("reward per step =",reward)

            rewards = get_rewards(reward,n_agents)
            states_s1, available_action_s1 = get_state_each_agent(env,n_agents)
            save_to_experience_replay(agents,n_agents,states_s0,actions,rewards,states_s1,terminated)
            states_s0 = states_s1
            # learn without memory
            #print(np.array(states_s0[0]))
            """
            if AI_type == "REINFORCE":
                learn(AI_type,n_agents,agents,memory)"""
                
        learn(AI_type,n_agents,agents,memory)
        if Running_away_from_battle == 1:
            episode_reward -= 20
        samp_rewards.append(episode_reward)
        if (e >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards[-100:])
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        print("Total reward in episode {} = {}".format(e, episode_reward))
        print("Best_avg_reward =", np.round(best_avg_reward,3),"Average_rewards =", np.round(avg_reward,3))
    env.save_replay()
    env.close()

    return avg_rewards, best_avg_reward,samp_rewards