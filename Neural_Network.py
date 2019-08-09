import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
#from pysc2.agents import base_agent
#from pysc2.lib import actions
#from pysc2.lib import features
#import gym
from collections import deque
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Activation, Input
#from keras.layers.merge import Add, Multiply
#import keras.backend as K
#from keras.optimizers import Adam
#import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 48)
        self.layer_3 = nn.Linear(48, action_dim)
        
    def forward(self,x):
        #print(x)
        #x = (torch.from_numpy(x))
        #print(x)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return x.data.numpy()

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=200)
        self.memory = ReplayBuffer()
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size,action_size).to(device)
        #self.optimizer = optim.RMSprop(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def remember(self, state, action, reward, next_state, done):

        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, x):
        x=torch.from_numpy(x)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.forward(x)#.cpu().data.numpy().flatten()
        return np.argmax(act_values[0])

    def replay(self, batch_size, iterations = 30, policy_freq = 2):
        if batch_size < len(self.memory.storage):
            for it in range (iterations):
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.memory.sample(batch_size)
    
                state = torch.Tensor(batch_states).to(device)
                next_state = torch.Tensor(batch_next_states).to(device)
                action = torch.Tensor(batch_actions).to(device)
                reward = torch.Tensor(batch_rewards).to(device)
                done = torch.Tensor(batch_dones).to(device)
                zeros = torch.zeros(batch_size, device=device)
                zeros = zeros.unsqueeze(1)
    
                action = action.long()
                Q_next_state = self.model.forward(next_state)
                Q_state = self.model.forward(state)
                #Q_target = Q_state
               
                target_of_action_in_the_state = reward.detach() if (done !=zeros).any() else (reward + self.gamma*np.max(Q_next_state[0])*(1-done)).detach()
                #target_of_action_in_the_state = (reward + self.gamma*np.max(Q_next_state[0])*(1-done)).detach()
                action = torch.unsqueeze(action,1)
                #action = action.repeat(1,9)
                
                state_action_values = self.model.forward(state)
                state_action_values = torch.from_numpy(state_action_values)
                state_action_values = state_action_values.view(30,9)
                state_action_values = state_action_values.gather(1,action)
                
                #Q_state = torch.from_numpy(Q_state)
                #Q_state = torch.tensor(Q_state, requires_grad=True)
                #Q_target = torch.from_numpy(Q_target)
                #Q_target = torch.tensor(Q_target, requires_grad=True)
                #Q_target = torch.squeeze(Q_target, 1)
                state_action_values = torch.tensor(state_action_values, requires_grad=True)
                state_action_values = torch.squeeze(state_action_values, 1)

    
                #loss = F.smooth_l1_loss(Q_state,Q_target.unsqueeze(1))
                #loss = F.mse_loss(Q_state,Q_target.unsqueeze(1))
                #loss = F.mse_loss(Q_state,state_action_values.unsqueeze(1))
                loss = F.mse_loss(state_action_values,target_of_action_in_the_state.unsqueeze(1))
                #print(state_action_values-target_of_action_in_the_state.unsqueeze(1))
                #print(Q_state-state_action_values.unsqueeze(1))
                #print(Q_state,"................................................")
                #print(state_action_values.unsqueeze(1))
                loss.requres_grad = True
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DoubleQAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DoubleQAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=200)
        self.memory = ReplayBuffer()
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.Q = DQN(state_size,action_size).to(device)
        self.Q_target = DQN(state_size,action_size).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters())
        self.Q_target_optimizer = torch.optim.Adam(self.Q_target.parameters())

    def remember(self, state, action, reward, next_state, done):

        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, x):
        x=torch.from_numpy(x)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.Q.forward(x)#.cpu().data.numpy().flatten()
        return np.argmax(act_values[0])

    def replay(self, batch_size, iterations = 30, policy_freq = 3):
        if batch_size < len(self.memory.storage):
            for it in range (iterations):
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.memory.sample(batch_size)
    
                state = torch.Tensor(batch_states).to(device)
                next_state = torch.Tensor(batch_next_states).to(device)
                action = torch.Tensor(batch_actions).to(device)
                reward = torch.Tensor(batch_rewards).to(device)
                done = torch.Tensor(batch_dones).to(device)
                zeros = torch.zeros(batch_size, device=device)
                zeros = zeros.unsqueeze(1)
    
                action = action.long()
                Q_next_state = self.Q_target.forward(next_state)
                Q_state = self.Q.forward(state)
                #Q_target = Q_state
               
                target_of_action_in_the_state = reward.detach() if (done !=zeros).any() else (reward + self.gamma*np.max(Q_next_state[0])*(1-done)).detach()
                #target_of_action_in_the_state = (reward + self.gamma*np.max(Q_next_state[0])*(1-done)).detach()
                target_of_action_in_the_state = torch.squeeze(target_of_action_in_the_state, 1)
                action = torch.unsqueeze(action,1)
                #action = action.repeat(1,9)
                
                state_action_values = self.Q.forward(state)
                state_action_values = torch.from_numpy(state_action_values)
                state_action_values = state_action_values.view(30,9)
                state_action_values = state_action_values.gather(1,action)
                
                #Q_state = torch.from_numpy(Q_state)
                #Q_state = torch.tensor(Q_state, requires_grad=True)
                #Q_target = torch.from_numpy(Q_target)
                #Q_target = torch.tensor(Q_target, requires_grad=True)
                #Q_target = torch.squeeze(Q_target, 1)
                state_action_values = torch.tensor(state_action_values, requires_grad=True)
                #state_action_values = torch.squeeze(state_action_values, 1)

    
                #loss = F.smooth_l1_loss(Q_state,Q_target.unsqueeze(1))
                #loss = F.mse_loss(Q_state,Q_target.unsqueeze(1))
                #loss = F.mse_loss(Q_state,state_action_values.unsqueeze(1))
                loss = F.mse_loss(state_action_values,target_of_action_in_the_state.unsqueeze(1))
                #print(state_action_values-target_of_action_in_the_state.unsqueeze(1))
                #print(Q_state-state_action_values.unsqueeze(1))
                #print(Q_state,"................................................")
                #print(state_action_values.unsqueeze(1))
                loss.requres_grad = True
                self.Q_optimizer.zero_grad()
                loss.backward()
                self.Q_optimizer.step()
                
                if (iterations%policy_freq) == 1:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class ReplayBuffer(object):
    
    def __init__(self,max_size = 300):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
    def add(self,transitions):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transitions
            self.ptr = (self.ptr+1)%self.max_size
        else: 
            self.storage.append(transitions)
            
    def sample(self, batch_size):
        ind = np.random.randint(0,len(self.storage),size = batch_size)
        batch_states, batch_actions, batch_rewards,batch_next_states, batch_dones = [],[],[],[],[]
        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            batch_states.append(np.array(state, copy = False))
            batch_next_states.append(np.array(next_state, copy = False))
            batch_actions.append(np.array(action, copy = False))
            batch_rewards.append(np.array(reward, copy = False))
            batch_dones.append(np.array(done, copy = False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        #self.dropout = nn.Dropout(p=0.1)
        self.layer_2 = nn.Linear(64, 48)
        self.layer_3 = nn.Linear(48, action_dim)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        #x = self.dropout(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        #action_scores = F.relu(self.layer_3(x))
        action_scores = F.relu(self.layer_3(x))
        
        #print(action_scores,"...............")
        #test = F.softmax(self.layer_3(x),dim=-1)
        #print(test)
        action_scores = F.softmax(action_scores, dim=1)
        return action_scores#F.softmax(action_scores, dim=1)
    
class REINFORCE(nn.Module):
    def __init__(self, state_size, action_size):
        super(REINFORCE, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model = Policy(state_size,action_size).to(device)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.eps = np.finfo(np.float32).eps.item()
        
    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        #print(state)
        probs = self.model.forward(state)
        #print(probs)
        m = Categorical(probs)
        #print(m)
        action = m.sample()
        self.model.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def train(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.model.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        
    def remember(self, state, action, reward, next_state, done):
        self.model.rewards.append(reward)
        
class ActorforA2C(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(ActorforA2C, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 48)
        self.layer_3 = nn.Linear(48, action_dim)
        self.saved_log_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        #x = F.relu(self.layer_1(x))
        #x = F.relu(self.layer_2(x))
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = F.relu(self.layer_3(x))
        return x
    
class CriticforA2C(nn.Module):
    
    def __init__(self, state_dim):
        super(CriticforA2C, self).__init__()
        # 1st critic NN
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 1)
        #self.layer_3 = nn.Linear(48, 1)#output 1 Q value
        self.values = []
    
    def forward(self, x):
        #concat
        #xu = torch.cat([x,u],1)#axis = 1 --> concat vertically
        x = torch.from_numpy(x).float().unsqueeze(0)
        #x1 = (self.layer_1(x))
        x1 = F.relu(self.layer_1(x))
        #x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_2(x1)
        #x1 = self.layer_3(x1)
        return x1

class A2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.learning_rate = 0.001
        self.eps = np.finfo(np.float32).eps.item()
        self.actor_model = ActorforA2C(state_size,action_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters())
        self.critic_model = CriticforA2C(state_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters())
        
    def select_action(self,x):
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        #print(state)
        probs = self.model.forward(state)
        #print(probs)
        m = Categorical(probs)
        #print(m)
        action = m.sample()
        """
        probs = self.actor_model.forward(x)
        
        m = Categorical(probs)
        action = m.sample()
        self.actor_model.saved_log_probs.append(m.log_prob(action))
        value = self.critic_model.forward(x)
        self.critic_model.values.append(value)
        return action.item()
    
    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns
    
    def train(self):
        next_state = self.actor_model.next_states[-1]
        #print("next_state = ",next_state)
        next_value = self.critic_model.forward(next_state)
        #print("next_value = ",next_value)
        rewards = self.actor_model.rewards
        masks = self.actor_model.dones
        returns = self.compute_returns(next_value, rewards, masks)
        #print("returns = ", returns)
        log_probs = self.actor_model.saved_log_probs
        log_probs = torch.cat(log_probs)
        #print("log_probs = ",log_probs)
        values = self.critic_model.values
        #print("values =", values)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values
        #print("logprobs =", log_probs)
        #print("advantages =", advantage.detach())
        #print("------------------------------",log_probs * advantage.detach())
        actor_loss = -(log_probs * advantage.detach()).mean()
        #print("actor_loss =", log_probs * advantage.detach().mean())
        critic_loss = F.smooth_l1_loss(returns,values)
        #critic_loss = advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        del self.actor_model.rewards[:]
        del self.actor_model.saved_log_probs[:]
        del self.actor_model.dones[:]
        del self.actor_model.next_states[:]
        del self.critic_model.values[:]
        
    def remember(self, state, action, reward, next_state, done):
        self.actor_model.rewards.append(reward)
        self.actor_model.dones.append(done)
        self.actor_model.next_states.append(next_state)

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self().__init__())
        self.layer_1 = nn.Linear(state_dim, 64)
        self.layer_2 = nn.Linear(64, 48)
        self.layer_3 = nn.Linear(48, action_dim)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action*torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self().__init__())
        # 1st critic NN
        self.layer_1 = nn.Linear(state_dim+action_dim, 64)
        self.layer_2 = nn.Linear(64, 48)
        self.layer_3 = nn.Linear(48, 1)#output 1 Q value
        # 2nd critic NN
        self.layer_4 = nn.Linear(state_dim+action_dim, 64)
        self.layer_5 = nn.Linear(64, 48)
        self.layer_6 = nn.Linear(48, 1)#output 1 Q value
   
    def forward(self, x, u):
        #concat
        xu = torch.cat([x,u],1)#axis = 1 --> concat vertically
        #Forward for 1st critic NN
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        #Forward for 2nd critic NN
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    
    def Q1(self, x, u):
        #concat
        xu = torch.cat([x,u],1)#axis = 1 --> concat vertically
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

#selecting the device

class TD3(object):
    
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.crtiic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor.forward(state).cpu().data.numpy().flatten()
        
    def train(self, replay_buffer, iterations, batch_size = 100, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
        
        for it in range(iterations):
            #sample a batch transitions from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # From the next state s', the actor target plays the next actions a'
            next_action = self.actor_target.forward(next_state)
            
            #add Guassian noise to this next action a' and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device)
            noise = noise.clamp(-noise_clip,noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            #The two Critic targets take each the couple (s', a') as input and return two Q-values Qt(s',a') and Qt2(s',a') as outputs
            target_Q1, target_Q2 = self.critic_target.forward(next_state, next_action)
            #get min of the two Q values
            target_Q = torch.min(target_Q1,target_Q2)
            # Final target of the two critic models (make it a target by using q_target = r+discount*(min(q1t,q2t)))
            target_Q = reward + ((1-done)*discount*target_Q).detach()

    
        
