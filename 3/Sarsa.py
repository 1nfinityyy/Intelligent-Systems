import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
MAX=500
infinity=np.Infinity
class Sarsa():
    def __init__(self, alpha, epsilon, gamma, env='CliffWalking-v0'): # 0 up, 1 right, 2 down, 3 left
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make(env)
        self.action_num = 4
        self.state_num = 48
        self.Q = np.zeros((48,4))
    def choose(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def Play(self):
        lst = []
        for i in range(47):
            for j in range(4):
                self.Q[i,j]=random.random()
        for episode in range(MAX):
            state = self.env.reset()
            action = self.choose(state)
            done = False
            sreward = 0
            while not done:
                obs, reward, done, info = self.env.step(action)
                sreward += reward
                next_action = self.choose(obs)
                self.Q[state,action]+=self.alpha * (reward + self.gamma * self.Q[obs, next_action] - self.Q[state, action])
                state, action = obs, next_action
            lst.append(sreward)
        self.env.close()
        return lst

class Q_learing():
    def __init__(self, alpha, epsilon, gamma, env='CliffWalking-v0'): # 0 up, 1 right, 2 down, 3 left
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make(env)
        self.action_num = 4
        self.state_num = 48
        self.Q = np.zeros((48,4))
    def choose(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    def Play(self):
        for i in range(47):
            for j in range(4):
                self.Q[i,j]=random.random()
        lst = []
        for episode in range(MAX):
            state = self.env.reset()
            action = self.choose(state)
            done = False
            sreward = 0
            while not done:
                obs, reward, done, info = self.env.step(action)
                sreward += reward
                self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[obs]) - self.Q[state, action])
                state = obs
                action = self.choose(state)
            lst.append(sreward)
        self.env.close()
        return lst

class nSarsa():
    def __init__(self, alpha, epsilon, gamma,n, env='CliffWalking-v0'):  # 0 up, 1 right, 2 down, 3 left
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make(env)
        self.action_num = 4
        self.state_num = 48
        self.n=n
        self.Q = np.zeros((48, 4))
    def choose(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    def Play(self):
        lst=[]
        for i in range(MAX):
            state = self.env.reset()
            action = self.choose(state)
            done = False
            state_list, action_list, reward_list = [state], [action], [0]
            T=infinity
            t=0
            while True:
                if t<T:
                    obs, reward, done, info = self.env.step(action_list[-1])
                    state_list.append(obs)
                    reward_list.append(reward)
                    if done:
                        T=t+1
                    else:
                        action_list.append(self.choose(state_list[-1]))
                temp=t-self.n+1
                if temp>0:
                    G = 0
                    for i in range(temp + 1, min(temp + self.n, T) + 1):
                        G += self.gamma ** (i - temp - 1) * reward_list[i-1]
                    if temp + self.n < T:
                        G += self.gamma ** self.n * self.Q[state_list[temp + self.n], action_list[temp + self.n]]#不用-1,因为A是从0开始的
                        s, a = state_list[temp], action_list[temp]
                        self.Q[s, a] += self.alpha * (G - self.Q[s, a])
                if temp == T - 1:
                    break
                t+=1
            lst.append(sum(reward_list))
        self.env.close()
        return lst

class Sarsalamda():
    def __init__(self, alpha, epsilon, gamma,lamda, env='CliffWalking-v0'):  # 0 up, 1 right, 2 down, 3 left
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.env = gym.make(env)
        self.action_num = 4
        self.state_num = 48
        self.lamda = lamda
        self.Q = np.zeros((48, 4))

    def choose(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    def Play(self):
        lst=[]
        for i in range(MAX):
            Z=np.zeros((48, 4))
            state = self.env.reset()
            action = self.env.action_space.sample()
            done = False
            sreward=0
            while not done:
                obs, reward, done, info = self.env.step(action)
                next_action=self.choose(obs)
                delta=reward+self.gamma*self.Q[obs,next_action] - self.Q[state,action]
                Z[state, action] += 1
                sreward+=reward
                for i in range(48):
                    for j in range(4):
                        self.Q[i,j]+=self.alpha*delta*Z[i,j]
                        Z[i,j]*=(self.gamma*self.lamda)
                state, action = obs, next_action
            lst.append(sreward)
        self.env.close()
        return lst


def process(lst):
    length = len(lst)
    res = []
    n = 10
    for i in range(0,length-n):
        res.append(sum(lst[i:i+n])/n)
    return res

sa=Sarsa(0.25,0.1,0.8)
reward1=sa.Play()
reward1=process(reward1)
plt.plot(range(len(reward1)),reward1,label='Sarsa')
QL=Q_learing(0.25,0.1,0.8)
rewardQ=QL.Play()
rewardQ=process(rewardQ)
plt.plot(range(len(rewardQ)),rewardQ,label='Q')
plt.xlabel("Episodes")
plt.ylabel("Sum of\nrewards\nduring\nepisode\n",linespacing=2,position=(-30,0.3),rotation=0)
plt.title("α=0.25, γ=0.8, Rewards of 2 algorithm")
# plt.legend()
plt.show()

# for n in [1,3,5]:
#     nstep_Sarsa = nSarsa(1,0.1,0.9,n)
#     rn = nstep_Sarsa.Play()
#     rn = process(rn)
#     plt.plot(range(len(rn)),rn,label='nSarsa n='+str(n))
# plt.xlabel("Episodes")
# plt.ylabel("Sum of\nrewards\nduring\nepisode\n",linespacing=2,position=(20,0.3),rotation=0)
# plt.title("α=1, γ=0.9, Rewards of n step Sarsa when n=1,3,5")
# plt.legend()
# plt.show()
#
# for l in [0,0.5,1]:
#     Sarsal = Sarsalamda(0.7,0.1,0.9,l)
#     rl = Sarsal.Play()
#     rl = process(rl)
#     plt.plot(range(len(rl)),rl,label='Sarsa(λ) λ='+str(l))
# plt.xlabel("Episodes")
# plt.ylabel("Sum of\nrewards\nduring\nepisode\n",linespacing=2,position=(20,0.3),rotation=0)
# plt.title("α=0.7, γ=0.9, Rewards of Sarsa(λ) when λ=0,0.5,1")
# plt.legend()
# plt.show()

