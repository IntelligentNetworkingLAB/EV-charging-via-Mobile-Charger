import os
import numpy as np
import math
from typing import Callable
from stable_baselines3 import PPO, A2C, DDPG, DQN
from gymnasium import spaces
import gymnasium
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from Constants import *
import Utils 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 클래스 구현
class Agent(gymnasium.Env): # 환경 초기화 코드 작성
    def __init__(self):
        self.time = 0
        self.CS_location = [0,150]
        self.EV_location = [[50,50],[50,100],[50,150],[50,200],[250,200],[250,150],[250,100],[250,50]]
        self.EV_MAX_Battery = [77.4, 77.4, 87.2, 77.4, 85, 105, 105, 84.96] # EV 배터리 용량
        #self.EV_Capacity = [15.48, 15.48, 17.44, 15.48, 17, 21, 21, 16.992] #초기 20% 배터리
        self.EV_Capacity = [0,0,0,0,0,0,0,0]
        self.EV_Efficiency = [5.2, 6.2, 4.3, 4.6, 5.4, 4.3, 4.1, 5.1] # 각 EV 연비
        #self.EV_driving = [80.496, 95.976, 74.992, 71.208, 91.8, 90.3, 86.1, 86.6592] # 초기 주행 가능 거리
        self.EV_driving = [0,0,0,0,0,0,0,0]
        self.MC_Capacity = MC_CAPACITY
        self.previous_status = 8
        obs_length = (EV_NUMBER * 2) + 4   
        self.observation_space = spaces.Box(low=0 , high=36500 ,shape=(obs_length, ), dtype=np.float32)
        self.action_space = spaces.Discrete(9, start = 0) #8대의 차 + 1곳의 CS      
        self.overcharge_count = 0
        self.penalty = 0
        self.terminated = False
        
    def get_state(self):
        battery_status = np.array(self.EV_Capacity).flatten()
        distance_status = np.array(self.EV_driving).flatten()
        #efficiency_status = np.array(self.EV_Efficiency).flatten()
        state = np.concatenate(( distance_status, battery_status,
                                [self.previous_status], [self.MC_Capacity], [DESTINATION], [self.time]))
        return state

    def reset(self, seed: Optional[int] = None): # 에피소드 초기화 코드 작성
        self.Step = 0
        self.Reward = 0
        self.time = 0 
        self.Penalty_count = 0
        self.overcharge_count = 0
        self.penalty = 0
        #self.EV_Capacity = [15.48, 15.48, 17.44, 15.48, 17, 21, 21, 16.992] #초기 20% 배터리
        #self.EV_driving = [80.496, 95.976, 74.992, 71.208, 91.8, 90.3, 86.1, 86.6592]
        self.EV_Capacity = [0,0,0,0,0,0,0,0]
        self.EV_driving = [0,0,0,0,0,0,0,0]
        self.MC_Capacity = MC_CAPACITY 

        self.previous_status = 8
        self.Obj_flag =False #성공 실패 확인
        self.terminated = False
        self.truncated = False
        
        return self.get_state(), {}
    
    
    def step(self, action):
        if action == 8: #action에 따른 이동 에너지 & time 변화
            if action == self.previous_status:
                self.time += MINUTE
            elif action != self.previous_status:
                distance = Utils.cal_distance(self.CS_location, self.EV_location[self.previous_status])
                self.MC_Capacity = Utils.MC_dis_consumption(self.MC_Capacity, distance)
                drive_time = Utils.cal_time(distance)
                self.time += drive_time + MINUTE
        else:
            if action == self.previous_status:
                self.time += MINUTE
            elif action != self.previous_status:
                if self.previous_status == 8:
                    distance = Utils.cal_distance(self.CS_location, self.EV_location[action])
                    self.MC_Capacity = Utils.MC_dis_consumption(self.MC_Capacity, distance)
                    drive_time = Utils.cal_time(distance)
                    self.time += drive_time + MINUTE
                else:
                    distance = Utils.cal_distance(self.EV_location[self.previous_status], self.EV_location[action])
                    self.MC_Capacity = Utils.MC_dis_consumption(self.MC_Capacity, distance)
                    drive_time = Utils.cal_time(distance)
                    self.time += drive_time + MINUTE
            
        if action == 8: #action에 따른 충전
            self.MC_Capacity += CS_CHARGING_POWER
            self.previous_status = action
        else:
            self.EV_Capacity[action] += MC_CHARGING_POWER
            self.MC_Capacity -= MC_CHARGING_POWER
            self.previous_status = action
        
        for i in range(EV_NUMBER):
            self.EV_driving[i] = self.EV_Capacity[i] * self.EV_Efficiency[i]


        #self.Reward = sum(self.EV_driving)/1000
        #self.Reward = sum(self.EV_driving) / np.std(self.EV_driving)
        #self.Reward = sum(self.EV_driving) / np.var(self.EV_driving)
        #self.Reward = sum(self.EV_driving) / abs(np.var(self.EV_driving))
        ########################제약사항#################################
        for i in range(EV_NUMBER): #과충전 패널티
            if self.EV_Capacity[i] > self.EV_MAX_Battery[i]:
                self.EV_Capacity[i] = self.EV_MAX_Battery[i]
                self.Reward = OVERCHARGE_PENALTY
                #print("EV Overcharge")
        
        if self.MC_Capacity > MC_CAPACITY:
            self.MC_Capacity = MC_CAPACITY 
            self.Reward = OVERCHARGE_PENALTY
            self.overcharge_count += 1
            if self.overcharge_count > 50:
                print("Overheat")
                self.Reward = MC_DIED
                self.terminated = True
            #print("MC Overcharge")

        if self.MC_Capacity <= 0: #MC 쥬금
                self.Reward = MC_DIED  
                self.terminated = True
                print("MC Died")
        ################################################################
        if self.time >= MAX_TIME:
            for i in range(EV_NUMBER):
                if DESTINATION > self.EV_driving[i]:
                    self.penalty +=(self.EV_driving[i] - DESTINATION)
                    self.Obj_flag = True 

            if self.Obj_flag == True:
                    self.Reward = self.penalty
                    #self.Reward = self.achieve_goal + self.de_achieve_goal
                    #self.Reward = MC_DIED * 2
                    print("Fail to Achieve Goal")
                    print("Distance", self.EV_driving, self.MC_Capacity)
            else:
                #self.Reward = sum(self.EV_driving) / abs(np.std(self.EV_driving))
                
                #self.Reward = sum(self.EV_driving)/ abs(np.std(self.EV_driving))
                #self.Reward = sum(self.EV_driving)
                #self.Reward = np.mean(self.EV_driving) 
                self.Reward = sum(self.EV_driving)/ abs(np.std(self.EV_driving))
                
                print("##################################")
                print("Success!, Reward = ", self.Reward)
                print("Distance", self.EV_driving, self.MC_Capacity, abs(np.std(self.EV_driving)))
                print("##################################")
            self.terminated =True
        #############################################################
        return self.get_state(), self.Reward, self.terminated, self.truncated, {}


env = Monitor(Agent())

model = PPO("MlpPolicy", env, tensorboard_log="./model/timeslot/PPO", device="cpu", learning_rate= 5e-5)

model.learn(total_timesteps=4e6)
model.save(f"./model/timeslot/PPO")

"""
# A2C 에이전트 초기화
model = A2C("MlpPolicy",  
            env, tensorboard_log="./model/A2C", learning_rate=5e-5)


model.learn(total_timesteps=8e6)
        
model.save(f"./model/A2C/A2C")
"""

