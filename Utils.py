from Constants import *
import math
def MC_dis_consumption(MC_capacity, distance)->float:
    remain_capacity = MC_capacity - (DISTANCE_CONSUMPTION * distance)
    return remain_capacity 

def cal_distance(Pos1, Pos2)->float:
        distance = math.sqrt((Pos1[0]-Pos2[0])**2 + (Pos1[1]-Pos2[1])**2)
        return distance

def cal_time(distance)->float:
    time = math.ceil(distance / VELOCITY)
    return time
      