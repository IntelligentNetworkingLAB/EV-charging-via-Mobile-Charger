from stable_baselines3 import PPO, A2C
import Simulation

if __name__ == '__main__':
    env = Simulation.Agent()
    model = PPO.load("./model/timeslot/PPO_4.zip")
    #model = A2C.load("./model/A2C/A2C_1.zip")
    state = env.reset()
    while True:
        action, _ = model.predict(state[0], deterministic=False)
        info = env.step(action) 
        #print(state[0])
        state = info

        if info[2] or info[3]:
            print(info[1])

            break
