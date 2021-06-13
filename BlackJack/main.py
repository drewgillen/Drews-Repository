import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import env


env = env.BlackjackEnv()

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise

    ##Stick == Return 0
    ##Hit == Return 1

    '''
    if score <=18 & dealer_score == 1 & usable_ace == True:
        return 1

    if score >18 & dealer_score == 1 & usable_ace == True:
        return 0

    if score <= 17 & dealer_score <=8 & usable_ace == True:
        return 1
    
    if score >17 & dealer_score <=8 & usable_ace == True:
        return 0

    if score <= 18 & dealer_score <=10 & usable_ace == True:
        return 1
    
    if score >18 & dealer_score <=10 & usable_ace == True:
        return 0 




    if score <=16 & dealer_score ==1 & usable_ace == False:
        return 1 

    if score >16 & dealer_score ==1 & usable_ace == False:
        return 0 

    if score <=12 & dealer_score <=3 & usable_ace == False:
        return 1 

    if score >12 & dealer_score <=3 & usable_ace == False:
        return 0 

    if score <=11 & dealer_score <=6 & usable_ace == False:
        return 1 

    if score >11 & dealer_score <=6 & usable_ace == False:
        return 0 

    if score <=16 & dealer_score <=10 & usable_ace == False:
        return 1 

    if score >16 & dealer_score <=10 & usable_ace == False:
        return 0
    '''




    return 0 if score >= 18 else 1




def main():

    df = list()
    win = list()
    tie = list()
    #df = pd.DataFrame
    #df.header = (['Win', 'Loss'])

    for i_episode in range(100000):
        observation = env.reset()
        for t in range(100):
            print_observation(observation)
            action = strategy(observation)
            print("Taking action: {}".format( ["Stick", "Hit"][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print("Game end. Reward: {}\n".format(float(reward)))

                if (float(reward)) == 1 :
                    win.append(float(reward))
                if (float(reward)) == 0 :
                    tie.append(float(reward))
                break
    
    print('Wins:')
    print(len(win))
    print('Ties: ')
    print(len(tie))



if __name__ == '__main__':
    main()
