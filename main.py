import tensorflow as tf
from tictactoe import *
from nnet import *
from collections import defaultdict
import pandas as pd

memories = pd.DataFrame(data = {'s1' : [[1,1]], 'a' : 5, 'r' : 3, 's2' : [[4,4]]},columns=['s1','a','r','s2'])

print memories

qtable = defaultdict(lambda:[1.0 for _ in range(9)])
alpha = 0.3
gamma = 0.8

def lerp(a,b,c):
    return a*c + b*(1.0-c)

def reformat(state):
    return tuple(state)

def choose(q,choices):
    q_avail = np.asarray(q)[choices]
    print q_avail
    return choices[np.argmax(q_avail)]

if __name__ == "__main__":

    # test with just dense layers
    net = Net(1, (3,3,1), (9,))
    #net.append(ConvolutionLayer(patch_size,1,16,activation='relu')) #depth : 3 -> 16
    #net.append(ConvolutionLayer(patch_size,16,16,activation='relu')) #depth : 16 -> 16
    #net.append(DropoutLayer(0.5))
    net.append(DenseLayer((9, 16), activation='relu'))
    net.append(DenseLayer((16, 9),activation='none'))
    net.setup()

    # roughly something like ...

    # train
    n_epoch = 100
    for epoch in range(n_epoch):
        print 'e', epoch
        ttt = TicTacToe() # NEW BOARD
        while True:
            s1 = ttt.sense()
            choices = ttt.available() #np.nonzero(s1==X) #available actions
            
            if len(choices) == 0: # draw
                break
            a = np.random.choice(choices)
            r = ttt.act(a)
            s2 = ttt.sense()
            #maxqn = -max(qtable[s2]) #negate opponent's1 best, expectimax
            # maxqn = -max(net.predict(s2))
            #qtable[s1][a] = lerp(r+gamma*maxqn,qtable[s1][a],alpha)

            memories.loc[epoch] = [s1,a,r,s2]

            if r != 0.0: #win/loss
                break

        #print memories
        #print memories['s2'].as_matrix()
        net.predict(memories['s2'].as_matrix())
        q_new = memories['r'] + gamma * -np.max(net.predict(memories['s2'].as_matrix()),1)
        net.run(memories['s1'], q_new)
        
        #net.run(memories['s1'],lerp(memories['r'] + gamma * -np.max(qtable[memories['s2']],1), qtable[memories[s1]][a],alpha))

    print "LEN : " ,len(qtable.keys())
# 
#     # test
#     while raw_input("Game? [y/n]") == 'y':
#         ttt = TicTacToe()
#         while True:
#             print ttt.board
#             s1 = ttt.sense()
#             choices = ttt.available() #np.nonzero(s1==X) #available actions
#             if len(choices) == 0: # draw
#                 break
#             a = choose(qtable[s1], choices)
#             if ttt.turn == A:
#                 a = int(raw_input("action :"))
#             else:
#                 a = choose(qtable[s1],choices)
#             #a = np.argmax(qtable[s1])
#             r = ttt.act(a)
#             if r != 0.0: #win/loss
#                 break
