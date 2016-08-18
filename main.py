import tensorflow as tf
from tictactoe import *
from nnet import *
from collections import defaultdict, deque
import pandas as pd

#memories = Queue.Queue(maxsize=1000)
memories = deque(maxlen=4096)

qtable = defaultdict(lambda:[1.0 for _ in range(9)])
alpha = 0.5
gamma = 0.9

arr = np.asarray

def lerp(a,b,c):
    return a*c + b*(1.0-c)

def reformat(state):
    return tuple(np.reshape(state, 9))

def choose(q,choices):
    q_avail = np.asarray(q)[choices]
    print q_avail
    return choices[np.argmax(q_avail)]

if __name__ == "__main__":

    # test with just dense layers
    net = Net()
    #net.append(ConvolutionLayer(patch_size,1,16,activation='relu')) #depth : 3 -> 16
    #net.append(ConvolutionLayer(patch_size,16,16,activation='relu')) #depth : 16 -> 16
    #net.append(DropoutLayer(0.5))
    net.append(DenseLayer((9, 16), activation='relu'))
    net.append(DenseLayer((16, 9), activation='none'))
    net.setup()

    # roughly something like ...

    # train
    n_epoch = 1
    u_freq = 1
    batch_size = 64
    n_iter = 32

    for epoch in range(n_epoch):
        ttt = TicTacToe() # NEW BOARD
        while True:
            # COLLECT DATA
            s1 = ttt.sense()
            choices = ttt.available() #np.nonzero(s1==X) #available actions
            if len(choices) == 0: # draw
                break
            a = np.random.choice(choices)
            r = ttt.act(a)
            s2 = ttt.sense()

            # TABLE
            s1_t = reformat(s1) 
            s2_t = reformat(s2)
            maxqn = -max(qtable[s2_t]) #negate opponent's1 best, expectimax
            qtable[s1_t][a] = lerp(r+gamma*maxqn,qtable[s1_t][a],alpha)

            # DQN
            memories.append((s1,a,r,s2))

            if r != 0.0: #win/loss
                break

        if (epoch % u_freq) == 0 and len(memories) == 4096:
            s1,a,r,s2 = zip(*memories)
            s1,a,r,s2 = arr(s1,dtype=np.float32),arr(a),arr(r),arr(s2,dtype=np.float32)
            q_new = r + gamma * -np.max(net.predict(s2),1)
            y_new = net.predict(s1)
            y_new[range(len(memories)),a] = lerp(q_new, y_new[range(len(memories)),a],alpha)
            net.train(s1, arr([qtable[reformat(s)] for s in s1]), batch_size, n_iter)
            #net.train(s1, y_new, len(memories), batch_size)
        # for optimization, allow one-hot-training

        #net.run(memories['s1'],lerp(memories['r'] + gamma * -np.max(qtable[memories['s2']],1), qtable[memories[s1]][a],alpha))
    print "LEN : " ,len(qtable.keys())
 
     # test
    while raw_input("Game? [y/n]") == 'y':
        ttt = TicTacToe()
        while True:
            print ttt.board
            s1 = ttt.sense()
            choices = ttt.available() #np.nonzero(s1==X) #available actions
            if len(choices) == 0: # draw
                break
            if ttt.turn == A:
                a = int(raw_input("action :"))
            else:
                #a = choose(qtable[s1],choices)
                print s1.shape
                s1 = np.reshape(s1, (1,) + s1.shape).astype(np.float32)
                print s1
                print s1.shape
                a = choose(np.reshape(net.predict(s1), 9), choices)
            #a = np.argmax(qtable[s1])
            r = ttt.act(a)
            if r != 0.0: #win/loss
                break
