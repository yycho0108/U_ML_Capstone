import tensorflow as tf
from tictactoe import *
from nnet import *
from collections import defaultdict, deque
import pandas as pd

#memories = Queue.Queue(maxsize=1000)

#def f():
#    print("NEW")
#    return 0.0

qtable = defaultdict(lambda:[0.5 for _ in range(9)])

arr = np.asarray

def lerp(a,b,c):
    return a*c + b*(1.0-c)

def reformat(state):
    return tuple(np.reshape(state, 9))

def choose(q,choices):
    #print choices
    q_avail = np.asarray(q)[choices]
    #print q_avail
    return choices[np.argmax(q_avail)]

if __name__ == "__main__":

    # test with just dense layers
    net = Net()
    #net.append(ConvolutionLayer(patch_size,1,16,activation='relu')) #depth : 3 -> 16
    #net.append(ConvolutionLayer(patch_size,16,16,activation='relu')) #depth : 16 -> 16
    #net.append(DropoutLayer(0.5))
    net.append(DenseLayer((9, 16), activation='relu'))
    net.append(DenseLayer((16, 16), activation='relu'))
    net.append(DenseLayer((16, 9), activation='none'))
    net.setup()

    # roughly something like ...

    # train
    alpha = 0.1
    dqn_alpha = 1.0
    gamma = 0.3
    n_epoch = 100000
    u_freq = 32
    batch_size = 16
    n_iter = 1000
    mem_size = 8192

    #memories = deque(maxlen=mem_size)
    memories = []

    # pre-train q table
    for epoch in range(n_epoch):
        ttt = TicTacToe() # NEW BOARD
        while True:
            # COLLECT DATA
            s1 = ttt.sense()
            # TABLE
            s1_t = reformat(s1) 

            choices = ttt.available() #np.nonzero(s1==X) #available actions
            if len(choices) == 0: # draw
                break
            a = np.random.choice(choices)
            r = ttt.act(a)
            s2 = ttt.sense()

            s2_t = reformat(s2)
            maxqn = -max(qtable[s2_t]) #negate opponent's1 best, expectimax
            qtable[s1_t][a] = lerp(r+gamma*maxqn,qtable[s1_t][a],alpha)

            if r != 0.0:
                break
                
            # DQN
            #memories.append((s1,a,r,s2))
            #if len(memories) >= mem_size:
            #    s1,a,r,s2 = zip(*memories)
            #    s1,a,r,s2 = arr(s1,dtype=np.float32),arr(a),arr(r),arr(s2,dtype=np.float32)
            #    q_new = r + gamma * -np.max(net.predict(s2),1)
            #    y_new = net.predict(s1)
            #    y_new[range(mem_size),a] = lerp(q_new, y_new[range(mem_size),a],dqn_alpha)
            #    net.train(s1, arr([qtable[reformat(s)] for s in s1]), batch_size, n_iter)
            #    memories = []

        #if (epoch % u_freq) == 0 and len(memories) > batch_size:

            #net.train(s1, y_new, batch_size, n_iter)
         #for optimization, allow one-hot-training

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
                #s1 = np.reshape(s1, (1,) + s1.shape).astype(np.float32)
                q_t = qtable[reformat(s1)]
                q_n = net.predict(np.reshape(s1, (1,) + s1.shape).astype(np.float32))[0]
                print 'q_t', q_t
                print 'q_n', q_n
                #a = choose(q_t, choices)
                a = choose(q_n, choices)
            r = ttt.act(a)
            if r != 0.0: #win/loss
                break
