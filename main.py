import tensorflow as tf
from tictactoe import *
from nnet import *
from collections import defaultdict, deque
import pandas as pd
qtable = defaultdict(lambda:[0.5 for _ in range(9)])

arr = np.asarray

def dummy():
    print "NEW"
    return 0

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

    # train parameters
    alpha = 0.25
    gamma = 0.4 # discount_rate
    n_epoch = 10000
    u_freq = 4
    batch_size = 32
    n_iter = 200
    mem_size = 8192

    # setup network
    net = Net()
    net.append(DenseLayer((9, 16), activation='relu'))
    net.append(DenseLayer((16, 16), activation='relu'))
    net.append(DenseLayer((16, 9), activation='none'))

    #net.append(ConvolutionLayer(patch_size,1,16,activation='relu')) #depth : 3 -> 16
    #net.append(ConvolutionLayer(patch_size,16,16,activation='relu')) #depth : 16 -> 16
    #net.append(DropoutLayer(0.5))
    net.setup(batch_size,(3,3,1),(9,))

    memories = deque(maxlen=mem_size)
    #memories = []

    # pre-train q table
    i = 0
    for epoch in range(n_epoch):
        print 'epoch : {}'.format(epoch)
        ttt = TicTacToe() # NEW BOARD
        end = False
        s2 = None

        while not end:
            # COLLECT DATA
            if s2 == None:
                s1 = ttt.sense().copy() # copy
            else:
                s1 = s2

            # TABLE
            s1_t = reformat(s1) 

            choices = ttt.available() #np.nonzero(s1==X) #available actions
            if len(choices) == 0: # draw
                break
            a = np.random.choice(choices)
            end, r = ttt.act(a)
            s2 = ttt.sense().copy()

            s2_t = reformat(s2)
            maxqn = -max(qtable[s2_t]) #negate opponent's best, expectimax
            qtable[s1_t][a] = lerp(r+gamma*maxqn,qtable[s1_t][a],alpha)

            # DQN
            memories.append((s1,a,r,s2))

    print "LEN : " ,len(qtable.keys())

    print len([1 for v in qtable.values() if tuple([0.5 for _ in range(9)]) == tuple(v)]) # 956..ish

    # train network
    i = 0
    for epoch in range(n_epoch):
        print 'epoch : {}'.format(epoch)
        ttt = TicTacToe() # NEW BOARD
        end = False
        s2 = None
        while not end:
            i = i+1
            # COLLECT DATA
            if s2 == None:
                s1 = ttt.sense().copy() # copy
            else:
                s1 = s2

            # TABLE
            s1_t = reformat(s1) 
            #print 's1_t', s1_t, type(s1_t)

            choices = ttt.available() #np.nonzero(s1==X) #available actions
            if len(choices) == 0: # draw
                break
            a = np.random.choice(choices)
            end, r = ttt.act(a)
            s2 = ttt.sense().copy() # copy

            s2_t = reformat(s2)
            maxqn = -max(qtable[s2_t]) #negate opponent's best, expectimax
            qtable[s1_t][a] = lerp(r+gamma*maxqn,qtable[s1_t][a],alpha)

            # DQN
            memories.append((s1,a,r,s2))

            if len(memories) >= mem_size and i % u_freq == 0:

                s1_v,a_v,r_v,s2_v = zip(*memories)
                s1_v,a_v,r_v,s2_v = arr(s1_v),arr(a_v),arr(r_v),arr(s2_v)

                q_new = r_v + gamma * -np.max(net.predict(s2_v),1)
                y_new = net.predict(s1_v)
                y_new[range(mem_size),a_v] = q_new # replace with new q_new

                # train with q table
                net.train(s1_v, arr([qtable[reformat(s)] for s in s1_v]), batch_size, n_iter)

                # train with network ( itself )
                #net.train(s1_v,y_new,batch_size,n_iter)
                #memories = []

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
                q_n = net.predict(np.reshape(s1, (1,) + s1.shape))[0]
                print 'q_t', q_t
                print 'q_n', q_n
                #a = choose(q_t, choices)
                a = choose(q_n, choices)
            r = ttt.act(a)
            if r != 0.0: #win/loss
                break
