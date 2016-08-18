#!/usr/bin/python
import random
import pygame
from pygame.locals import *

from Tetris import *
from Net import *
import numpy as np

import pickle

from matplotlib import pyplot as plt

import signal
import sys

ALPHA = 0.6
GAMMA = 0.4

W = 8
H = 20

DISPLAY_SCREEN = False 


class State(object):
    def __init__(self):
        pass

class Action(object):
    def __init__(self):
        pass

class Agent(object):
    def __init__(self):
        pass

scaleFactor = 1/50.0

class TetrisState(State):
    def __init__(self,prev=None,shape=None): #board,block
        if prev == None:
            self.board = Board(*shape)
            self.block = Block()
            self.reward = 0.0
        else:
            self.board = prev.board + prev.block
            self.block = prev.nextBlock 
            var = np.var(self.board.summary())
            #default 1, extra points for empty spaces & line-clears, minus points for too much variance(height difference)
            
            se = self.spaceEmpty()
            lc = self.lineClear()

            self.reward = (1 + .9*se + .86*lc - .2*var) * scaleFactor
            #print(self.reward)
            #self.reward = (1 + self.lineClear()) * 0.1
            #print(self.reward)
        self.nextBlock = Block()

    def getActions(self):
        return self.block.valid()
    def next(self,a):
        self.block.r = a.rot
        self.block.j = a.loc
        return TetrisState(prev=self)

    def spaceEmpty(self): #first-pass reward model with Immediate Reward
            return (self.board.h - max(self.board.summary())) # lower height = better.
    def summary(self,axis=None):
        #board
        if axis == None:
            s = self.board.summary()
        else:
            s = self.board.summary(axis)
        
        m = min(s)
        s = [float(h-m)/self.board.h for h in s]
        return s + self.nextBlock.summary() + self.block.summary()#block summary is just type.

    def lineClear(self):
        if self.board.over:
            return -5
        else:
            c = self.board.check()
            #if c != 0:
            #    print ("########JACKPOT!!#########")
            return 5*c 
    def done(self):
        return self.board.over
    def draw(self,screen):
        self.board.draw(screen)

class TetrisAction(Action):
    def __init__(self,rot,loc):
        self.rot = rot
        self.loc = loc
    def summary(self,axis=None):
        rs = [(1.0 if self.rot==i else 0.0) for i in range(4)]#
        ls = [(1.0 if self.loc==i else 0.0) for i in range(-3,W+3)]
        if axis == None:
            return rs+ls
        else:
            return rs

        #return [self.rot/3.0,self.loc/float(W)]
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "R:{},L:{}".format(self.rot,self.loc)


class TetrisAgent(Agent):
    def __init__(self,shape):
        self.shape = shape
        self.w = shape[0]
        self.h = shape[1]
        
        W = self.w
        H = self.h

        self.state = None
        #1 for block, 1 for nextblock, 2 for action(loc/rot of block)
        #8 for each height in board width, outputing 1 Q-value
        I = len(TetrisState(shape=shape).summary(0) + TetrisAction(0,0).summary(0)) 
        #print(I)
        t = [I,I/2,1]
        self.net = Net(t)
        #print(self.net.W)
    def chooseBest(self):
        actions = self.state.getActions()
        s = None
        a = None 
        q = -50000 

        #qList = []
        #aList = []
        for r,lr in enumerate(actions): #rotation
            #print(lr)
            for l in range(0-lr[0],self.w+lr[1]-3): #location(j)
                #print("L",l)
                _s = self.state.summary(l)
                #print("S",_s)
                _a = TetrisAction(r,l)# l now migrated into state-summary
                #print("A",_a)
                #print(self.state)
                ns = self.state.next(_a) #next-state
                #print(self.state)

                if ns.done():
                    _q = ns.reward
                else:
                    _q = self.net.FF(_s+_a.summary(l))
                    #print(_q)

                #print(_s)
                #print(_q)
                #qList += [(_s,_q)]
                if _q > q:
                    s = _s
                    q = _q
                    a = _a
        #print('A:', a)
        #print('Q:',qList)
        return s,a,q

    def chooseRand(self):
        actions = self.state.getActions()
        r = np.random.randint(4)
        lr = actions[r]
        l = np.random.randint(0-lr[0],self.w + lr[1] - 3)
        
        s = self.state.summary(l)
        a = TetrisAction(r,l)
        return s, a, self.net.FF(s+a.summary(l)) 

    def chooseNext(self): #choose "best next state"
        if np.random.random()<0.1:
            return self.chooseRand()
        else:
            return self.chooseBest()

    def draw(self,screen):
        self.state.draw(screen)

    def run(self,screen=None,delay=0):
        self.state = TetrisState(shape=self.shape)
        epoch = 0
        while not self.state.done():
            epoch += 1
            #s = self.state.summary()
            s,a,q = self.chooseNext() #select action
            #print(a.summary())
            self.state = self.state.next(a)
            _,_,_q = self.chooseBest() #best of next
            #print(_q)
            q2 = (1-ALPHA)*q + ALPHA*(self.state.reward + GAMMA*_q)
            #if self.state.getReward() == -100:
            #    print("U:{}, U2:{}".format(u, u2))
            self.net.BP(s+a.summary(0),q2) #update Q value

            if(DISPLAY_SCREEN):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        return -1
                if screen is not None:
                    self.draw(screen)
                    pygame.display.update()
                    if delay != 0:
                        pygame.time.wait(delay)
            #raw_input("...")
        return epoch 


agent = None
def askSave():
    save = raw_input("SAVE? (Y/N)")
    if save == "Y" or save == "y":
        with open('agent','w') as f:
            pickle.dump(agent,f)

def handler(signum,frame):
    print("TERMINATE...")
    askSave()
    sys.exit(0)

signal.signal(signal.SIGINT,handler) # register SIGINT handler


def main():
    w,h = 10,20
    if(DISPLAY_SCREEN):
        pygame.init()
        screen = pygame.display.set_mode((w*50,h*50))
        pygame.display.set_caption('Tetris_AI')
    global agent
    agent = TetrisAgent((w,h))
    #with open('agent','r') as f:
    #    agent = pickle.load(f)
    
    scores = []
    for i in range(1000):
        score = agent.run()
        if score == -1:
            break
        scores += [score]
        print("[{}] SCORE : {}".format(i,score))
    
    
    #for i in range(10000):
    #    agent.run()
    #    print(i)
    print("Final SCORE : {}".format(agent.run()))

    askSave()
    plt.plot(scores)
    plt.show()

if __name__ == "__main__":
    main()


