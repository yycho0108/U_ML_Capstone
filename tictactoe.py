import numpy as np

X = 0
A = 1
B = 2

class TicTacToe:
    def __init__(self,h=3,w=3):
        self.h = h
        self.w = w
        self.turn = A
        self.board = np.zeros((h,w),dtype=np.uint8) 

    def sense(self):
        return np.reshape(self.board,(self.h,self.w,1))
        #return tuple(np.reshape(self.board,(self.h*self.w))) #to column vector
        #return np.reshape(self.board,(self.h*self.w,1)) #to column vector
    def available(self):
        return [i*self.w+j for i in range(self.h) for j in range(self.w) if self.board[i,j] == X]

    def act(self,a):
        i,j = a/self.w, a%self.w
        self.board[i,j] = self.turn
        won = self.check()

        if won == X:
            reward = 0.0
        elif won == self.turn:
            reward = 10.0
        else:
            reward = -10.0
        self.turn = B if self.turn==A else A
        return reward
    @staticmethod
    def win(board):
        v = np.all(board,0)
        h = np.all(board,1)
        d1 = np.all(np.diag(board))
        d2 = np.all(np.diag(np.fliplr(board)))
        return np.any(v+h+d1+d2)
    def check(self):
        if TicTacToe.win(self.board == A):
            return A
        elif TicTacToe.win(self.board == B):
            return B
        else:
            return X
