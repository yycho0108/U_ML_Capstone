import numpy as np

X = 0
A = 1
B = 2

class TicTacToe:
    def __init__(self,h=3,w=3):
        self.h = h
        self.w = w
        self.turn = A
        self.board = np.zeros((h,w),dtype=np.float32) 

    def sense(self):
        return np.reshape(self.board,(self.h,self.w,1))
        #return tuple(np.reshape(self.board,(self.h*self.w))) #to column vector
        #return np.reshape(self.board,(self.h*self.w,1)) #to column vector
    def available(self):
        choices = [i*self.w+j for i in range(self.h) for j in range(self.w) if self.board[i,j] == X]
        return choices

    def act(self,a):
        i,j = a/self.w, a%self.w
        self.board[i,j] = self.turn
        won = self.check()
        end = True 

        if won == X:
            # no win
            reward = 0.0
            end = False
        elif won == self.turn:
            # my win
            reward = 0.1
        else:
            # other's win
            reward = -0.1

        self.turn = B if self.turn==A else A

        return end, reward
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
