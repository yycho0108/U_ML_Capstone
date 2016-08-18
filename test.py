from tictactoe import *

import numpy as np

ttt = TicTacToe()

ttt.board = np.atleast_2d([[B,X,A],
                        [A,A,X],
                        [A,A,B]])
print ttt.check()
