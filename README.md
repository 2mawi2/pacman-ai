# pacman-ai

This projects implements the Q-learning and TD(0) reinforcement learning algorithms.\


It is learning the best solution of 43 points in the following pacman environment:\

o o o o o o o o o o o o\
o W W W W W o W o W W o\ 
o W g o o W o W o x W o\ 
o W o o p W o W o o W o\ 
o W o o W W g W o o W o\ 
o o o o o o o W W W W d\

o -> point [reward = 1]\
W -> wall\
x -> star [reward = 10]\
g -> ghost (terminal state) [reward = -100]\
d -> door (terminal state) [reward = 0]\
