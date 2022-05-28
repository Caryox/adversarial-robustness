'''
ENSEMBLE PAPER
@bene: I have to many troubles with building the ensemble from scratch as in the article so I decided to swith back to Plan A:
Use the ensemble method as in the paper https://arxiv.org/pdf/2008.10400.pdf
They also use majority voting, just no explaination, hence I need your help to understand what they do
'''
import numpy as np 
#As I understand, we dont need this cuz we have param, right? 
import argparse

cnt = 1
best = 10000
curr = 10000

p = argparse.ArgumentParser()
p.add_argument("--kernel_size", default=5, type=int)
args = p.parse_args()
KERNEL_SIZE = args.kernel_size # could be 3, 5, 7, or 9

for i in range(10):
    for j in range(i+1,10):
        for k in range(j+1,10):
			#I understand that we need to load weights from modified models hier. but dont get the loop at all TT_TT
            w1 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, i)).astype(np.int)
            w2 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, j)).astype(np.int)
            w3 = np.loadtxt("../logs/modelM%d/wrong%03d.txt"%(KERNEL_SIZE, k)).astype(np.int)

            board = np.zeros((10000))
            board[w1] += 1
            board[w2] += 1
            board[w3] += 1
			#This part must be the majority voting part, which I dont understand. Still can use it, but it would be nice to know how it works
            board = board // 2
            curr = np.sum(board)
            if curr < best:
                best = curr
            print("%4d %4d %4d %4d %4d %4d"%(cnt, len(w1), len(w2), len(w3), curr, best))
            cnt += 1