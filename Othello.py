# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:57:23 2018

@author: jevcl
"""

import os
import shutil

import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.nn.parameter import Parameter
import pickle

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
import torch.nn.functional as F

EPSILON = 0.25

#====================================================================================
#Othello
#====================================================================================

class GameState:
    
    #define adjacent cells
    adjacentCells = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    height = 8
    width = 8

    #define enemy of player
    def enemy(self):
        if(self.player == 1):
            return 2
        elif(self.player == 2):
            return 1

    #out of bounds function
    def oob(x,y):
        if x > GameState.height-1 or y > GameState.width-1 or x < 0 or y < 0:
            return True
        return False
    
    def moveToIndex(move):
        if move == (-1,-1):
            return GameState.height*GameState.width
        return move[0]*GameState.height+move[1]
    
    #initialize new game state, can be a copy of existing game state
    def __init__(self, original=None):
        if original != None:
            self.board = np.copy(original.board)
            self.player = np.copy(original.player)
    
    #Creates Board. Empty slot = 0. Black = 1. White = 2.
    def makeInitialState(self):
        board = np.zeros((GameState.height,GameState.width))
        board[int(GameState.height/2-1)][int(GameState.width/2-1)] = 2
        board[int(GameState.height/2)][int(GameState.width/2)] = 2
        board[int(GameState.height/2-1)][int(GameState.width/2)] = 1
        board[int(GameState.height/2)][int(GameState.width/2-1)] = 1
        self.board = board
        self.player = 1
        
    def toVector(self, cuda = False):
        player = -1 if self.player == 1 else 1
        ret = self.board.flatten()
        ret = np.append(ret, player)
        ret = torch.from_numpy(ret).float()
        if cuda:
            ret = ret.cuda()
        return ret

    def toNetState(self, cuda = False):
        ret = torch.zeros(1,2,GameState.height,GameState.width)
        enemy = self.enemy()
        for i in range(GameState.height):
            for j in range(GameState.width):
                if self.board[i,j] == self.player:
                    ret[0,0,i,j] = 1
                elif self.board[i,j] == enemy:
                    ret[0,1,i,j] = 1
        if cuda:
            ret = ret.cuda()
        return ret

    #Update all cells after move. Observe, this function does not care wether the move is legal or not.
    def updateState(self, move):
        newGameState = GameState(self)
        if move != (-1,-1):
            xCord, yCord = move
            for xVec,yVec in GameState.adjacentCells:
                newGameState.updateDirection(xCord, yCord, xVec, yVec)
        newGameState.player = newGameState.enemy()
        return newGameState

    #Update cells in one direction
    def updateDirection(self, xCord, yCord, xVec, yVec):
        x = xCord + xVec
        y = yCord + yVec
        if GameState.oob(x,y) or self.board[x][y] == 0:
            return False
        elif self.board[x][y] == self.player:
            self.board[xCord][yCord] = self.player
            return True
        else:
            ret = self.updateDirection(x, y, xVec, yVec)
            if ret:
                self.board[xCord][yCord] = self.player
            return ret

    #Find all legal moves for player
    def findLegalMoves(self, hasCheckedEnemy=False):
        legalMoves = [];
        for xCord in range(GameState.height):
            for yCord in range(GameState.width):
                if self.board[xCord][yCord] != 0:
                    continue
                for xVec,yVec in GameState.adjacentCells:
                    if self.testLegalDirection(xCord, yCord, xVec, yVec, False):
                        legalMoves.append((xCord,yCord))
                        break
        if len(legalMoves) == 0 and hasCheckedEnemy == False:
            enemyState = self.updateState((-1,-1))
            enemyMoves = enemyState.findLegalMoves(True)
            if len(enemyMoves) > 0:
                legalMoves.append((-1,-1)) #do nothing, give over the turn to the enemy
        return legalMoves

    #Determine if move is legal based on one direction
    def testLegalDirection(self, xCord, yCord, xVec, yVec, enemySquare):
        x = xCord + xVec
        y = yCord + yVec
        if GameState.oob(x,y) or self.board[x][y] == 0:
            return False
        elif self.board[x][y] == self.enemy():
            return self.testLegalDirection(x, y, xVec, yVec, True)
        elif self.board[x][y] == self.player and enemySquare:
            return True
        else:
            return False

    def gameScore(self):
        p, e = self.score()
        if p > e:
            return 1
        elif e > p:
            return -1
        else:
            return 0

    def gameWinner(self):
        p, e = self.score();
        if self.player == 1 and p > e or self.player == 2 and e > p:
            print("Black Wins!")
        elif self.player == 1 and p < e or self.player == 2 and e < p:
            print("White Wins!")
        else:
            print("Draw!")
	
    def score(self):
        enemy = self.enemy()
        p = 0
        e = 0
        for xCord in range(GameState.height):
            for yCord in range(GameState.width):
                if self.board[xCord][yCord] == self.player:
                    p += 1
                elif self.board[xCord][yCord] == enemy:
                    e += 1
        return p, e
    
#====================================================================================
#Monte-Carlo
#====================================================================================
    
class MCTS:
    
    def __init__(self, numberOfSimulations, smartQ = False, network = None, cuda = False):
        self.numberOfSimulations = numberOfSimulations
        self.smartQ = smartQ
        self.network = network
        self.cuda= cuda
        
    def simulate(self, root, dirichletInRoot):
        self.root = root
        isRoot = dirichletInRoot
        nTotalRoot = int(root.N)
        for nTotal in range(nTotalRoot + 1, nTotalRoot + 1 + self.numberOfSimulations):
            currentNode = self.root
            while(True):
                if currentNode.isLeaf():
                    break
                else:
                    isRoot = False
                    currentNode = currentNode.chooseNodeToExplore(nTotal)
            gameResult = currentNode.expand(nTotal, isRoot)
            currentNode.backpropagate(gameResult, currentNode.gameState.player)
                
    def rootWithActionProbs(self):
        totalVisits = sum(c.N for c in self.root.children)
        actionProbs = torch.zeros(GameState.height*GameState.width + 2)
        for c in self.root.children:
            actionProbs[GameState.moveToIndex(c.move)] = c.N/totalVisits
        return self.root.gameState.toNetState(), actionProbs
        
    def bestChildState(self):
        bestChild = None
        #print("looking for move for state: ")
        #print(self.root.gameState.board)
        visits = []
        for child in self.root.children:
            visits.append(child.N)
            #print(child.numberOfVisits)
            #print(child.move)
        visits = np.array(visits)
        ######print(visits)                         #this is the one        
        best = np.argwhere(visits == max(visits))
        bestChild = self.root.children[random.choice(best)[0]]
        #print(bestChild.move)
        return bestChild
    
    def sampleChildState(self):
        totalVisits = sum(c.N for c in self.root.children)
        probs = [c.N/totalVisits for c in self.root.children]
        child = np.random.choice(self.root.children, 1, p = probs)[0]
        return child
    
class TreeNode:
    children = None
    parent = None
    
    def isLeaf(self):
        return self.children == None or len(self.children) == 0
    
class MCTSNode(TreeNode):
    
    def __init__(self, gameState, prior = 1., move = None, parent = None, network = None, smartQ = False, cuda = False):
        self.gameState = gameState
        self.move = move
        self.W = 0.
        self.N = 0.
        self.Q = 0.
        self.P = prior
        self.parent = parent
        self.network = network
        self.smartQ = smartQ
        self.cuda = cuda
        
    def setToRoot(self):
        self.parent = None
        
    def expand(self, nTotal, isRoot):
        self.children, gameResult = self.computeChildNodes(isRoot)
        if self.isLeaf():
            gameResult = self.gameState.gameScore()     
        elif self.network == None:
            gameResult = self.rollout()
        #print(gameResult)
        return gameResult
    
    def chooseNodeToExplore(self, nTotal, cParam = 1.):
        choicesWeights = [c.nodeChoiceMetric(nTotal, cParam) for c in self.children]
        return self.children[np.argmax(choicesWeights)]
        
    def computeChildNodes(self, isRoot):
        legalMoves = self.gameState.findLegalMoves()     
        if self.network == None:
            prior = torch.ones(1,GameState.height*GameState.width+1)
            gameResult = None
        else:
            with torch.no_grad():
                inp = self.gameState.toNetState(self.cuda)
                #inp = getRandomRotationOrReflection(inp) #this should not be used in its current form!
                net = self.network(inp)
            prior = net[0]
            gameResult = net[1]
            #######
            #print(priori)
            prior -= 100
            for move in legalMoves:
                prior[0,GameState.moveToIndex(move)] += 100
            prior = softmax(prior, dim=1)
            if isRoot:
                prior *= (1-EPSILON)
                dirichlet = EPSILON*np.random.dirichlet([0.03] * len(legalMoves))
                for i in range(len(legalMoves)):
                    move = legalMoves[i]
                    prior[0,GameState.moveToIndex(move)] += dirichlet[i]
            #print(priori)
            #priori = torch.ones(1,GameState.height*GameState.width)
            #print(priori)
            #for move in legalMoves:
            #    print(priori[0,GameState.moveToIndex(move)])
            #print(gameResult)
        childNodes = [MCTSNode(self.gameState.updateState(move), prior[0,GameState.moveToIndex(move)], move, self, self.network, self.smartQ, self.cuda) for move in legalMoves]
        #childNodes = [MCTSNode(self.gameState.updateState(move), priori[GameState.moveToIndex(move)], move, self, self.network, self.smartQ, self.cuda) for move in legalMoves]
        return childNodes, gameResult
    
    def nodeChoiceMetric(self, nTotal, cParam):
        #if numberOfVisits == 0:
        #    return float("inf") #if the node has not been visited before, we should do that asap
        #print(self.Q)
        #print(cParam * self.priori * math.sqrt((np.log(nTotal) / (1 + numberOfVisits))))
        v =  self.Q + cParam * self.P * math.sqrt(nTotal) / (1 + self.N)
        return v
    
    def rollout(self):
        currentRolloutState = self.gameState
        while(True):
            possibleMoves = currentRolloutState.findLegalMoves()
            if len(possibleMoves) > 0:
                move = self.rolloutPolicy(possibleMoves)
                currentRolloutState = currentRolloutState.updateState(move)
            else:
                break
        rolloutResult = currentRolloutState.gameScore()
        return rolloutResult
            
    
    def backpropagate(self, gameResult, leafPlayer):
        if self.gameState.player == leafPlayer:
            direction = -1
        else:
            direction = 1
        self.N += 1.
        self.W += direction*gameResult
        self.Q = self.W / self.N
        if self.parent != None:
            self.parent.backpropagate(gameResult, leafPlayer)
    
    def rolloutPolicy(self, possibleMoves):
        return possibleMoves[np.random.randint(len(possibleMoves))]
    
#====================================================================================
#Machine Learning
#====================================================================================

#network specifications
learningRate = 0.01        
channels = 2
numFiltersConv1 = 32
kernelSizeConv1 = 3
strideConv1 = 1
paddingConv1 = 1

numFiltersConvVH = 1
kernelSizeConvVH = 1
strideConvVH = 1
inFeaturesLVH = numFiltersConvVH * GameState.height * GameState.width
outFeaturesLVH = GameState.height * GameState.width

numFiltersConvPH = 2
kernelSizeConvPH = 1
strideConvPH = 1
inFeaturesLPH = numFiltersConvPH * GameState.height * GameState.width
outFeaturesLPH = GameState.height * GameState.width + 1

class Net(nn.Module):
    """Policy network"""

    def __init__(self):
        super(Net, self).__init__()
        
        self.c1 = Conv2d(in_channels=channels,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm1 = nn.BatchNorm2d(numFiltersConv1)
        
        # Residual layer 1
        self.c2 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm2 = nn.BatchNorm2d(numFiltersConv1)
        
        self.c3 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm3 = nn.BatchNorm2d(numFiltersConv1)
        
        # Residual layer 2
        self.c4 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm4 = nn.BatchNorm2d(numFiltersConv1)
        
        self.c5 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm5 = nn.BatchNorm2d(numFiltersConv1)
        
        # Residual layer 3
        self.c6 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm6 = nn.BatchNorm2d(numFiltersConv1)
        
        self.c7 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm7 = nn.BatchNorm2d(numFiltersConv1)
        
        # Residual layer 4
        self.c8 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm8 = nn.BatchNorm2d(numFiltersConv1)
        
        self.c9 = Conv2d(in_channels=numFiltersConv1,
                         out_channels=numFiltersConv1,
                         kernel_size=kernelSizeConv1,
                         stride=strideConv1,
                         padding=paddingConv1)
        
        self.batchNorm9 = nn.BatchNorm2d(numFiltersConv1)
        
        # PH head
        self.convPH = Conv2d(in_channels=numFiltersConv1,
            out_channels=numFiltersConvPH,
            kernel_size=kernelSizeConvPH,
            stride=strideConvPH)
        
        self.batchNormPH = nn.BatchNorm2d(numFiltersConvPH)
        
        self.lPH = Linear(in_features=inFeaturesLPH,
                          out_features=outFeaturesLPH,
                          bias=True)
        
        # VH
        self.convVH = Conv2d(in_channels=numFiltersConv1,
            out_channels=numFiltersConvVH,
            kernel_size=kernelSizeConvVH,
            stride=strideConvVH)
        
        self.batchNormVH = nn.BatchNorm2d(numFiltersConvVH)
        
        self.lVH = Linear(in_features=inFeaturesLVH,
                          out_features=outFeaturesLVH,
                          bias=True)
        
        self.lVHout = Linear(in_features=outFeaturesLVH,
                          out_features=1,
                          bias=False)
        
    def forward(self, x):
        x = self.batchNorm1(relu(self.c1(x)))
        res1 = x
        x = self.batchNorm2(relu(self.c2(x)))
        x = self.batchNorm3(relu(self.c3(x)))
        x += res1
        res2 = x
        x = self.batchNorm4(relu(self.c4(x)))
        x = self.batchNorm5(relu(self.c5(x)))
        x += res2
        res3 = x
        x = self.batchNorm6(relu(self.c6(x)))
        x = self.batchNorm7(relu(self.c7(x)))
        x += res3
        res4 = x
        x = self.batchNorm8(relu(self.c8(x)))
        x = self.batchNorm9(relu(self.c9(x)))
        x += res4
        ph = self.batchNormPH(relu(self.convPH(x)))
        ph = ph.view(-1, inFeaturesLPH)
        ph = self.lPH(ph)
        vh = self.batchNormVH(relu(self.convVH(x)))
        vh = vh.view(-1, inFeaturesLVH)
        vh = relu(self.lVH(vh))
        vh = torch.tanh(self.lVHout(vh))
        return ph, vh

    def loss(self, actionProbabilities, returns):
        v = torch.flatten(actionProbabilities[1])
        z = returns[:,-1]
        dif = (z-v)
        p = actionProbabilities[0]
        pi = returns[:,:-1]
        meanSqrErr = torch.mean(torch.mul(dif,dif))    
        cEL = -torch.mean(torch.sum(pi * torch.log(softmax(p, dim=1)), dim=1))
        #print(pi)
        #print(p)
        #print(torch.log(p))
        #print(torch.log(1-p))
        #print(meanSqrErr)
        #print(cEL)
        if meanSqrErr != meanSqrErr or cEL != cEL:
            raise Exception("This should never happen, we might have inf or nan in loss.")
        return meanSqrErr + cEL
    
def compute_returns(rewards, discount_factor):
    """Compute discounted returns."""
    returns = np.zeros(len(rewards))
    returns[-1] = rewards[-1]
    for t in reversed(range(len(rewards)-1)):
        returns[t] = rewards[t] + discount_factor * returns[t+1]
    return returns

def saveCheckpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        print("=> saving new best network: ", filename) 
        
def loadCheckpoint(network, resume):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        network.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume))
    else:
        print("=> no checkpoint found at '{}'".format(resume)) 
    
    if cuda:
        print('##converting network to cuda-enabled')
    return network
        
def selfPlay(player):
    initialState = GameState()
    initialState.makeInitialState()
    
    state = initialState
    stateNode = MCTSNode(state, 1., None, None, player.network, player.smartQ, player.cuda)
    
    states = []
    rewards = []
    
    moveNo = 0
    
    while(len(state.findLegalMoves()) > 0):
        #stateNode = MCTSNode(state, 1., None, None, player.network, player.smartQ, player.cuda)
        player.simulate(stateNode, True)
        sample = player.rootWithActionProbs()
        s = sample[0]
        s = getRotationsAndReflections(s)
        r = torch.stack([sample[1][:-2].view(GameState.height,GameState.width)])
        r = getRotationsAndReflections(r)
        rew = r.view(-1,GameState.height*GameState.width)
        rOut = []
        for r in rew:
            rOut.append(torch.cat((r,torch.tensor([sample[1][-2]]))))
        r = torch.stack(rOut)
        
        if len(states) == 0:
            states = s
            rewards = r
        else:
            states = torch.cat((states,s))
            rewards = torch.cat((rewards,r))
        if moveNo < 10:
            childNode = player.sampleChildState()
        else:
            childNode = player.bestChildState()
        stateNode = childNode
        state = stateNode.gameState
        moveNo += 1
    gs = state.gameScore()
    rOut = []
    i = 0
    for r in reversed(rewards):
        if i%8 == 0:
            gs*=-1
        rOut.insert(0,torch.cat((r,torch.tensor([gs]).float())))
        i += 1
    rOut = torch.stack(rOut)
    #for i in range(len(states)):
    #    print(states[i])
    #    print(rOut[i])
    return states, rOut

def testVsBest(currentNetwork, bestNetwork, noOfGames):
    w = 0
    l = 0
    d = 0
    for i in range(noOfGames):        
        if i % 2 == 0:
            blackPlayer = currentNetwork
            whitePlayer = bestNetwork
            direction = 1
        else:
            blackPlayer = bestNetwork
            whitePlayer = currentNetwork
            direction = -1
        
        initialState = GameState()
        initialState.makeInitialState()
        
        isBlackplayersTurn = True
        player = blackPlayer
        
        state = initialState

        while(len(state.findLegalMoves()) > 0):
            stateNode = MCTSNode(state, 1., None, None, player.network, player.smartQ, player.cuda)
            player.simulate(stateNode, False)
            state = player.bestChildState().gameState
            direction *= -1

            if isBlackplayersTurn:
                #print("white players turn")
                player = whitePlayer
                isBlackplayersTurn = False
            else:
                #print("black players turn")
                player = blackPlayer
                isBlackplayersTurn = True
                
        score = state.gameScore()
        score *= direction
        if score == -1:
            l += 1
        elif score == 1:
            w += 1
        else:
            d += 1
                
        #state.gameWinner()
        
    if noOfGames == d:
        return 0., w, l, d
    return w/(noOfGames-d), w, l, d

def shuffle(states, rewards):
    r = torch.randperm(len(states))
    return states[r], rewards[r]

def getSlice(states, rewards, batchSize, i):
    return states[range(i*batchSize,(i+1)*batchSize)], rewards[range(i*batchSize,(i+1)*batchSize)]

def randomBatch(states, rewards, batchSize):
    r = random.sample(range(len(states)), batchSize)
    return states [r], rewards[r]

def getRotationsAndReflections(t):
    ret = t.clone()
    ret = torch.cat((t,t.transpose(-2,-1)))
    ret = torch.cat((ret,t.flip(-2)))
    ret = torch.cat((ret,t.flip(-1)))
    ret = torch.cat((ret,t.transpose(-2,-1).flip(-2)))
    ret = torch.cat((ret,t.transpose(-2,-1).flip(-1)))
    ret = torch.cat((ret,t.flip(-2).flip(-1)))
    ret = torch.cat((ret,t.transpose(-2,-1).flip(-2).flip(-1)))
    return ret

def getRandomRotationOrReflection(t):
    rot = getRotationsAndReflections(t)
    r = random.randint(0, len(t)-1)
    return rot[r:r+1]    
#'''
selfPlayGames = 30
numberOfEpisodes = 1000
batchSize = 1
minibatchSize = 32
trainingBatches = 10
memorySize = 1
vsTestGames = 10
isBest = False
mctsSims = 1600

cuda = torch.cuda.is_available()
cuda = False
if cuda:
    print("Using cuda")
else:
    print("Using CPU")

t = time.time()

#try:
    
bestNet = Net()
#Potential restart
torch.save(bestNet.state_dict(), "bestNet8by8.pth")
torch.save(bestNet.state_dict(), "checkpoint8by8_0.pth")

states = []
rewards = []

currentNet = Net()
#optimizer = optim.Adam(currentNet.parameters(), lr=learningRate)
#criterion = nn.MSELoss()
optimizer = optim.SGD(currentNet.parameters(), lr=0.05)
#optimizer = optim.SGD(currentNet.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0001)
#currentNet = loadCheckpoint(currentNet, resume)

for e in range(numberOfEpisodes):
	
	if isBest == False:          
		print("Loading best network!")
		currentNet.load_state_dict(torch.load("bestNet8by8.pth"))
		
	#isBest = False
	isBest = True #continue working on the same network
	if cuda:
		#currentNet = torch.nn.DataParallel(currentNet, device_ids=None)
		#initialNet = torch.nn.DataParallel(initialNet, device_ids=None)
		currentNet.cuda()
		bestNet.cuda()
		
	#reset states and rewards    
	states = []
	rewards = []
	
	print("Starting self play.")
	EPSILON = 0.25
	for i in range(selfPlayGames):
		print(i+1, end=' ')
		currentPlayer = MCTS(mctsSims, True, currentNet, cuda)
		currentNet.eval()
		samples = selfPlay(currentPlayer)
		if len(states) == 0:
			states = samples[0]
			rewards = samples[1]
		else:
			states = torch.cat((states,samples[0]))
			rewards = torch.cat((rewards,samples[1]))
	
	print("\nSelf play completed.")
	elapsed = time.time() - t
	print(elapsed)
	
	if len(states) >= memorySize:
		
		statesName = "states8by8_" + str(e+1) + ".txt"
		rewardsName = "rewards8by8_" + str(e+1) + ".txt"
		with open(statesName, 'wb') as f:
			pickle.dump(states, f)
		with open(rewardsName, 'wb') as f:
			pickle.dump(rewards, f)
		
		#states = states[-memorySize:,:,:,:]
		#rewards = rewards[-memorySize:,:]
		sumLoss = 0
		print("Starting training.")
		for i in range(trainingBatches): #fix this len(states)//batchSize
			#stateBatch, rewardsBatch = randomBatch(states, rewards, batchSize)
			batchSize = len(states)
			stateBatch = states
			rewardBatch = rewards
			sumLossBatch = 0
			currentNet.train()
			for j in range(int(batchSize/minibatchSize)):
				stateList = stateBatch[j*minibatchSize:(j+1)*minibatchSize]
				rewardList = rewardBatch[j*minibatchSize:(j+1)*minibatchSize]
				#stateList = states
				#rewardList = rewards
				
				#for t in range(1000):
				optimizer.zero_grad()
				if cuda:
					preds = currentNet(stateList.cuda())
				else:
					preds = currentNet(stateList)
				if cuda:
					rewards = rewardList.cuda()
				#print(stateList)
				#print(preds[1])
				#print(rewardList[:,-1])
				#for b in range(8):
				#    print(preds[0][b])
				#    print(preds[1][b])
				#    print(rewardList[b])
				loss = currentNet.loss(preds, rewardList)
				#loss2 = criterion(preds[1],rewardList[:,-1])
				sumLossBatch += loss
				#print(loss)
				loss.backward()
				optimizer.step()
			sumLoss += sumLossBatch
			
			if i % 5 == 0:
				print("Training iter: ", i+1, " loss: ", sumLossBatch)
				#print(preds)
				#print("Training iter: ", t+1, " loss: ", loss)                
			
		print("Training loss: ", sumLoss)
			
		print("Training completed.")    
		elapsed = time.time() - t
		print(elapsed)
		print("Starting evaluation.")
		EPSILON = 0.0    
		bestNet.load_state_dict(torch.load("bestNet8by8.pth"))
		currentNet.eval()
		bestNet.eval()
		currentPlayer = MCTS(mctsSims, True, currentNet, cuda)
		bestPlayer = MCTS(mctsSims, True, bestNet, cuda)
		wr = testVsBest(currentPlayer, bestPlayer, vsTestGames)
		if wr[0] >= 0.55:
			isBest = True
			torch.save(currentNet.state_dict(), "bestNet8by8.pth")
			print("New best net found in iter ", e+1)
		checkpointName = "checkpoint8by8_" + str(e+1) + ".pth"
		torch.save(currentNet.state_dict(), checkpointName)
	
		print(wr)
		elapsed = time.time() - t
		print(elapsed)
		
	else:
		print("Memory is at: ", len(states))
    
#    print('done')
#except KeyboardInterrupt:
#    print('interrupt')   
