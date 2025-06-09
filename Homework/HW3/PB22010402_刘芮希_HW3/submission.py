'''
Licensing Information: piech@cs.stanford.edu
'''
import collections
import math
import random
import util
from engine.const import Const
from util import Belief


class ExactInference:

    # Function: Init
    # --------------
    # Constructor that initializes an ExactInference object which has numRows x numCols number of tiles.
    def __init__(self, numRows: int, numCols: int):
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()


    '''
    # Problem a:
    # Function: Observe (update the probabilities based on an observation)
    # Params:
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - use util.rowToY() and util.colToX() to convert indices into locations, and then compute distance
    # - update probability by util.pdf and self.belief.setProb()
    # - don't forget to normalize self.belief after you update its probabilities!
    '''

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                # 将格子索引转换为位置坐标
                tileY = util.rowToY(row)
                tileX = util.colToX(col)
                # 计算真实距离
                trueDist = math.sqrt((agentX - tileX) ** 2 + (agentY - tileY) ** 2)
                # 计算观测的概率密度并更新信念
                self.belief.setProb(row, col, self.belief.getProb(row, col) * util.pdf(trueDist, Const.SONAR_STD, observedDist))
        # 归一化信念
        self.belief.normalize()
        # END_YOUR_CODE


    '''
    # Problem b:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # Params: ..
    #
    # Notes:
    # - take self.Belief and add probability
    # - use the transition probabilities in self.transProb, which is a dictionary
    #   containing all the ((oldTile, newTile), transProb) key-val pairs that youmust consider.
    # - use the addProb and getProb methods of the Belief class to modify
    #   and access the probabilities associated with a belief.  (See util.py.)
    # - normalize and update
    # - be careful that you are using only the CURRENT self.belief distribution to compute updated beliefs.  
    '''

    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for oldRow in range(self.belief.getNumRows()):
            for oldCol in range(self.belief.getNumCols()):
                oldProb = self.belief.getProb(oldRow, oldCol)
                if oldProb > 0:  # 只考虑非零概率的格子
                    oldTile = (oldRow, oldCol)
                    # 应用转移概率
                    for (oTile, nTile), transProb in self.transProb.items():
                        if oTile == oldTile:  # 找到匹配的旧状态
                            newRow, newCol = nTile
                            newBelief.addProb(newRow, newCol, oldProb * transProb)
        newBelief.normalize()
        self.belief = newBelief
        # END_YOUR_CODE
#python drive.py -a -d -i ExactInference
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile 
    def getBelief(self) -> Belief:
        return self.belief



# Class: Likelihood Weighting Inference
# -----------------------------------
# Uses likelihood weighting sampling to approximate the belief distribution
# over a car's position on the grid.
class LikelihoodWeighting:
    NUM_SAMPLES = 200

    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(float)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize samples uniformly
        self.samples = [(random.randint(0, numRows - 1), random.randint(0, numCols - 1)) for _ in range(self.NUM_SAMPLES)]
        self.weights = [1.0 for _ in range(self.NUM_SAMPLES)]
        self.updateBelief()

    '''
    # Problem c
    # Function: Update Belief
    # params: ..
    #
    # Notes:
    # - this function is called after observation (when weights are updated)
    #   and after time elapse (when samples are moved).
    # - a util.Belief object represents the probability of being in each grid cell
    # - For each sample, you should add its weight to the corresponding tile.
    # - also, don't forget to normalize self.belief!!   
    '''
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)

        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # 遍历所有样本和对应的权重
        for i, (row, col) in enumerate(self.samples):
            # 将权重添加到对应的网格位置
            newBelief.addProb(row, col, self.weights[i])
    
        # 归一化信念分布
        newBelief.normalize()
    
        # 更新信念
        self.belief = newBelief
        # END_YOUR_CODE
        

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        for i, (r, c) in enumerate(self.samples):
            x = util.colToX(c)
            y = util.rowToY(r)
            trueDist = math.sqrt((agentX - x) ** 2 + (agentY - y) ** 2)
            self.weights[i] = util.pdf(trueDist, Const.SONAR_STD, observedDist)
        self.updateBelief()

    '''
    # Problem d
    # Function: Elapse Time
    # Params: ..
    #
    # Notes:
    # - for each current sample, sample a new location according to the transition
    #   probabilities from that position. This reflects the car moving in the world.
    # - you may find util.weightedRandomChoice() useful
    # - if a sample is in a location with no outgoing transitions defined,
    #   keep the sample in place (identity transition).
    # - after updating the sample positions, recompute the belief distribution
    #   using the new sample set and current weights by calling updateBelief()
    '''

    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        # 存储新的样本位置
        newSamples = []
        # 对每个当前样本
        for i, (row, col) in enumerate(self.samples):
            oldTile = (row, col)
        
            # 检查是否有从当前位置出发的转移
            if oldTile in self.transProbDict:
                # 根据转移概率采样新位置
                newTile = util.weightedRandomChoice(self.transProbDict[oldTile])
                newSamples.append(newTile)
            else:
                # 如果没有定义转移，保持在原位置
                newSamples.append(oldTile)
        # 更新样本位置
        self.samples = newSamples
        # 重新计算信念分布
        self.updateBelief()
        # END_YOUR_CODE


    def getBelief(self) -> Belief:
        return self.belief




class ParticleFilter:
    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has (numRows x numCols) number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        '''
        # Problem e: initialize the particles randomly
        # Notes:
        # - you need to initialize self.particles, which  is a defaultdict 
        #   from grid locations to number of particles at that location
        # - self.particles should contain |self.NUM_PARTICLES| particles randomly distributed across the grid.
        # - after initializing particles, you must call |self.updateBelief()| to compute the initial belief distribution.
        '''
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # 初始化粒子为空的defaultdict
        self.particles = collections.defaultdict(int)
    
        # 获取所有可能的位置（包括oldTile和newTile）
        possiblePositions = set()
        for oldTile in self.transProbDict:
            possiblePositions.add(oldTile)
            for newTile in self.transProbDict[oldTile]:
                possiblePositions.add(newTile)
        possiblePositions = list(possiblePositions)
    
        # 如果转移概率字典为空，则在整个网格上随机分布粒子
        if not possiblePositions:
            for _ in range(self.NUM_PARTICLES):
                row = random.randint(0, self.belief.getNumRows() - 1)
                col = random.randint(0, self.belief.getNumCols() - 1)
                self.particles[(row, col)] += 1
        else:
            # 从可能的位置中随机选择NUM_PARTICLES个（可重复）
            for _ in range(self.NUM_PARTICLES):
                position = random.choice(possiblePositions)
                self.particles[position] += 1
        # END_YOUR_CODE
        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles| and ensures that the probabilites sum to 1
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    '''
    # Problem f
    # Function: Observe:(Takes |self.particles| and updates them based on the distance observation and your position )
    # Params:
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #             
    # Notes:
    # - your reweight operation should correspond to your initialization!!
    # - update the particle distribution with the emission probability associated with the observed distance
    # - tiles with 0 probabilities (i.e. those with no particles) do not need to be updated.
    # - this makes particle filtering runtime to be O(|particles|).
    '''

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # Reweight the particles
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        newParticles = collections.defaultdict(int)

        # 遍历当前所有有粒子的位置
        for tile, count in self.particles.items():
            if count > 0:  # 只考虑有粒子的位置
                row, col = tile
                # 转换为坐标
                tileY = util.rowToY(row)
                tileX = util.colToX(col)
                # 计算真实距离
                trueDist = math.sqrt((agentX - tileX) ** 2 + (agentY - tileY) ** 2)
                # 计算观测到observedDist的概率
                emissionProb = util.pdf(trueDist, Const.SONAR_STD, observedDist)
                # 更新粒子权重
                newParticles[tile] = count * emissionProb

        # 更新粒子分布
        self.particles = newParticles
        # END_YOUR_CODE

        # Resample the particles
        # Now we have the reweighted (unnormalized) distribution, we can now re-sample the particles from 
        # this distribution, choosing a new grid location for each of the |self.NUM_PARTICLES| new particles.
        newParticles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            p = util.weightedRandomChoice(self.particles)
            newParticles[p] += 1  
        self.particles = newParticles
        self.updateBelief()

  
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # We have a particle distribution at current time $t$, and we want
    # to propose the particle distribution at time $t+1$. We would like
    # to sample again to see where each particle would end up using the transition model.
    #
    # Notes:
    # - Remember that if there are multiple particles at a particular location,
    #   you will need to call util.weightedRandomChoice() once for each of them!
    # - You should NOT call self.updateBelief() at the end of this function.
    def elapseTime(self) -> None:
        newParticles = collections.defaultdict(int)
        for particle in self.particles:
            for _ in range(self.particles[particle]):
                newParticle = util.weightedRandomChoice(self.transProbDict[particle])
                newParticles[newParticle] += 1
        self.particles = newParticles

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile
    def getBelief(self) -> Belief:
        return self.belief
