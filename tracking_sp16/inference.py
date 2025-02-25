# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import random
import busters
import game

from util import manhattanDistance


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        z = float(self.total())
        if z  == 0: return 

        for key, value in self.items():
            self[key] = float(value) / float(z)

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        dictCopy = self.copy()
        dictCopy.normalize()
        currInterval = 0.0
        intervals = dictCopy

        for k, v in dictCopy.items():
            if not float(dictCopy[k] == 0):
                currInterval += float(dictCopy[k])
                intervals[k] = currInterval
            else:
                del intervals[k]

        u = random.random()

        for k, v in dictCopy.items():
            if u < v:
                return k

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
                                    observaton | 
        """
        # "*** YOUR CODE HERE ***"
        # However, there is the special case of jail that we have to handle as well. Specifically, when we capture a ghost and send it to the jail location, our distance sensor deterministically returns None, and nothing else. So, if the ghost's position is the jail position, then the observation is None with probability 1, and everything else with probability 0. 

        # Conversely, if the distance reading is not None, then the ghost is in jail with probability 0. If the distance reading is None, then the ghost is in jail with probability 1. Make sure you handle this special case in your implementation.

        if ghostPosition == jailPosition:
            return 1.0 if (noisyDistance == None) else 0.0

        if noisyDistance == None and ghostPosition != jailPosition:
            return 0.0

        return busters.getObservationProbability(noisyDistance, manhattanDistance(pacmanPosition, ghostPosition))

        


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"

        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()

        for position in self.allPositions:
            newBelief = self.getObservationProb(observation, pacmanPosition, position, jailPosition)
            self.beliefs[position] *= newBelief

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # The elapseTime step should, for this problem, update the belief at every position on the map after one time step elapsing. Your agent has access to the action distribution for the ghost through self.getPositionDistribution. 

        # In order to obtain the distribution over new positions for the ghost, given its previous position, use this line of code:

        # newPosDist = self.getPositionDistribution(gameState, oldGhostPos)
        # print newPosDist

        newBeliefs = DiscreteDistribution()
        
        for position in self.allPositions:
            newPosDist = self.getPositionDistribution(gameState, position)
            for key, value in newPosDist.items():
                b = self.beliefs[position]
                newBeliefs[key] += value * b
                
        self.beliefs = newBeliefs




    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        particleAmt = self.numParticles
        legals = self.legalPositions
        self.particles = []
        particlesPlaced = 0

        # particlesPerSquare = particleAmt/len(legals)
        while particlesPlaced < particleAmt:
            for legal in legals:
                self.particles.append(legal)
                particlesPlaced += 1


        # for legal in legals:
        #     particlesPlaced = 0
        #     while (particlesPlaced < particlesPerSquare):
        #         self.particles.append(legal)
        #         particlesPlaced += 1


    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"

        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        weightDist = DiscreteDistribution()

        for particle in self.particles:
            weightDist[particle] += self.getObservationProb(observation, pacmanPosition, particle, jailPosition)
            # += because multiple particles in same location!

        weightDist.normalize()

        newParticles = [weightDist.sample() for i in range(int(len(self.particles)))]

        if weightDist.total() == 0:
            self.initializeUniformly(gameState)
            return

        self.particles = newParticles


        # for position in self.allPositions:
        #     newBelief = self.getObservationProb(observation, pacmanPosition, position, jailPosition)
        #     self.beliefs[position] *= newBelief

        # self.beliefs.normalize()

        # There is one special case that a correct implementation must handle. 
        # When all particles receive zero weight, the list of particles should be 
        # reinitialized by calling initializeUniformly. The total method of the DiscreteDistribution may be useful.

        # for position in self.allPositions:
        #     newBelief = self.getObservationProb(observation, pacmanPosition, position, jailPosition)
        #     self.beliefs[position] *= newBelief

        #     # if all particles have 0 weight: reinitialize
        #     if beliefs.total() == 0:
        #         self.initializeUniformly(gameState)

        # self.beliefs.normalize()


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"

        # newBeliefs = DiscreteDistribution()
        
        # for position in self.allPositions:
        #     newPosDist = self.getPositionDistribution(gameState, position)
        #     for key, value in newPosDist.items():
        #         b = self.beliefs[position]
        #         newBeliefs[key] += value * b
                
        # self.beliefs = newBeliefs

        # newParticles = []
        from collections import Counter

        counter = Counter(self.particles)
        index = 0
        for particle, count in counter.items():
            newPosDist = self.getPositionDistribution(gameState, particle)
            for i in range(count):
                self.particles[index] = newPosDist.sample()
                index += 1


        # for i in range(len(self.particles)):
        #     newPosDist = self.getPositionDistribution(gameState, self.particles[i])
        #     self.particles[i] = newPosDist.sample()

        # self.particles = newParticles


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        distrib = DiscreteDistribution()
        particlePos = self.particles
        for particle in particlePos:
            distrib[particle] += 1.0
        distrib.normalize()
        return distrib


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        ghostPositions = [perm for perm in itertools.product(self.legalPositions, repeat = self.numGhosts) if len(set(perm)) == self.numGhosts]
        random.shuffle(ghostPositions)
        numGhostPositions = len(ghostPositions)

        # i = 0
        # while i + numGhostPositions < self.numParticles:
        #     self.particles += ghostPositions
        #     i += numGhostPositions
        # self.particles += ghostPositions[:self.numParticles - i]

        if self.numParticles <= numGhostPositions:
            self.particles = ghostPositions[:self.numParticles]
        else:
            while len(self.particles) != self.numParticles:
                for ghostPosition in ghostPositions:
                    if len(self.particles) != self.numParticles:
                        self.particles.append(ghostPosition)
                    else:
                        break

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"


        # pacmanPosition = gameState.getPacmanPosition()
        # jailPosition = self.getJailPosition()
        # weightDist = DiscreteDistribution()

        # for particle in self.particles:
        #     weightDist[particle] += self.getObservationProb(observation, pacmanPosition, particle, jailPosition)
        #     # += because multiple particles in same location!

        # weightDist.normalize()

        # newParticles = [weightDist.sample() for i in range(int(len(self.particles)))]

        # if weightDist.total() == 0:
        #     self.initializeUniformly(gameState)
        #     return

        # self.particles = newParticles

        # weight and resample the entire list of particles based on the observation of all ghost distances

        pacmanPosition = gameState.getPacmanPosition()
        weightDist = DiscreteDistribution()

        for particle in self.particles:
            particleProb = 1
            for i in range(self.numGhosts):
                jailPosition = self.getJailPosition(i)
                obs = self.getObservationProb(observation[i], pacmanPosition, particle[i], jailPosition)
                particleProb *= obs
            weightDist[particle] += particleProb * self.numGhosts

        weightDist.normalize()
        newParticles = [weightDist.sample() for i in range(int(len(self.particles)))]

        if weightDist.total() == 0:
            self.initializeUniformly(gameState)
            return

        # print(newParticles)
        self.particles = newParticles
        # random.shuffle(self.particles)






    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            prevGhostPositions = list(oldParticle)
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

        # beliefDist = self.getBeliefDistribution()
        # newParticles = []
        # for i in range(len(self.particles)):
        #     new = self.getPositionDistribution(gameState, self.particles[i])
        #     newParticles.append(new.sample())

        # self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
