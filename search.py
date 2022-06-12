# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"


    currentState = problem.getStartState()
    frontier = util.Stack()
    explored = [currentState]
    chain = {}
    currentStateRaw = (currentState, 'None', 0)


    while not problem.isGoalState(currentStateRaw[0]):
        children = problem.getSuccessors(currentStateRaw[0])
        for child in children:
            if child[0] not in explored:
                frontier.push(child)
                chain[child]=currentStateRaw
        #print(f'frontier.isEmpty() = {frontier.isEmpty()}')
        if not frontier.isEmpty():
            currentStateRaw  = frontier.pop()
            #print(f' currentStateRaw = { currentStateRaw}')
            #print(f'currentStateRaw[0] = {currentStateRaw[0]}, currentStateRaw[1] = {currentStateRaw[1]}')
            currentState = currentStateRaw[0]

            #print(f'currentState = {currentState}')
            #print(f'currentState is goal state: {problem.isGoalState(currentState)}')
            #print(f'chain = {chain.items()}')
            explored.append(currentState)


        else: return []



    path = []
    i=0
    state = currentStateRaw
    while state[0] != problem.getStartState() and i<20:
        #print(f'state = {state}, chain[state] = {chain[state]}')
        path.insert(0, state[1])
        #print(f'path = {path}')
        state = chain[state]



    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    currentState = problem.getStartState()
    frontier = util.Stack()
    explored = [currentState]
    path = []
    # saves the last coordinates where the path branches to backtrack later
    lastBranchingPathIndexes = []

    while not problem.isGoalState(currentState):
        children = problem.getSuccessors(currentState)
        # number of children of the current node, which are not yet explored
        numUnexploredChildren = 0
        for child in children:
            if child[0] not in explored:
                frontier.push(child)
                numUnexploredChildren += 1
        currentStateWithDirection = frontier.pop()
        currentState = currentStateWithDirection[0]

        # no more children of current node to explore
        if numUnexploredChildren == 0:
            if len(lastBranchingPathIndexes) > 0:
                # cut nodes from the end of path up to last branching of path
                path = path[:lastBranchingPathIndexes.pop()]
                # path branches and multiple children can be explored
        elif numUnexploredChildren > 1:
            while numUnexploredChildren > 1:
                lastBranchingPathIndexes.append(len(path))
                numUnexploredChildren -= 1
        explored.append(currentState)

        # alternative for testing (in that case the line that returns the path has to be exchanged too):
        # path.append(currentStateWithDirection)
        path.append(currentStateWithDirection[1])
    # alternative for testing:
    # return [item[1] for item in path]
    return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
