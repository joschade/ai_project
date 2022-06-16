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
    currentStateRaw, path = ((currentState, 'None', 0), [])

    while not problem.isGoalState(currentStateRaw[0]):
        children = problem.getSuccessors(currentStateRaw[0])
        for child in children:
            if child[0] not in explored:
                childPath = path + [child[1]]
                frontier.push((child, childPath))
        if not frontier.isEmpty():
            currentStateRaw, path = frontier.pop()
            #print(f' currentStateRaw = { currentStateRaw}')
            #print(f'currentStateRaw[0] = {currentStateRaw[0]}, currentStateRaw[1] = {currentStateRaw[1]}')
            currentState = currentStateRaw[0]
            if problem.isGoalState(currentState): 
                return path
            explored.append(currentState)
        else: 
            return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    def stateInFrontier(state, frontier):
        for item in frontier.list:
            if item[0][0] == state: 
                return True
        return False

    currentState = problem.getStartState()
    frontier = util.Queue()
    explored = []
    currentNode = ((currentState, None, 0), None)
    frontier.push(currentNode)

    while not frontier.isEmpty():
        currentNode = frontier.pop()
        explored.append(currentNode[0][0])
        if problem.isGoalState(currentNode[0][0]):
            path = []
            while currentNode[1] is not None:
                path.insert(0, currentNode[0][1])
                currentNode = currentNode[1]
            return path
        for succ in problem.getSuccessors(currentNode[0][0]):
            #print(f'succ[0] not in explored: {succ[0] not in explored}, stateInFrontier(succ[0], frontier): {stateInFrontier(succ[0], frontier)}')
            if succ[0] not in explored and not stateInFrontier(succ[0], frontier):
                frontier.push((succ, currentNode))
                #print(f'{succ, currentNode} pushed')

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    currentState = problem.getStartState()
    frontier = util.PriorityQueue()
    explored = []
    currentNode = (currentState, [], 0)
    frontier.push(currentNode, 0)

    while not frontier.isEmpty():
        currentNode = frontier.pop()
        print(f'currentNode cost: {currentNode[2]}')
        currentState = currentNode[0]
        path = currentNode[1]
        if (currentState not in explored):
            if problem.isGoalState(currentState):
                return path
            else:
                explored.append(currentState)
                for childState, action, childCost in problem.getSuccessors(currentState):
                    newPath = path + [action]
                    childCost = currentNode[2] + childCost
                    child = (childState, newPath, childCost)
                    frontier.update(child, childCost)
    return []

def greedyBestFirstSearch(problem):
    """Search the node with the least immediate cost first"""

    currentState = problem.getStartState()
    frontier = util.PriorityQueue()
    explored = []
    currentNode = (currentState, [], 0)
    frontier.push(currentNode, 0)

    while not frontier.isEmpty():
        currentNode = frontier.pop()
        print(f'currentNode cost: {currentNode[2]}')
        currentState = currentNode[0]
        path = currentNode[1]
        if (currentState not in explored):
            if problem.isGoalState(currentState):
                return path
            else:
                explored.append(currentState)
                for childState, action, childCost in problem.getSuccessors(currentState):
                    newPath = path + [action]
                    child = (childState, newPath, childCost)
                    frontier.update(child, childCost)
    return []

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
gbfs = greedyBestFirstSearch

# Alternatives
GreedyBestFirstSearch = greedyBestFirstSearch
