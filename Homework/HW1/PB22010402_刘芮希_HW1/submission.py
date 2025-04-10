from typing import List, Tuple
import heapq

from mapUtil import (
    CityMap,
    computeDistance,
    createUSTCMap,
    createHefeiMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch

# BEGIN_YOUR_CODE (You may add some codes here to assist your coding below if you want, but don't worry if you deviate from this.)

# END_YOUR_CODE

# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(location=self.startLocation, memory=None)
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags.get(state.location, [])
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        successors = []
        for neighbor, cost in self.cityMap.distances.get(state.location, {}).items():
            action = neighbor
            newState = State(location=neighbor, memory=None)
            successors.append((action, newState, cost))
        return successors
        # END_YOUR_CODE


########################################################################################
# Problem 1b: Custom -- Plan a Route through USTC


def getUSTCShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of USTC, specifying your own
    `startLocation`/`endTag`.

    Run `python mapUtil.py > readableUSTCMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/USTC-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "coffee", "food")
    """
    cityMap = createUSTCMap()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = "10588133363"
    endTag = "landmark=west_campus_library"
    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(location=self.startLocation, memory=frozenset(self.waypointTags))
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (
            not state.memory  # All waypoint tags have been covered
            and self.endTag in self.cityMap.tags.get(state.location, [])
        )
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
        successors = []
        currentLocation = state.location
        remainingTags = state.memory

        # Iterate over neighbors of the current location
        for neighbor, cost in self.cityMap.distances.get(currentLocation, {}).items():
            # Compute the new set of remaining tags after moving to the neighbor
            newTags = set(remainingTags)
            for tag in self.cityMap.tags.get(neighbor, []):
                if tag in newTags:
                    newTags.remove(tag)

            # Create the new state
            newState = State(location=neighbor, memory=frozenset(newTags))
            successors.append((neighbor, newState, cost))

        return successors
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through USTC


def getUSTCWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of USTC, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableUSTCMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createUSTCMap()
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    startLocation = "10588133363"
    waypointTags = ["landmark=8348", "landmark=also_west_lake","landmark=west_campus_library"]
    endTag = "landmark=art_teaching_centre"
    # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def __init__(self):
            # BEGIN_YOUR_CODE (our solution is 3 line of code, but don't worry if you deviate from this)
            self.problem = problem
            self.heuristic = heuristic
            # END_YOUR_CODE

        @property
        def startLocation(self) -> str:
            return self.problem.startLocation

        @property
        def endTag(self) -> str:
            return self.problem.endTag
        
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return self.problem.startState()
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return self.problem.isEnd(state)
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
            successors = []
            current_h = self.heuristic.evaluate(state)
            for action, newState, cost in self.problem.successorsAndCosts(state):
                new_cost = cost + self.heuristic.evaluate(newState) - current_h
                successors.append((action, newState, new_cost))
            return successors
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3c: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        endLocation = locationFromTag(endTag, cityMap)
        if endLocation is None:
            raise ValueError("No location with the specified endTag found in cityMap.")
        self.endLocation = endLocation
        self.endGeo = cityMap.geoLocations[endLocation]
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        currentGeo = self.cityMap.geoLocations[state.location]
        return computeDistance(currentGeo, self.endGeo)
        # END_YOUR_CODE


########################################################################################
# Problem 3e: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        # Precompute
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        self.endTag = endTag
        self.cityMap = cityMap
        self.targetLocations = []
        for loc, tags in cityMap.tags.items():
            if endTag in tags:
                self.targetLocations.append(loc)
        if not self.targetLocations:
            raise ValueError("No location with the specified endTag found in cityMap.")
        self.distanceToGoal = {}  
        pq = []  
        for target in self.targetLocations:
            self.distanceToGoal[target] = 0.0
            heapq.heappush(pq, (0.0, target))
        while pq:
            d, loc = heapq.heappop(pq)
            if d > self.distanceToGoal.get(loc, float('inf')):
                continue
            for neighbor, cost in cityMap.distances.get(loc, {}).items():
                newd = d + cost
                if newd < self.distanceToGoal.get(neighbor, float('inf')):
                    self.distanceToGoal[neighbor] = newd
                    heapq.heappush(pq, (newd, neighbor))
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.distanceToGoal.get(state.location, float('inf'))
        # END_YOUR_CODE


########################################################################################
# Problem 3f: Plan a Route through Hefei with or without a Heuristic

def getHefeiShortestPathProblem(cityMap: CityMap) -> ShortestPathProblem:
    """
    Create a search problem using the map of Hefei
    """
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ShortestPathProblem(startLocation, endTag, cityMap)
    # END_YOUR_CODE

def getHefeiShortestPathProblem_withHeuristic(cityMap: CityMap) -> ShortestPathProblem:
    """
    Create a search problem with Heuristic using the map of Hefei
    """
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    baseProblem = ShortestPathProblem(startLocation, endTag, cityMap)
    heuristic = StraightLineHeuristic(endTag, cityMap)
    return aStarReduction(baseProblem, heuristic)
    # END_YOUR_CODE
