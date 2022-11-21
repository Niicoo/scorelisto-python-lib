import heapq
import logging
from typing import Dict, List, Tuple

# Get Logger
logger = logging.getLogger(__name__)


class Vertex:
    def __init__(self, id: str):
        self.id = id
        self.connections: Dict["Vertex", float] = {}
        self.distance: float = float("inf")
        self.visited: bool = False
        self.previous = None

    def __lt__(self, other: "Vertex"):
        return((self.distance > other.distance) ^ (self.distance < other.distance))

    def __str__(self):
        return(f"Vertex: {self.id} (connections: {[(vertex.id, weight) for vertex, weight in self.connections.items()]})")

    def reset(self):
        self.distance = float("inf")
        self.visited = False
        self.previous = None

    def addConnection(self, vertex: "Vertex", weight: float = 0):
        self.connections[vertex] = weight

    def getConnections(self) -> List["Vertex"]:
        return(list(self.connections.keys()))

    def getId(self) -> str:
        return(self.id)

    def getWeight(self, id: str) -> float:
        return(self.connections[id])

    def setDistance(self, distance: float):
        self.distance = distance

    def getDistance(self) -> float:
        return(self.distance)

    def setPrevious(self, previous: "Vertex"):
        self.previous = previous

    def getPrevious(self) -> "Vertex":
        return(self.previous)

    def setVisited(self):
        self.visited = True

    def isVisited(self) -> bool:
        return self.visited


class Graph:
    def __init__(self):
        self.vertices: Dict[str, Vertex] = {}

    def __str__(self):
        output_str = "Graph("
        for vertex in self.vertices.values():
            output_str += vertex.__str__() + "\n"
        output_str += ")"
        return output_str

    def addVertex(self, id: str):
        logger.debug(f"Adding vertex with ID: {id}")
        self.vertices[id] = Vertex(id)

    def getVertex(self, id: str) -> Vertex:
        return(self.vertices[id])

    def addEdge(self, id_from: str, id_to: str, weight: float = 0, two_ways: bool = False):
        logger.debug(f"Adding connection {id_from} -> {id_to} [two way connection: {two_ways}]")
        self.vertices[id_from].addConnection(self.vertices[id_to], weight)
        if two_ways:
            self.vertices[id_to].addConnection(self.vertices[id_from], weight)

    def _resetVertices(self):
        for v in self.vertices.values():
            v.reset()

    def _getUnvisitedQueue(self) -> List[Tuple[float, Vertex]]:
        unvisited_queue = [(v.getDistance(), v) for v in self.vertices.values() if not v.visited]
        heapq.heapify(unvisited_queue)
        return unvisited_queue

    def _shortest(self, v: Vertex, path: List[str]):
        if(v.getPrevious()):
            path.append(v.getPrevious().getId())
            self._shortest(v.getPrevious(), path)

    def performDijkstraShortestPath(self, id_start: str, id_target: str) -> List[str]:
        self._resetVertices()
        self.vertices[id_start].setDistance(0)
        unvisited_queue = self._getUnvisitedQueue()
        while(len(unvisited_queue)):
            _, current = heapq.heappop(unvisited_queue)
            current.setVisited()
            for next in current.getConnections():
                if(next.isVisited()): continue
                new_dist = current.getDistance() + current.getWeight(next)
                if(new_dist < next.getDistance()):
                    next.setDistance(new_dist)
                    next.setPrevious(current)
            unvisited_queue = self._getUnvisitedQueue()
        path = [id_target]
        self._shortest(self.vertices[id_target], path)
        path = path[::-1]
        if path[0] != id_start:
            raise RuntimeError(f"Fail to find a path between vertex '{id_start}' and vertex '{id_target}'")
        return path


if __name__ == "__main__":
    g = Graph()
    g.addVertex('a')
    g.addVertex('b')
    g.addVertex('c')
    g.addVertex('d')
    g.addVertex('e')
    g.addVertex('f')
    g.addVertex('g')
    g.addVertex('h')
    g.addVertex('i')
    g.addVertex('j')
    g.addVertex('k')
    g.addEdge('a', 'c', 5)  
    g.addEdge('a', 'b', 5)
    g.addEdge('a', 'g', 1)
    g.addEdge('b', 'd', 4)
    g.addEdge('b', 'h', 3)
    g.addEdge('c', 'd', 2)
    g.addEdge('c', 'e', 1)
    g.addEdge('d', 'k', 1)
    g.addEdge('e', 'f', 1)
    g.addEdge('g', 'b', 2)
    g.addEdge('g', 'h', 6)
    g.addEdge('h', 'k', 6)
    g.addEdge('h', 'i', 1)
    g.addEdge('j', 'g', 2)
    print(g)
    path = g.performDijkstraShortestPath('a', 'h')
    print('The shortest path : %s' % path)
