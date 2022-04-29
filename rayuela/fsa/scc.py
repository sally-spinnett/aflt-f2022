import numpy as np
from numpy import linalg as LA

from collections import deque

from rayuela.base.semiring import Boolean, Real
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import MinimizeState

class SCC:

    def __init__(self, fsa):
        self.fsa = fsa

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self) -> "list[frozenset]":
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """
		# Homework 3: Question 4
        scc = []
        visited = set([])
        self_rev = self.fsa.reverse()

        for q in self.fsa.finish():
            if q in visited:
                continue
            current = set([])
            Q = [q]
            while Q:
                q_ = Q.pop()
                current.add(q_)
                visited.add(q_)
                for a, j, w in self_rev.arcs():
                    if j not in visited:
                        Q.add(j)
            scc.append(current)
        return scc


        
