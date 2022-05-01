from collections import defaultdict
import numpy as np
from numpy import linalg as LA
from frozendict import frozendict

from rayuela.base.datastructures import PriorityQueue
from rayuela.base.semiring import Real, Semiring

from rayuela.fsa.state import State
from rayuela.fsa.scc import SCC

from rayuela.fsa.fsa import FSA
from collections import defaultdict as dd

class Strategy:
	VITERBI = 1
	BELLMANFORD = 2
	DIJKSTRA = 3
	LEHMANN = 4
	JOHNSON = 5
	FIXPOINT = 6
	DECOMPOSED_LEHMANN = 7

class Pathsum:

	def __init__(self, fsa):

		# basic FSA stuff
		self.fsa = fsa
		self.R = fsa.R
		self.N = self.fsa.num_states

		# state dictionary
		self.I = {}
		for n, q in enumerate(self.fsa.Q):
			self.I[q] = n

		# lift into the semiring
		self.W = self.lift()

	def _convert(self):
		mat = np.zeros((self.N, self.N))
		for n in range(self.N):
			for m in range(self.N):
				mat[n, m] = self.W[n, m].score
		return mat

	def max_eval(self):
		# computes the largest eigenvalue
		mat = self._convert()
		if len(mat) == 0:
			return 0.0
		vals = []
		for val in LA.eigvals(mat):
			vals.append(np.abs(val))
		return np.max(vals)

	def lift(self):
		""" creates the weight matrix from the automaton """
		W = self.R.zeros(self.N, self.N)
		for p in self.fsa.Q:
			for a, q, w in self.fsa.arcs(p):
				W[self.I[p], self.I[q]] += w
		return W

	def pathsum(self, strategy):
		if strategy == Strategy.DIJKSTRA:
			assert self.R.superior, "Dijkstra's requires a superior semiring"
			return self.dijkstra_early()

		elif strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_pathsum()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_pathsum()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_pathsum()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_pathsum()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_pathsum()

		elif strategy == Strategy.DECOMPOSED_LEHMANN:
			return self.decomposed_lehmann_pathsum()

		else:
			raise NotImplementedError

	def forward(self, strategy):
		
		if strategy == Strategy.DIJKSTRA:
			assert self.R.superior, "Dijkstra's requires a superior semiring"
			return self.dijkstra_fwd()

		if strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_fwd()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_fwd()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_fwd()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_fwd()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_fwd()

		else:
			raise NotImplementedError

	def backward(self, strategy):
		if strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_bwd()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_bwd()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_bwd()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_bwd()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_bwd()

		else:
			raise NotImplementedError

	def allpairs(self, strategy):
		if strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann()
		
		elif strategy == Strategy.FIXPOINT:
			raise self.fixpoint()

		else:
			raise NotImplementedError

	def allpairs_pathsum(self, W):
		pathsum = self.R.zero
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				pathsum += self.fsa.λ[p] * W[p, q] * self.fsa.ρ[q]
		return pathsum

	def allpairs_fwd(self, W):
		α = self.R.chart()
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				α[q] += self.fsa.λ[p] * W[p, q]
		return frozendict(α)

	def allpairs_bwd(self, W):
		𝜷 = self.R.chart()
		W = self.lehmann()
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				𝜷[p] += W[p, q] * self.fsa.ρ[q]
		return frozendict(𝜷)

	def viterbi_pathsum(self):
		pathsum = self.R.zero
		# 𝜷 = self.viterbi_bwd()
		# for q in self.fsa.Q:
		# 	pathsum += self.fsa.λ[q] * 𝜷[q]
		α = self.viterbi_fwd()
		for q in self.fsa.Q:
			pathsum += self.fsa.ρ[q] * α[q]
		return pathsum

	def viterbi_fwd(self):
		""" Assignment 2 Question 2.1 """
		assert self.fsa.acyclic

		# chart
		α = self.R.chart()

		# base case (paths of length 0)
		for q, w in self.fsa.I:
			α[q] = w
		
		# recursion
		for p in self.fsa.toposort(rev=False):
			for _, q, w in self.fsa.reverse().arcs(p):
				α[p] += α[q] * w
		
		return frozendict(α)

	def viterbi_bwd(self) -> "defaultdict[State, Semiring]":
		""" The Viterbi algorithm run backwards"""

		assert self.fsa.acyclic


		# chart
		𝜷 = self.R.chart()

		# base case (paths of length 0)
		for q, w in self.fsa.F:
			𝜷[q] = w

		# recursion
		for p in self.fsa.toposort(rev=True):
			for _, q, w in self.fsa.arcs(p):
				𝜷[p] += 𝜷[q] * w

		return 𝜷

	def dijkstra_early(self):
		""" Dijkstra's algorithm with early stopping."""
		raise NotImplementedError


	def dijkstra_fwd(self, I=None):
		""" Dijkstra's algorithm without early stopping. """

		assert self.fsa.R.superior

		# initialization
		α = self.R.chart()
		agenda = PriorityQueue(R=self.fsa.R)
		popped = set([]) 

		# base case
		if I is None:
			for q, w in self.fsa.I:
				agenda.push(q, w)
		else:
			for q in I:
				agenda.push(q, self.R.one)

		# main loop
		while agenda:
			i, v = agenda.pop()
			popped.add(i)
			α[i] += v

			for _, j, w in self.fsa.arcs(i):
				if j not in popped:
					agenda.push(j, v * w)

		return α

	def _lehmann(self, zero=True):
		"""
		Lehmann's (1977) algorithm.
		"""

		# initialization
		V = self.W.copy()
		U = self.W.copy()


		# basic iteration
		for j in range(self.N):
			V, U = U, V
			V = self.R.zeros(self.N, self.N)
			for i in range(self.N):
				for k in range(self.N):
					# i ➙ j ⇝ j ➙ k
					V[i,k] = U[i,k] + U[i,j] * U[j,j].star() * U[j,k]

		# post-processing (paths of length zero)
		if zero:
			for i in range(self.N):
				V[i,i] += self.R.one


		return V

	def lehmann(self, zero=True):

		V = self._lehmann(zero=zero)

		W = {}
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				if p in self.I and q in self.I:
					W[p, q] = V[self.I[p], self.I[q]]
				elif p == q and zero:
					W[p, q] = self.R.one
				else:
					W[p, q] = self.R.zero

		return frozendict(W)

	def lehmann_pathsum(self): return self.allpairs_pathsum(self.lehmann())
	def lehmann_fwd(self): return self.allpairs_fwd(self.lehmann())
	def lehmann_bwd(self): return self.allpairs_bwd(self.lehmann())

	def decomposed_lehmann_pathsum(self) -> Semiring:
		# Homework 3: Question 4
		raise NotImplementedError

	def bellmanford_pathsum(self) -> Semiring:
		pathsum = self.R.zero
		𝜷 = self.bellmanford_bwd()
		for q in self.fsa.Q:
			pathsum += self.fsa.λ[q] * 𝜷[q]
		return pathsum

	def bellmanford_fwd(self) -> frozendict[State, Semiring]:
		# TODO： fix that this type annotation that was commented out
		# raise NotImplementedError
		""" 
		The Bellman-Ford algorithm run forwards.
		Using Tropical Semiring.
		"""
		assert self.R.idempotent
		# Step 1: Initialize distances from src to all other vertices as INFINITE
		# dist = dd(lambda : self.R.zero)
		dist = self.R.chart()
		for i, weight in self.fsa.I:
			dist[i] = weight
		# Step 2: Relax all edges |V| - 1 times, |V| because we added a dummy state
		for _ in range(len(self.fsa.Q)):
			# for every arc in the fsa
			for p in self.fsa.Q:
				for _, q, w in self.fsa.arcs(p):
					dist[q] = dist[q] + dist[p]*w
		# Step 3: Check for negative-weight cycles.
		for p in self.fsa.Q:
			for _, q, w in self.fsa.arcs(p):
				if p in dist and q in dist:
					assert dist[q] == dist[q] + dist[p]*w
		return frozendict(dist)



	def bellmanford_bwd(self) -> frozendict[State, Semiring]:
		# TODO: fix that I commented this out
		return Pathsum(self.fsa.reverse()).bellmanford_fwd()


	def johnson(self) -> "defaultdict[(State,State), Semiring]":
		# raise NotImplementedError
		"""
		Assignment 5 Question 2
		"""
		path = self.R.chart()
		# Step 1: Bellman-ford to calculate forward weights
		W = self.forward(Strategy.BELLMANFORD)
		W = frozendict({k: ~v for k,v in W.items()})
		# Step 2: weight pushing using forward weights as potential function
		from rayuela.fsa.transformer import Transformer
		pfsa = Transformer._push(self.fsa, W)
		# Step 3: Apply Dijkstra's algo once for every vertex
		for q in pfsa.Q:
			for k, v in self.dijkstra_fwd(I=[q]).items:
				path[(q,k)] = v
		return path



	def johnson_pathsum(self): return self.allpairs_pathsum(self.johnson())
	def johnson_fwd(self): return self.allpairs_fwd(self.johnson())
	def johnson_bwd(self): return self.allpairs_bwd(self.johnson())

	def fixpoint(self):
		raise NotImplementedError

	def fixpoint_pathsum(self): return self.allpairs_pathsum(self.fixpoint())
	def fixpoint_fwd(self): return self.allpairs_fwd(self.fixpoint())
	def fixpoint_bwd(self): return self.allpairs_bwd(self.fixpoint())
