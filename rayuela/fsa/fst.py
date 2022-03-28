from frozendict import frozendict
from itertools import product

from rayuela.base.semiring import Boolean
from rayuela.base.misc import epsilon_filter
from rayuela.base.symbol import Sym, ε, ε_1, ε_2

from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State, PairState


class FST(FSA):

	def __init__(self, R=Boolean):

		# DEFINITION
		# A weighted finite-state transducer is a 8-tuple <Σ, Δ, Q, F, I, δ, λ, ρ> where
		# • Σ is an alphabet of symbols;
		# • Δ is an alphabet of symbols;
		# • Q is a finite set of states;
		# • I ⊆ Q is a set of initial states;
		# • F ⊆ Q is a set of final states;
		# • δ is a finite relation Q × Σ × Δ × Q × R;
		# • λ is an initial weight function;
		# • ρ is a final weight function.

		# NOTATION CONVENTIONS
		# • single states (elements of Q) are denoted q
		# • multiple states not in sequence are denoted, p, q, r, ...
		# • multiple states in sequence are denoted i, j, k, ...
		# • symbols (elements of Σ and Δ) are denoted lowercase a, b, c, ...
		# • single weights (elements of R) are denoted w
		# • multiple weights (elements of R) are denoted u, v, w, ...

		super().__init__(R=R)

		# alphabet of output symbols
		self.Delta = set()

	def add_arc(self, i, a, b, j, w=None):
		if w is None: w = self.R.one

		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(a, Sym): b = Sym(b)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.Delta.add(b)
		self.δ[i][(a, b)][j] += w

	def set_arc(self, i, a, b, j, w):
		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(a, Sym): b = Sym(b)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.Delta.add(b)
		self.δ[i][(a, b)][j] = w

	def freeze(self):
		self.Sigma = frozenset(self.Sigma)
		self.Delta = frozenset(self.Delta)
		self.Q = frozenset(self.Q)
		self.δ = frozendict(self.δ)
		self.λ = frozendict(self.λ)
		self.ρ = frozendict(self.ρ)

	def arcs(self, i, no_eps=False):
		for ab, T in self.δ[i].items():
			if no_eps and ab == (ε, ε):
				continue
			for j, w in T.items():
				if w == self.R.zero:
					continue
				yield ab, j, w

	def accept(self, string1, string2):
		""" determines whether a string is in the language """
		# Requires composition
		raise NotImplementedError

	def top_compose(self, fst):
		# Homework 3 
		"""
		on-the-fly weighted top composition
		"""

		# the two machines need to be in the same semiring
		assert self.R == fst.R 

		# the output alphabet of self must match the input alphabet of fst
		assert self.Delta == fst.Sigma

		# add initial states
		product_fst = FST(R=self.R)
		for (q1, w1), (q2, w2) in product(self.I, fst.I):
			product_fst.add_I(PairState(q1, q2), w=w1 * w2)
		
		self_initials = {q: w for q, w in self.I}
		fst_initials = {q: w for q, w in fst.I}

		visited = set([(i1, i2) for i1, i2 in product(self_initials, fst_initials)])
		stack = [(i1, i2) for i1, i2 in product(self_initials, fst_initials)]

		self_finals = {q: w for q, w in self.F}
		fsa_finals = {q: w for q, w in fst.F}

		while stack:
			q1, q2, qf = stack.pop()

			E1 = [ab for (ab, j, w) in self.arcs(q1)]
			E2 = [ab for (ab, j, w) in fst.arcs(q2)]

			M = [((ab1, j1, w1), (ab2, j2, w2))
				 for (ab1, j1, w1), (ab2, j2, w2) in product(E1, E2)
				 if ab1[1] == ab2[0]]

			for (ab1, j1, w1), (ab2, j2, w2) in M:

				product_fst.set_arc(
					PairState(q1, q2), ab1,
					PairState(j1, j2), w=w1*w2)

				if (j1, j2) not in visited:
					stack.append((j1, j2))
					visited.add((j1, j2))

			# final state handling
			if q1 in self_finals and q2 in fsa_finals:
				product_fst.add_F(
					PairState(q1, q2), w=self_finals[q1] * fsa_finals[q2])

		return product_fst

	def bottom_compose(self, fst):
		# Homework 3
		raise NotImplementedError
