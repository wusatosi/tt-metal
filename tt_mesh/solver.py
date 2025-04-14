from itertools import combinations

from pysat.solvers import Glucose3


class BoolVar:
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __neg__(self):
        return -self.index

    def __int__(self):
        return self.index

    def __repr__(self):
        return f"BoolVar({self.name})"


class SATSolver:
    def __init__(self):
        self.solver = Glucose3()
        self.var_counter = 1
        self.var_map = {}
        self.rev_map = {}

    def new_var(self, name):
        if name in self.var_map:
            return self.var_map[name]
        var = BoolVar(name, self.var_counter)
        self.var_map[name] = var
        self.rev_map[self.var_counter] = var
        self.var_counter += 1
        return var

    def _add_clause(self, literals):
        self.solver.add_clause(literals)

    def at_most_one(self, vars):
        lits = [v.index for v in vars]
        for v1, v2 in combinations(lits, 2):
            self._add_clause([-v1, -v2])

    def at_least_one(self, vars):
        lits = [v.index for v in vars]
        self._add_clause(lits)

    def exactly_one(self, vars):
        self.at_least_one(vars)
        self.at_most_one(vars)

    def implies(self, v, vars):
        head = -v.index
        body = [v.index for v in vars]
        self._add_clause([head] + body)

    def solve(self):
        return self.solver.solve()

    def get_true_vars(self):
        raw_model = self.solver.get_model()
        result = {}
        for lit in raw_model:
            idx = abs(lit)
            val = lit > 0
            if idx in self.rev_map:
                result[self.rev_map[idx]] = val
        return result

    def delete(self):
        self.solver.delete()
