from itertools import product

from tt_mesh.solver import SATSolver


class Mesh:
    def __init__(self, width, height, toroidal=False):
        self.width = width
        self.height = height
        self.toroidal = toroidal

    def get_neighbors(self, x, y):
        neighbors = []
        if self.toroidal:
            neighbors.append(((x + 1) % self.width, y))
            neighbors.append((x, (y + 1) % self.height))
            neighbors.append(((x - 1) % self.width, y))
            neighbors.append((x, (y - 1) % self.height))
        else:
            if x + 1 < self.width:
                neighbors.append((x + 1, y))
            if y + 1 < self.height:
                neighbors.append((x, y + 1))
            if x - 1 >= 0:
                neighbors.append((x - 1, y))
            if y - 1 >= 0:
                neighbors.append((x, y - 1))

        return neighbors

    def get_line(self, length, ring=False):
        solver = SATSolver()
        vars = {}
        for i in range(length):
            for x, y in product(range(self.width), range(self.height)):
                var = solver.new_var(f"line_segment_{i}_at_cell_{x}_{y}")
                vars[(i, x, y)] = var

        for i in range(length):
            solver.exactly_one([vars[(i, x, y)] for x, y in product(range(self.width), range(self.height))])

        for x, y in product(range(self.width), range(self.height)):
            solver.at_most_one([vars[(i, x, y)] for i in range(length)])

        for x, y in product(range(self.width), range(self.height)):
            for i in range(length if ring else length - 1):
                solver.implies(
                    vars[(i, x, y)], [vars[((i + 1) % length, nx, ny)] for nx, ny in self.get_neighbors(x, y)]
                )

        if solver.solve():
            model = solver.get_true_vars()
            line_segments = []
            for i in range(length):
                for x, y in product(range(self.width), range(self.height)):
                    if model[vars[(i, x, y)]]:
                        line_segments.append((i, x, y))
            return line_segments

        return None

    def get_lines(self, lengths, ring=False):
        solver = SATSolver()
        vars = {}
        for line, length in enumerate(lengths):
            for i in range(length):
                for x, y in product(range(self.width), range(self.height)):
                    var = solver.new_var(f"line_{line}_segment_{i}_at_cell_{x}_{y}")
                    vars[(line, i, x, y)] = var

        for line, length in enumerate(lengths):
            for i in range(length):
                solver.exactly_one([vars[(line, i, x, y)] for x, y in product(range(self.width), range(self.height))])

        for x, y in product(range(self.width), range(self.height)):
            solver.at_most_one([vars[(line, i, x, y)] for i in range(length) for line, length in enumerate(lengths)])

        for x, y in product(range(self.width), range(self.height)):
            for line, length in enumerate(lengths):
                for i in range(length if ring else length - 1):
                    solver.implies(
                        vars[(line, i, x, y)],
                        [vars[(line, (i + 1) % length, nx, ny)] for nx, ny in self.get_neighbors(x, y)],
                    )

        if solver.solve():
            model = solver.get_true_vars()
            line_segments = []
            for line, length in enumerate(lengths):
                for i in range(length):
                    for x, y in product(range(self.width), range(self.height)):
                        if model[vars[(line, i, x, y)]]:
                            line_segments.append((line, i, x, y))
            return line_segments

        return None

    def __str__(self):
        return f"Mesh(width={self.width}, height={self.height}, toroidal={self.toroidal})"
