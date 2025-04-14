from solver.solver import SATSolver


def test_named_vars():
    s = SATSolver()

    a = s.new_var("a")
    b = s.new_var("b")
    c = s.new_var("c")

    s.at_least_one([a, b, c])  # at most one of a, b, c is true
    s.implies(a, [b, c])  # a => b ∨ c

    assert s.solve()
    model = s.get_true_vars()
    print(model)  # {a: True, b: True, c: False} or similar (only one of b/c is True)


if __name__ == "__main__":
    test_named_vars()
