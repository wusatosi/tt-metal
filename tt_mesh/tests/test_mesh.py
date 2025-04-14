import time

from tt_mesh.mesh import Mesh


def test_get_line():
    mesh = Mesh(8, 4, toroidal=False)
    line_segments = mesh.get_line(32, ring=True)
    assert line_segments is not None
    print(line_segments)


def test_get_lines():
    t0 = time.time()
    h, w = 16, 16
    mesh = Mesh(h, w, toroidal=False)
    line_segments = mesh.get_lines([10 for _ in range(25)], ring=True)
    print("Time taken:", time.time() - t0)

    assert line_segments is not None
    print(line_segments)

    segment_coordinates = set()
    for line_segment in line_segments:
        _, _, x, y = line_segment
        segment_coordinates.add((x, y))
    assert len(segment_coordinates) == h * w


if __name__ == "__main__":
    test_get_lines()
