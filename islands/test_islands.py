from islands import count_islands


def test_case_1():
    grid = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
    ]
    assert count_islands(grid) == 2


def test_case_2():
    grid = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ]
    assert count_islands(grid) == 3


def test_case_3():
    grid = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ]
    assert count_islands(grid) == 2


def test_empty_grid():
    assert count_islands([]) == 0


def test_all_ocean():
    assert count_islands([[0, 0], [0, 0]]) == 0


def test_all_land():
    assert count_islands([[1, 1], [1, 1]]) == 1


def test_single_cell_island():
    assert count_islands([[1]]) == 1


def test_diagonal_cells_are_not_connected():
    grid = [
        [1, 0],
        [0, 1],
    ]
    assert count_islands(grid) == 2
