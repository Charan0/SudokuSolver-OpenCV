import numpy as np


class Solver:
    def __init__(self, puzzle: np.ndarray):
        assert puzzle.shape == (9, 9)
        self.puzzle = puzzle.copy()

    def solve(self):
        if self.is_solvable():
            print('Solvable!')
            self.__solve_helper()
            return self.puzzle
        else:
            print('Not Solvable')
            return None

    def __solve_helper(self):
        """
        This is where the actual logic of the solver resides.
        A backtracking approach of solving a sudoku puzzle
        Inspired from a video by ComputerPhile: https://www.youtube.com/watch?v=G_UYXzGuqvM
        """
        for row_index in range(9):  # Indexing into the rows
            for col_index in range(9):  # Indexing into the columns
                if self.puzzle[row_index, col_index] == 0:  # If the indexed box is empty
                    for val in range(1, 10):  # For values in 1-9
                        # If a particular value can be filled in the indexed box
                        if self.possible(row_index, col_index, val):
                            self.puzzle[row_index, col_index] = val  # Filling the box with the value
                            self.__solve_helper()  # Recursive call to solve

                            if 0 not in self.puzzle:
                                return
                            self.puzzle[row_index, col_index] = 0  # Backtracking
                    return

    def possible(self, row_index: int, col_index: int, value: int):
        """
        :param row_index: Index of the row where we are checking for possibility
        :param col_index: Index of the col where we are checking for possibility
        :param value: The digit we are about to place
        :return: A boolean stating whether or not the digit can be placed in the position
        """
        if self.puzzle[row_index, col_index] != 0:
            print('Grid is not empty')
            return False
        # Fool Proofing the code
        assert value != 0 and value <= 9  # 0 => Empty Grid not an actual value

        # If the value is already present in that row or in that column
        if value in self.puzzle[row_index, :] or value in self.puzzle[:, col_index]:
            return False  # Row/Column contains it already

        row_start = row_index // 3  # Row where the 3x3 containing (row_index, col_index)
        col_start = col_index // 3  # Column where the 3x3 containing (row_index, col_index)

        # Get the 3x3 block the position `(row_index, col_index)` is part of
        block = self.puzzle[row_start * 3:(row_start + 1) * 3, col_start * 3:(col_start + 1) * 3]

        # If the value is in that block
        if value in block:
            return False  # Present in the block

        return True  # Value can be put in that location

    @staticmethod
    def __has_duplicates(series: np.ndarray):
        series = np.ravel(series)  # Flatten the array
        values, counts = np.unique(series, return_counts=True)  # Finding the unique elements and their counts
        # If more than one duplicate(0 can be duplicated because 0=> Empty grid) digits exist
        if np.sum(counts > 1) > 1:
            return True
        # If a duplicated digit exists in the series
        if values[counts > 1] and values[counts > 1].item() != 0:  # Duplicate exists and is not empty grid
            return True
        return False

    def is_solvable(self):
        """
        :return: A boolean stating whether the configuration of the puzzle is solvable or not
        """
        for i in range(9):
            row = self.puzzle[i, :]  # The i-th row
            column = self.puzzle[:, i]  # The i-th column

            # If the row or columns has any duplicates other than 0
            if self.__has_duplicates(row) or self.__has_duplicates(column):
                return False

        # Checking the 3x3 blocks for duplicates
        for i in range(3):
            for j in range(3):
                block = self.puzzle[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                if self.__has_duplicates(block):
                    return False
        return True
