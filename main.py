from Sudoku_CV import *
from Solver import Solver

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def set_resolution(width, height):
    capture.set(3, width)
    capture.set(4, height)


def parse_and_solve(image: np.ndarray):
    # Gets the points on the largest `connected-contour` in the image
    outline = get_largest_contour(image.copy())  # The outline of the sudoku grid
    corners = find_corners(outline)  # Find the corners of the contour
    # Get the dimensions of the grid and confirm the largest contour is from the sudoku grid
    width, height, nearly_equal = get_dimensions(corners, 1e-1, 1e-1)
    if not nearly_equal:  # If the sides of the contour are not nearly equal
        # print(f'Width : {width}, Height : {height}')
        # print('Side lengths not nearly equal')
        return image

    if not are_right_angled(corners, .9e-1):  # If the angles are not nearly 90 deg(not the grid we are looking for)
        # print('The angles are not 90 degree to each other')
        return image

    # Localizing only the sudoku grid
    new = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    # Warping the image
    warped_image, matrix = warp_image(image.copy(), int(width), int(height), corners, new)
    # cv2.imshow('Warped', warped_image)  # Localized image of the grid

    locations, dst, grid = locate_and_predict(warped_image)
    solver = Solver(grid.copy())  # Initialising the sudoku solver
    solved_grid = solver.solve()  # Solving the puzzle

    mask = grid.astype('bool')  # The original grid `unsolved-sudoku` will be used as the mask

    # init_grid = np.array([
    #     [4, 0, 1, 2, 9, 0, 0, 7, 5],
    #     [2, 0, 0, 3, 0, 0, 8, 0, 0],
    #     [0, 7, 0, 0, 8, 0, 0, 0, 6],
    #     [0, 0, 0, 1, 0, 3, 0, 6, 2],
    #     [1, 0, 5, 0, 0, 0, 4, 0, 3],
    #     [7, 3, 0, 6, 0, 8, 0, 0, 0],
    #     [6, 0, 0, 0, 2, 0, 0, 3, 0],
    #     [0, 0, 7, 0, 0, 1, 0, 0, 4],
    #     [8, 9, 0, 0, 6, 5, 1, 0, 7]
    # ])
    #
    # solved_grid = np.array(
    #     [[4, 8, 1, 2, 9, 6, 3, 7, 5],
    #      [2, 5, 6, 3, 1, 7, 8, 4, 9],
    #      [3, 7, 9, 5, 8, 4, 2, 1, 6],
    #      [9, 4, 8, 1, 5, 3, 7, 6, 2],
    #      [1, 6, 5, 9, 7, 2, 4, 8, 3],
    #      [7, 3, 2, 6, 4, 8, 9, 5, 1],
    #      [6, 1, 4, 7, 2, 9, 5, 3, 8],
    #      [5, 2, 7, 8, 3, 1, 6, 9, 4],
    #      [8, 9, 3, 4, 6, 5, 1, 2, 7]]
    # )
    # cv2.imshow('WarpedImage', warped_image)
    solved_warp = write_on_image(warped_image, locations, solved_grid, mask)
    # cv2.imshow('WrittenImage', written_image)
    solved_sudoku_image, inverse_warp = warp_and_stitch(image, solved_warp, matrix)
    # cv2.imshow('InverseWarp', inverse_warp)

    return solved_sudoku_image


set_resolution(1280, 720)
while True:
    ret_val, frame = capture.read()

    result = parse_and_solve(frame)
    cv2.imshow('Image', result)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()
