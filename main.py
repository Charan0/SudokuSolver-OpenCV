import argparse
from Sudoku_CV import *
from Solver import Solver
from pathlib import Path

parser = argparse.ArgumentParser(
    description='This is a real-time sudoku solver. You can either send an image of a sudoku '
                'puzzle or even better holding a sudoku paper with your webcam turned on.')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--camera', action='store_true', help='Use the webcam as the input')
group.add_argument('--file', action='store_true', help='Take input from file')

parser.add_argument('-i', '--input', type=str, required=False, help='Location of the sudoku image')
parser.add_argument('--save', required=False, action='store_true', help='Save the output')
parser.add_argument('-vis', '--visualize', required=False, action='store_true',
                    help='Save all the intermediate images')

args = parser.parse_args()


def parse_and_solve(image: np.ndarray, is_video=True, save_result: bool = False):
    # Gets the points on the largest `connected-contour` in the image
    outline = get_largest_contour(image.copy(), save_result)  # The outline of the sudoku grid
    corners = find_corners(outline, image.copy(), save_result)  # Find the corners of the contour
    # Get the dimensions of the grid and confirm the largest contour is from the sudoku grid
    width, height, nearly_equal = get_dimensions(corners, 1e-1, 1e-1)

    # Since live feed from a camera may contain other objects in the background
    # and we don't want to pick any element from the bg even if the largest contour
    # is from something other than the puzzle we don't want to pick it up
    if is_video:
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
    warped_image, matrix = warp_image(image.copy(), int(width), int(height), corners, new, save_result)
    # cv2.imshow('Warped', warped_image)  # Localized image of the grid

    locations, grid = locate_and_predict(warped_image)
    solver = Solver(grid.copy())  # Initialising the sudoku solver
    solved_grid = solver.solve()  # Solving the puzzle

    # The predictions might be wrong or the puzzle itself might be wrong
    # So as to avoid a crash we perform some checks
    if solved_grid is None:  # Puzzle is not solvable
        return image

    mask = grid.astype('bool')  # The original grid `unsolved-sudoku` will be used as the mask
    # The solved localized sudoku image
    solved_warp = write_on_image(warped_image, locations, solved_grid, mask)
    # The localized puzzle stitched back onto the input
    solved_sudoku_image, inverse_warp = warp_and_stitch(image, solved_warp, matrix)

    return solved_sudoku_image


def solve_sudoku(from_video: bool, input_image: np.ndarray = None, save_result: bool = False):

    def set_resolution(width, height):
        capture.set(3, width)
        capture.set(4, height)

    if from_video:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        set_resolution(1280, 720)
        while True:  # Capturing the input from the camera
            ret_val, frame = capture.read()

            result = parse_and_solve(frame)
            cv2.imshow('Frame', result)

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        capture.release()
        return
    else:
        solved_image = parse_and_solve(input_image, is_video=False, save_result=save_result)
        return solved_image


if __name__ == '__main__':
    if args.camera:  # If input from the webcam
        solve_sudoku(from_video=True)
    elif args.file:  # Send an image as the input
        if args.input is None:
            parser.error('--file requires --input')
        else:
            image_loc = Path(args.input)  # Getting the path of the input image
            if not image_loc.exists():  # If the file does not exist
                print('The image path does not exist')
                exit(-1)
            src_image = cv2.imread(str(image_loc), cv2.IMREAD_UNCHANGED)  # Loading in the image
            src_width, src_height = src_image.shape[:-1]  # The original dimensions of the image
            if src_width >= 1000 or src_height >= 1000:  # If the image it too large resize it
                print('Image dimensions too large to work with')
                dst = resize(src_image.copy(), 650, True)  # Resize with output width set to 650
                print(f'Image resized to {dst.shape}')
                dst = cv2.bilateralFilter(dst, 15, 95, 95)  # Bilateral filtering to remove noise from the resized image
            else:
                dst = src_image.copy()
            if args.visualize:  # If visualize is set to true
                # Store all the intermediate images
                solved_sudoku = solve_sudoku(False, dst, save_result=True)
            else:  # If visualize is not set
                solved_sudoku = solve_sudoku(False, dst)
            if args.save:  # If `save` is set to True
                cv2.imwrite('Output_image.png', solved_sudoku)  # Saving the output image
            cv2.imshow('Solved', solved_sudoku)  # Displaying the output image
            cv2.waitKey(0)
            cv2.destroyAllWindows()
