import cv2  # `OpenCV`for performing operation over images, capturing input from camera and other vision operations
import numpy as np  # `Numpy` for arrays and fast operations over those arrays
from typing import Union, List  # Used for type checking
from scipy.ndimage.measurements import center_of_mass  # Used to centralize a particular part in the image
from tensorflow.keras import models  # `Tensorflow` for loading NeuralNetwork model
from Solver import Solver

model = models.load_model('Models/img_model')  # Loading the pre-trained model


def morph_image(image: np.ndarray, perform_blur: bool = True, perform_threshold: bool = True, thresh_type: int = 0,
                dilate: bool = False, erode: bool = False, save_result: bool = False, c: int = 2) -> np.ndarray:
    """
    :param image: Input image to morph
    :param perform_blur: Boolean that specifies whether or not to perform blurring
    :param perform_threshold: Boolean that specifies whether or not to perform thresholding
    :param thresh_type: integer that specifies the type of thresholding
    :param dilate: Boolean that specifies whether or not to perform dilation
    :param erode: Boolean that specifies whether or not to perform erosion
    :param save_result: Boolean that specifies whether or not to show the output of the function
    :param c: integer used as a parameter in adaptive thresholding
    :return: Morphed image of the input
    """
    if image.ndim == 3:  # Converting to black and white
        dst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # If received a black and white image we make a copy of it for additional operations
        dst = image.copy()
    if perform_blur:  # Performing the blur operation
        dst = cv2.GaussianBlur(dst, (5, 5), 0)
    if perform_threshold:  # Performing binary-thresholding
        if thresh_type:
            dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, c)
        else:
            dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, c)
    # The structuring element that will be used for the dilation or erosion operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if dilate:
        dst = cv2.dilate(dst, kernel, iterations=1)
    if erode:
        dst = cv2.erode(dst, kernel, iterations=1)
    if save_result:
        cv2.imwrite('Image-Morphed.png', dst)
    return dst


def resize(src_image: np.ndarray, dest_width: int, dest_height: int = None, keep_ratio: bool = True):
    if dest_width <= 0 or dest_height <= 0:  # If encountered invalid values
        print('Invalid arguments provided')
        return src_image
    if not keep_ratio:  # If we do not want to preserve the aspect ratio
        resized_image = cv2.resize(src_image, (dest_height, dest_width))
        return resized_image
    else:
        height, width, _ = src_image.shape  # Original width and height
        aspect_ratio = width / height  # Calculating the aspect ratio
        calculated_height = int(dest_width / aspect_ratio)  # New height based on the aspect_ratio
        resized_image = cv2.resize(src_image, (dest_width, calculated_height))
        return resized_image


def get_largest_contour(image: np.ndarray, save_result: bool = False) -> np.ndarray:
    """
    :param image: Input image to find the largest contour
    :param save_result: Boolean to specify whether or not to show the output image
    :return: Returns the points in the largest contour found in the image
    """
    if image.ndim == 3:
        dst = morph_image(image.copy())
    else:
        dst = image.copy()
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlines = max(contours, key=cv2.contourArea)
    if save_result:
        image_with_contours = cv2.drawContours(image, [outlines], -1, (0, 255, 50), 3)
        cv2.imwrite('Image-Contoured.png', image_with_contours)
    return outlines


def find_corners(outline: np.ndarray, image: np.ndarray = None, save_result: bool = False) -> Union[np.ndarray, None]:
    """
    :param outline: A set of outline/contour points
    :param image: Input image
    :param save_result: Boolean that specifies whether or not to show the output
    :return: Finds the 4 corners of the polygon
    """
    if save_result and image is None:
        # print('To show results the image should not be none.')
        return None
    corners = np.zeros((4, 2), dtype='int32')
    outline = np.squeeze(outline)
    added_points = np.sum(outline, axis=-1)
    diff_points = np.diff(outline, axis=-1)
    corners[0] = outline[np.argmin(added_points)]
    corners[1] = outline[np.argmin(diff_points)]
    corners[2] = outline[np.argmax(added_points)]
    corners[3] = outline[np.argmax(diff_points)]
    if save_result:
        dst = image.copy()
        for i in range(4):
            cv2.circle(dst, tuple(corners[i].tolist()), 4, (0, 255, 0), -1)
        cv2.imwrite('Image-Corners.png', dst)
    return corners


def are_right_angled(corners: np.ndarray, tolerance=1e-4, rel_tol=1e-8):
    """
    :param corners: Set of points
    :param tolerance: The tolerance/limit used as atol to numpy allclose method
    :param rel_tol: The tolerance/limit used as rtol to numpy allclose method
    :return: Returns a boolean stating whether or not the angles are nearly 90 degrees wrt each other or not
    """
    # I ran a loop instead of duplicating the code, hence it needed to circle over
    modified_corners = np.vstack([corners, corners[0]])
    # A loop that finds every vector of two adjacent sides and the angles between them
    for i in range(1, modified_corners.shape[0] - 1):
        vec1 = (modified_corners[i - 1] - modified_corners[i]) / np.linalg.norm(
            modified_corners[i - 1] - modified_corners[i])
        vec2 = (modified_corners[i] - modified_corners[i + 1]) / np.linalg.norm(
            modified_corners[i] - modified_corners[i + 1])
        cos_theta = np.dot(vec1, vec2)
        nearly_equal = np.allclose(cos_theta, [0], atol=tolerance, rtol=rel_tol)
        if not nearly_equal:
            return False
    return True


def get_dimensions(corners: np.ndarray, tolerance=1e+5, rel_tol=1e-4):
    """
    Given a set of four corners of a rectangle(A polygon to be precise)
    calculates the height and width and returns them
    :param corners: Corner points of the rectangle(polygon)
    :param tolerance: A factor which determines how much can the lengths differ used in `numpy.allclose()`
    :param rel_tol: Another factor which determines how much can the lengths differ used in `numpy.allclose()`
    :return: Returns the dimensions and a boolean stating whether or not the lengths are nearly equal
    """
    width = max([
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[2] - corners[3]),
    ])
    height = max([
        np.linalg.norm(corners[0] - corners[3]),
        np.linalg.norm(corners[1] - corners[2]),
    ])

    nearly_equal = np.allclose([height], [width], atol=tolerance, rtol=rel_tol)

    return width, height, nearly_equal


def warp_image(image: np.ndarray, width: int, height: int, old_points: np.ndarray, new_points: np.ndarray,
               save_result: bool = False):
    """
    Performs standard image warping, used to extract and localize only the
    sudoku puzzle from the camera input
    :param image: The complete input from the camera
    :param width: The desired output width
    :param height: The desired output height
    :param old_points: The locations of the sudoku puzzle
    :param new_points: The new points to which the localized puzzle will be mapped to
    :param save_result: A boolean when set to True displays the warped image
    :return: Returns the warped image
    """
    if 'float' not in str(old_points.dtype):
        old_points = old_points.astype('float32')
    if 'float' not in str(new_points.dtype):
        new_points = new_points.astype('float32')

    matrix = cv2.getPerspectiveTransform(old_points, new_points)
    dst = cv2.warpPerspective(image, matrix, (width, height))
    if save_result:
        cv2.imwrite('Image-Warped.png', dst)
    return dst, matrix


def centralize(image):
    center_y, center_x = center_of_mass(image)
    rows, cols = image.shape
    x_shift = np.round(cols / 2.0 - center_x).astype(int)
    y_shift = np.round(rows / 2.0 - center_y).astype(int)

    matrix = np.array([[1, 0, x_shift], [0, 1, y_shift]], dtype='float32')
    dst = cv2.warpAffine(image, matrix, (cols, rows))
    return dst


def clear_border(src: np.ndarray, ratio: float = 0.25, thresh: float = 180):
    num_rows, num_cols = src.shape
    dst = src.copy()
    while dst.size and (np.sum(dst[0] >= thresh) / num_cols) > ratio:
        dst = dst[1:]
    while dst.size and (np.sum(dst[-1] >= thresh) / num_cols) > ratio:
        dst = dst[:-1]
    while dst.size and (np.sum(dst[:, 0] >= thresh) / num_rows) > ratio:
        dst = dst[:, 1:]
    while dst.size and (np.sum(dst[:, -1] >= thresh) / num_rows) > ratio:
        dst = dst[:, :-1]
    return np.pad(dst, 2)


def extract_digit(src: np.ndarray):
    src = cv2.medianBlur(src, 5)  # MedianBlur to remove salt-pepper like noise
    src = clear_border(src, ratio=0.4, thresh=180)  # We first remove the white borders on the edges
    src = src.astype('uint8')  # The input should be of type uint8 for connectedComponents to work
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8)
    # stats contains statistics about the connected components
    # stats is an np array of shape (-x-, 5) [x0, y0, width, height, area]
    # We select the component that has the largest area
    # print(f'Shape of stats : {stats.shape}')
    areas = stats[1:, 4]  # We want to leave out the background so we start from the second element
    if not areas.size:
        return src
    # print(f'Size of areas components : {areas.size}')
    max_label = np.argmax(areas) + 1  # Since we ignored the initial component we append `1` to get the correct index
    dst = np.zeros_like(labels)  # A complete black image
    dst[labels == max_label] = 255  # Filling in white pixels i.e filling the digit(only) in the image

    return np.pad(dst, 2)


def preprocess(digit_image: np.ndarray, size: int = 40):
    src = digit_image.astype('uint8')  # ConnectedComponents expects the image to be of unsigned integer
    extracted = extract_digit(src).astype('uint8')  # Ignoring noise and extracting only the digit from the box

    dst = centralize(extracted)  # Centralizing the digit in the box
    _, dst = cv2.threshold(dst, 160, 255, cv2.THRESH_BINARY_INV)  # Making it a binary image
    dst = cv2.resize(dst, (size, size), interpolation=cv2.INTER_AREA)
    dst = dst / 255.  # Image pixel values now lie in [0, 1.]

    return dst


def percent_filled(image: np.ndarray, buffer: int, thresh: float):
    assert len(np.unique(image)) <= 2  # Binary image is expected
    dst = image[buffer:, buffer:]  # Removing some of the rows, columns i.e the buffer
    percent = np.sum(dst == thresh) / dst.size  # The amount of the image that is filled
    return percent


def make_prediction(image: np.ndarray):
    assert image.ndim != 3  # We send in only a b/w image
    if image.ndim < 4:  # ConvNet expects (batch_dimensions, Dimension1, Dimension2, num_channels)
        image = np.expand_dims(image, axis=[0, -1])  # Expand dims to accommodate batch and channel dims
    assert image.ndim == 4
    prediction = model.predict(image)  # Softmax probabilities
    return np.argmax(prediction) + 1  # Label of 0 implies 1. Hence we need to increment it by 1


def locate_and_predict(warped_image: np.ndarray):
    if warped_image.ndim == 3:
        dst = morph_image(warped_image)  # Morphing the image if the input received is 3-Channeled(BGR most probably)
    else:
        dst = warped_image.copy()  # We do not want modify the original image in anyway

    count = 0
    locations = []  # This will contain the locations of each of the box in the puzzle
    height, width = dst.shape  # The dimensions of the localized sudoku puzzle
    box_width, box_height = width // 9, height // 9  # The dimensions of each box in the sudoku
    offset_w, offset_h = box_width // 12, box_height // 12  # Offset used to eliminate noise
    grid = np.zeros((9, 9))  # This will store the detected sudoku puzzle
    for i in range(9):
        for j in range(9):
            x1, y1 = box_height * i + offset_h, box_width * j + offset_w  # Top left corner of the box
            x2, y2 = box_height * (i + 1) - offset_h, box_width * (j + 1) - offset_w  # Bottom right corner of the box
            locations.append([(x1, y1), (x2, y2)])
            roi = dst[x1:x2, y1:y2]  # Extracting the box
            assert len(np.unique(roi)) <= 2
            assert (np.max(roi) == 255 or np.max(roi) == 0) and (np.min(roi) == 0)
            if percent_filled(roi, 5, 255) < 0.08:  # Encountered a blank grid, leave that
                continue
            else:  # Found a grid with a digit, make predictions
                # Preprocessing and then predicting the digit in the image
                preprocessed = preprocess(roi, 40)  # Preprocessing the image => Resizing and Centralizing
                # Making the prediction
                grid[i, j] = make_prediction(preprocessed)  # BottleNeck for performance (Too Slow?)
                count += 1
    # print(f'Identified a total of {count} digits in the sudoku grid')
    return locations, grid


def write_on_image(image: np.ndarray, locations: List, strings: np.ndarray, mask: np.ndarray):
    """
    :param image: The image on which the text is to be written onto
    :param locations: The locations of the boxes (The top-left and the bottom-right corner)
    :param strings: The text to be written onto the image
    :param mask: A boolean array indicating whether or not a grid is empty
    :return: An image of a filled sudoku grid
    """
    index = 0
    dst = image.copy()  # Making a copy of the image to avoid modifying the original

    offset_x = 5
    offset_y = 5

    if not np.issubdtype(mask.dtype, bool):
        mask = mask.astype('bool')

    mask = mask.flatten()  # Boolean nd-array
    strings = strings.astype('int').flatten()

    font = cv2.FONT_HERSHEY_COMPLEX  # The font-face being used
    height, width, _ = image.shape
    factor = max(height / width, width / height)  # Factor to be multiplied by the base size of the font

    font_scale = 1.25 * factor

    for (_, x), (y, _) in locations:  # Coordinates of the bottom left corner of the box
        if not mask[index]:  # If the box in empty in the 'Unsolved' Sudoku
            cv2.putText(dst, str(strings[index]), (x + offset_x, y - offset_y), font, font_scale, (150, 0, 0), 2)
        index += 1
    return dst


def warp_and_stitch(original_image: np.ndarray, warped_image: np.ndarray, matrix: np.ndarray):
    """
    Performs OpenCV's warpPerspective and stitches the `solved` localized grid into the main image.

    :param original_image: The main input image
    :param warped_image: The localized sudoku grid
    :param matrix: Original matrix used which is used for localizing the sudoku grid from the input image
    :return: Returns an image(numpy array) which is the stitched result of the solved sudoku grid
    with the input image
    """
    height, width, _ = original_image.shape
    # An image of dimensions the same as the original image
    # `inverse_warped` has the localized sudoku grid and every other part of the image is just black
    inverse_warped = cv2.warpPerspective(warped_image.copy(), matrix, (width, height), flags=cv2.WARP_INVERSE_MAP)
    # The main logic behind the stitching
    stitched = np.where(
        np.sum(inverse_warped, axis=-1, keepdims=True) == 0,  # Wherever the pixel value is black
        # Replace the black pixel values with the values from the original image
        # aka substituting the background
        original_image,
        # If not black pixel then leave it unchanged
        inverse_warped
    )
    return stitched, inverse_warped