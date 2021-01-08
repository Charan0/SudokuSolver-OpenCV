# SudokuSolver-OpenCV
A realtime sudoku solver built using Python, Keras and OpenCV

# How it works?

### Broadly there are 4 steps involved in this

* Detecting and Extracting the Sudoku Puzzle from the camera input.
* Identifying the digits in the localized Sudoku Puzzle.
* Solving the identified sudoku puzzle.
* Projecting the solution back on to the original input frame.

<strong>Note: For most of the image processing involved I have worked with grayscale images and also the extracted digits and the images on which the CNN is trained on are single channeled grayscale images.</strong>

## A) Detecting and extracting only the sudoku puzzle from the camera input

* We first capture the input from the camera and morph the input (This makes it a lot easier to apply any kind of image processing techniques)
* We then find the contours in the image using OpenCV's `findContour()`.
* The Sudoku puzzle as we know is an 9x9 Grid (A square, if not most prolly a rectangle) so it forms a closed contour and given ideal background conditions this contour should be the one with the largest area.
* We make use of this fact to extract and localize only the sudoku puzzle from the whole input.


<img src='Samples/Original.png' width=450 height=550>
<p text-align="center">
The sample image shown, which I found on reddit (A meme called accidental swastika :joy:) contains a sudoku grid on a paper. This is the original image and as an example we will look at how each step is carried out on this particular image.
</p>
