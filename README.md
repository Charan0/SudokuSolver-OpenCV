# SudokuSolver-OpenCV
A realtime sudoku solver built using Python, Keras and OpenCV

# How it works?

## Broadly there are 4 steps involved in this

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

## The Source image

<div>
<p align="middle">
  <img src='Samples/Original.png' width=450 height=550">
</p>
<p align="middle">
The image shown, which I found on reddit (A meme called accidental swastika :joy:) contains a sudoku grid on a paper. This is the original 3 channeled RGB source image and this will be serving as an example, where we will look at how each step is carried out on this particular image.
</p>
</div>
                 
## Morphed image

<div>
<p align="middle">
  <img src='Samples/Morphed.png' width=450 height=550">
</p>
<p align="middle">
This is the image obtained after applying some morphological transformations using `morph_image()`. Specifically for this task I have first converted the image from color to black and white and then performed Gaussian Blur to smooth the image and then applied a threshold operation. The end result of these operations is as shown in the above sample.
</p>
</div>
                 
## Contours


<div>
<p align="middle">
  <img src='Samples/All-Contours.png' width=450 height=550">
</p>
<p align="middle">
We then find out all the contours in the image using OpenCV's `findContours()` method. The above image shows all the identified contours in the image. Next we need to extract the region of image that contains the largest contour area.
</p>
</div>
                 
## Largest contour and localizing the puzzle



<div>
<p align="left" width=30%>
  <img src='Samples/LargestContour.png' width=450 height=550">
</p>
<p align="right" width=30%>
  <img src='Samples/LocalizedSudoku(WarpPerspective).png'>
</p>
<p align="middle">
Hello there
</p>
</div>
