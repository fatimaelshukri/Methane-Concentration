Temporal and Spatial Pattern Analysis and Forecasting of Methane concentrations
# Temporal and Spatial Pattern Analysis and Forecasting of Methane
Table of contents 

## 1.  Data collection
This section is code to obtain TIF Files from Google Earth Engine (GEE). 

1.1 to obtain the tif imgaes through looping. 
In this program the code uses an API from GEE to open the data set and obtain an image with the given start and end data and with the given longitude and latitude.

## 2. Average images

Using the mean average formula, the average image is found by calculating the average Column-averaged dry air mixing ratio of methane, as parts-per-billion, corrected for surface albedo at that point over the range (2019-02-08 to 2024-06-30). There are 5 types of averages:

- Daily: This creates an average based on the day (e.g. Jan-1-2019, Jan-1-2020, Jan-1-2021 etc.)
- Monthly: This average is based on the pixel values for the complete month form each year
- monthly per year: this creates an average for each month (jan-dec) for its corresponding year thus outpting 12 images for each year between 2019-2024.  
 (72 images in total)
- yearly- a singular averaged image for each year, that is a result after averaging the monthly per year images. ( 2019 to 2024 = 6 yearly images ) 
- Annual (Overall): This average is computed using the entire years data (i.e. every image, therefore it is also called the overall average)

## 3. thresholding

The purpose of thresholding is to classify each pixel in an image based on its value, typically by converting it into a binary image where pixels above a certain value (threshold) are set to one color (e.g., white), and those below or equal to the threshold are set to another color (e.g., black). This helps in highlighting specific areas of interest in the image. 

### 4. sobel gradient 

The Sobel Gradient is a method of finding the gradient with respect to its adjacent and diagonally connected neighbours. There is a vertical and horizontal operator, and the magnitude of the gradient is calculated. it highlights areas where there are significant changes in pixel values

## 5. Linear Regression Prediction
Using Linear Regression, the Carbon monoxide density for each month is detected for the year 2024 july-december and the entire 12 months of 2025.
      5.1 Generate Regression Prediction Images
Individual pixels are predicted using pixels from the monthly per images (72 images,from 2019 to 2024) to train the regressison model to predict for the year 2024 and then using those 72 images and the newly formed monthly predicted images for 2024 (84 images) we will predict for 2025.
