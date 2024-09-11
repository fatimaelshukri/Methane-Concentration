## Temporal and Spatial Pattern Analysis and Forecasting of Methane concentrations

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

## 6. Calculate RMSE
The Root Mean Square Error (RMSE) for each month is calculated against the real images of 2024. 

## 7. extra codes : 
   7.1  to generate graphs to visualize trends monthly 
   7.2  to generate graphs to visualize trends yearly     
   7.3 to generate missing dates from the GEE, the missing dates where the erath engine could not capture images for that day. (unavailability of datset for that day from the date range specified)

1.  **Data Collection**.
```console
 
import ee
import datetime
import os
import geemap  # Ensure you have geemap installed: pip install geemap

# Initialize Earth Engine
ee.Initialize()

# Define a region of interest (ROI)
roi = ee.Geometry.Polygon(
    [[[50.717442, 24.455254],
      [51.693314, 24.455254],
      [50.717442, 26.226590],
      [51.693314, 26.226590]]])

# Define the time range
start_date = '2019-02-01'
end_date = '2024-07-01'

# Convert start and end dates to datetime objects
start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

# Specify the output root folder
output_root_folder = r"C:\Users\Wajeeha\Desktop\2024 aug"

# Iterate over each day in the specified period
current_date = start_datetime
while current_date < end_datetime:
    # Convert the current date to string
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    # Create a folder for the current month and day
    output_folder = os.path.join(output_root_folder, current_date.strftime('%m-%d'))
    os.makedirs(output_folder, exist_ok=True)
    
    # Sentinel-5P CH4 dataset (Offline)
    collection = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
                  .filterBounds(roi)
                  .filterDate(ee.Date(current_date_str), ee.Date(current_date_str).advance(1, 'day'))
                  .select('CH4_column_volume_mixing_ratio_dry_air_bias_corrected'))

    # Get the mean image
    image = collection.median()

    # Get the minimum and maximum values for the specified region
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=1000  # Adjust the scale based on your data
    )

    # Debugging: Print the contents of stats
    print(f"Stats for {current_date_str}: {stats.getInfo()}")

    # Extract min and max values from the statistics for the 'CH4_column_volume_mixing_ratio_dry_air_bias_corrected' band
    min_value = stats.get('CH4_column_volume_mixing_ratio_dry_air_bias_corrected_min')
    max_value = stats.get('CH4_column_volume_mixing_ratio_dry_air_bias_corrected_max')

    # Check if min and max values are not None before using them
    if min_value is not None and max_value is not None:
        try:
            min_value_scaled = min_value.getInfo()
            max_value_scaled = max_value.getInfo()
            print(f"Image {image.id()} - Minimum CH4 density: {min_value_scaled}, Maximum CH4 density: {max_value_scaled}")

            # Define the output path
            output_path = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}.tif")

            # Download the image and save it to the local folder
            geemap.ee_export_image(image, filename=output_path, region=roi, scale=1000)
        except Exception as e:
            print(f"Error processing {current_date_str}: {e}")
    else:
        print(f"No data available for {current_date_str}")
    
    # Move to the next day
    current_date += datetime.timedelta(days=1)
```
   
## 2. Average images

 **2.1 Daily average**

```console
import os
import re
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from collections import defaultdict

# ROI coordinates for plotting
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

def find_tiff_files(directory):
    """Recursively find all TIFF files in the given directory and its subdirectories."""
    tiff_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                tiff_files.append(os.path.join(root, file))
    return tiff_files

def extract_date_from_folderpath(folderpath):
    """Extract the month-day from the folder path, assuming format 'MM-DD'."""
    match = re.search(r"(\d{2}-\d{2})", folderpath)
    if match:
        return match.group(1)
    else:
        print(f"Folder path {folderpath} does not match the expected date format.")
        return None

def group_images_by_date(tiff_files):
    """Group images by date across different years."""
    grouped_files = defaultdict(list)
    for tiff_file in tiff_files:
        folder_path = os.path.dirname(tiff_file)
        date_key = extract_date_from_folderpath(folder_path)
        if date_key:
            grouped_files[date_key].append(tiff_file)
    return grouped_files

def calculate_daily_average(image_files):
    """Calculate the average image from a list of image files."""
    if not image_files:
        return None
    
    first_image = tf.imread(image_files[0]).astype(np.float64)
    
    pixel_sum = np.zeros_like(first_image)
    count = np.zeros_like(first_image)

    for image_file in image_files:
        image = tf.imread(image_file).astype(np.float64)
        pixel_sum += np.nan_to_num(image)
        count += ~np.isnan(image)

    return np.divide(pixel_sum, count, out=np.zeros_like(pixel_sum), where=(count != 0))

def plot_with_roi(image, output_file, date_label):
    """Plot the image with ROI outline and save as PNG with a colorbar and title."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    im = ax.imshow(image, cmap='viridis', extent=[50.717442, 51.693314, 24.455254, 26.226590])
    
    # Add coastlines, borders, and land features
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    
    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=5)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='black', linewidth=5)

    # Add colorbar and title
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='CH4 column volume mixing ratio dry air (as parts-per-billion)')
    plt.title(f'Daily Column-averaged dry air mixing ratio of methane - {date_label}')

    # Save the figure
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def process_images(main_directory, output_directory):
    """Process all images, calculate daily averages, and save them."""
    tiff_files = find_tiff_files(main_directory)
    grouped_files = group_images_by_date(tiff_files)

    for date_key, image_files in sorted(grouped_files.items()):
        average_image = calculate_daily_average(image_files)
        if average_image is None:
            continue
        
        output_subdir = os.path.join(output_directory, date_key)
        os.makedirs(output_subdir, exist_ok=True)

        # Save TIFF image without colormap or ROI
        tiff_output_file = os.path.join(output_subdir, f'Average_Image_{date_key}.tif')
        tf.imwrite(tiff_output_file, average_image)

        # Extract date label from image path
        date_label = os.path.basename(os.path.dirname(image_files[0]))

        # Save PNG image with Viridis colormap and ROI outline
        png_output_file = os.path.join(output_subdir, f'Average_Image_{date_key}_ROI.png')
        plot_with_roi(average_image, png_output_file, date_label)

# Specify the directories
main_directory = r"C:\Users\Wajeeha\Desktop\methane"
output_directory = r"C:\Users\Wajeeha\Desktop\AverageDailymethanebar"

# Process the images
process_images(main_directory, output_directory)
```
**2.2 Monthly average**
```console
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ROI coordinates for plotting
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

# Function to load and average images for a specific month
def average_images_for_month(directory, month_prefix):
    images = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') and file.startswith(f"Average_Image_{month_prefix}"):
                tiff_path = os.path.join(root, file)
                image = tiff.imread(tiff_path)
                images.append(image)

    if len(images) == 0:
        raise ValueError(f"No TIFF files found for month {month_prefix}")

    images = np.array(images)
    average_image = np.nanmean(images, axis=0)

    if average_image.ndim != 2:
        raise ValueError(f"Invalid shape for average image for month {month_prefix}")

    return average_image

# Function to plot and save the PNG image with colormapping and ROI outlined
def plot_with_roi(image, output_file, month_label):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    im = ax.imshow(image, cmap='viridis', extent=[50.717442, 51.693314, 24.455254, 26.226590])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=2)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='black', linewidth=2)

    # Add colorbar and title
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='CH4 column volume mixing ratio dry air (as parts-per-billion)')
    plt.title(f'Monthly Column-averaged dry air mixing ratio of methane - {month_label}')

    # Save the PNG image
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

# Directories
daily_avg_directory = r"C:\Users\Wajeeha\Desktop\AverageDailymethanebar"
output_directory = r"C:\Users\Wajeeha\Desktop\Averagemonthlymethane"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process and save monthly averages
for month_number in range(1, 13):
    month_prefix = f"{month_number:02d}"
    try:
        average_image = average_images_for_month(daily_avg_directory, month_prefix)
        
        # Save TIFF image
        tiff_output_file = os.path.join(output_directory, f'Monthly_Average_{month_prefix}.tif')
        tiff.imwrite(tiff_output_file, average_image.astype(np.float32))
        
        # Save PNG image with colormapping and ROI outlined
        png_output_file = os.path.join(output_directory, f'Monthly_Average_{month_prefix}.png')
        plot_with_roi(average_image, png_output_file, month_prefix)
        
        print(f"Processed and saved: {tiff_output_file} and {png_output_file}")
        
    except Exception as e:
        print(f"Error processing month {month_prefix}: {e}")


```
**2.3 overall yearly average**

```console
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the directory containing the monthly average images
monthly_images_directory = r"C:\Users\Wajeeha\Desktop\Averagemonthlymethane"

# Define the output folder for saving the results
output_folder = r"C:\Users\Wajeeha\Desktop\overallyearly"

# Define the ROI coordinates
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

def load_monthly_average_images(directory):
    """Load all monthly average TIFF images."""
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    images = [tiff.imread(file_path) for file_path in file_paths]
    return images

def generate_overall_average_image(monthly_average_images):
    """Generate the overall average image by averaging all monthly average images."""
    overall_average_image = np.nanmean(np.array(monthly_average_images), axis=0)
    return overall_average_image

def plot_image_with_roi(image, roi_coords, output_file):
    """Plot the image with ROI outline and save it."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [roi_coords[0][0], roi_coords[1][0], roi_coords[0][1], roi_coords[2][1]]
    ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=2)

    plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='CH4 column volume mixing ratio dry air (as parts-per-billion)')
    plt.title('Overall Yearly Column-averaged dry air mixing ratio of methane', fontweight='bold')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.show()

if __name__ == "__main__":
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load all monthly average images
    monthly_average_images = load_monthly_average_images(monthly_images_directory)
    
    # Generate the overall average image
    overall_average_image = generate_overall_average_image(monthly_average_images)
    
    # Define file names for TIFF and PNG images
    tiff_file_name = "overall_average_image.tif"
    png_file_name = "overall_average_with_ROI.png"
     
    # Save the overall average image as a TIFF file
    output_tiff_file = os.path.join(output_folder, tiff_file_name)
    tiff.imwrite(output_tiff_file, overall_average_image.astype(np.float32))
    print(f"Saved TIFF: {output_tiff_file}")
    
    # Save the overall average image with ROI outline as a PNG file
    output_png_file = os.path.join(output_folder, png_file_name)
    plot_image_with_roi(overall_average_image, roi_coords, output_png_file)
    print(f"Saved PNG: {output_png_file}")

```
**2.4 monthly per year images (72 images- 12 for each year from 2019-2024)**

```console
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ROI coordinates for plotting
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

def load_daily_images(directory):
    """Load all daily TIFF images from the specified directory and organize by year, month, and day."""
    images_by_month_year = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                file_path = os.path.join(root, file)
                date_part = os.path.basename(file).split('.')[0]  # Extract date part from filename
                year, month, day = date_part.split('-')
                year_month = f"{year}-{month}"
                
                if year_month not in images_by_month_year:
                    images_by_month_year[year_month] = []
                
                image = tf.imread(file_path)
                images_by_month_year[year_month].append(image)
    
    return images_by_month_year

def calculate_monthly_averages(images_by_month_year):
    """Calculate the monthly average images for each year."""
    monthly_averages = {}
    
    for year_month, images in images_by_month_year.items():
        if images:  # Check if there are images for the month
            # Stack images and calculate mean across the stack
            stacked_images = np.stack(images, axis=0)
            average_image = np.mean(stacked_images, axis=0)
            monthly_averages[year_month] = average_image
    
    return monthly_averages

def save_images(monthly_averages, output_directory):
    """Save the averaged images as TIFF and PNG with ROI outline."""
    for year_month, image in sorted(monthly_averages.items()):
        year, month = year_month.split('-')
        year_folder = os.path.join(output_directory, year)
        os.makedirs(year_folder, exist_ok=True)
        
        # Save the TIFF image
        tiff_filename = os.path.join(year_folder, f"{year_month}.tif")
        tf.imwrite(tiff_filename, image)
        
        # Save the PNG image with colormap and ROI outline
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [50.717442, 51.693314, 24.455254, 26.226590]
        im = ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')

        # Plot ROI polygon as a closed loop
        ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=2)
        ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='black', linewidth=2)

        # Add colorbar and title
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('CH4 column volume mixing ratio dry air (as parts-per-billion)')
        plt.title(f'Column-averaged dry air mixing ratio of methane for {year_month}', fontweight='bold')

        # Save the PNG image
        png_filename = os.path.join(year_folder, f"{year_month}_ROI.png")
        plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

# Main directory containing the daily TIFF images
main_directory = r"C:\Users\Wajeeha\Desktop\methane"
output_directory = r"C:\Users\Wajeeha\Desktop\NP"

# Load daily TIFF images and organize by year and month
images_by_month_year = load_daily_images(main_directory)

# Calculate monthly averages
monthly_averages = calculate_monthly_averages(images_by_month_year)

# Save the monthly averaged images as TIFF and PNG
save_images(monthly_averages, output_directory)


```
**2.5 yearly per year from 2019-2024 (6 images for each year)**

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_monthly_averages(base_directory):
    """Load monthly average TIFF images for each year."""
    monthly_averages_by_year = {}

    for year in range(2019, 2025):
        year_folder = os.path.join(base_directory, str(year))
        if os.path.isdir(year_folder):
            monthly_images = []

            for month_file in sorted(os.listdir(year_folder)):
                if month_file.endswith('.tif') or month_file.endswith('.tiff'):
                    image_path = os.path.join(year_folder, month_file)
                    image = tf.imread(image_path)
                    monthly_images.append(image)

            if monthly_images:
                # Calculate the yearly average image from monthly averages
                yearly_average_image = np.mean(np.stack(monthly_images), axis=0)
                monthly_averages_by_year[year] = yearly_average_image

    return monthly_averages_by_year

def save_yearly_averages(yearly_images, output_directory):
    """Save the yearly average images as TIFF and PNG with ROI outline."""
    roi_coords = np.array([
        [50.717442, 24.455254],
        [51.693314, 24.455254],
        [51.693314, 26.226590],
        [50.717442, 26.226590],
        [50.717442, 24.455254]
    ])

    for year, image in sorted(yearly_images.items()):
        # Save the TIFF image
        tiff_filename = os.path.join(output_directory, f"{year}_yearly_average.tif")
        tf.imwrite(tiff_filename, image.astype(np.float32))
        print(f"Saved TIFF: {tiff_filename}")

        # Save the PNG image with colormap and ROI outline
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [50.717442, 51.693314, 24.455254, 26.226590]
        im = ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')

        # Plot ROI polygon as a closed loop
        ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
        ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

        # Add colorbar and title
        cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(('CH4 column volume mixing ratio dry air (as parts-per-billion)'))
        plt.title(f'Column-averaged dry air mixing ratio of methane for {year}', fontweight='bold')

        # Save the PNG image
        png_filename = os.path.join(output_directory, f"{year}_yearly_average_with_ROI.png")
        plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        print(f"Saved PNG: {png_filename}")

# Main directory containing the monthly average TIFF images for each year
base_directory = r"C:\Users\Wajeeha\Desktop\NP"

# Directory to save the output TIFF and PNG images
output_directory = r"C:\Users\Wajeeha\Desktop\NP\yeraly averages"

# Load monthly averages and calculate yearly averages
yearly_images = load_monthly_averages(base_directory)

# Save the yearly average images as TIFF and PNG with ROI outline
save_yearly_averages(yearly_images, output_directory)
```

#### 3. Sobel gradient

**3.2 sobel gradient for 12 average monthly images**
```console
import os
import numpy as np
import cv2
import tifffile as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def calculate_sobel_gradient(image):
    """Calculate Sobel gradient magnitude of the image."""
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

def plot_gradient_image(gradient_image, date_key, output_dir, roi_coords):
    """Plot and save the Sobel gradient image."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [50.717442, 51.693314, 24.455254, 26.226590]
    im = ax.imshow(gradient_image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('CH4 column volume mixing ratio dry air (as parts-per-billion)', fontweight='bold')
    plt.title(f'Sobel Gradient for {date_key}', loc='center', fontweight='bold')
    plt.axis('off')
    
    output_file = os.path.join(output_dir, f'Sobel_Gradient_{date_key}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def process_images(main_directory, roi_coords):
    """Process images to calculate and save Sobel gradients."""
    for year in range(2019, 2025):  # From 2019 to 2024
        year_folder = os.path.join(main_directory, str(year))
        if not os.path.exists(year_folder):
            continue
        
        output_dir = os.path.join(year_folder, 'sobel_gradients')
        os.makedirs(output_dir, exist_ok=True)
        
        for month_file in sorted(os.listdir(year_folder)):
            if month_file.endswith('.tif') or month_file.endswith('.tiff'):
                image_path = os.path.join(year_folder, month_file)
                image = tf.imread(image_path)
                
                if image is None:
                    print("Error: Unable to read image from:", image_path)
                else:
                    gradient_image = calculate_sobel_gradient(image)
                    date_key = month_file.split('.')[0]  # Extract date part from filename
                    plot_gradient_image(gradient_image, date_key, output_dir, roi_coords)

# Define the coordinates for the ROI (Region of Interest)
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

# Main directory containing the yearly folders with monthly average TIFF images
main_directory = r"C:\Users\Wajeeha\Desktop\NP"

# Process all images in the main directory
process_images(main_directory, roi_coords)
```

 **3.3 sobel gradient for 6 yearly images (2019-2024)**

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import cv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def calculate_sobel_gradient(image):
    # Calculate the gradient using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return gradient_magnitude

def save_sobel_gradient_image(gradient_image, output_path, roi_coords):
    # Plot the Sobel gradient image with ROI outline
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [50.717442, 51.693314, 24.455254, 26.226590]
    im = ax.imshow(gradient_image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('CH4 column volume mixing ratio dry air (as parts-per-billion)', fontweight='bold')
    plt.title('Sobel Gradient Image', fontweight='bold')

    # Save the plot as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_yearly_images(base_directory, roi_coords):
    # Iterate over all yearly average TIFF files in the base directory
    for file_name in os.listdir(base_directory):
        if file_name.endswith('.tif'):
            year = file_name.split('_')[0]
            year_dir = os.path.join(base_directory, year)
            if not os.path.exists(year_dir):
                os.makedirs(year_dir)

            image_path = os.path.join(base_directory, file_name)
            output_dir = os.path.join(base_directory, year)

            # Read the yearly average image
            image = tf.imread(image_path)
            if image is None:
                print(f"Error: Unable to read image from: {image_path}")
                continue

            # Calculate Sobel gradient
            gradient_image = calculate_sobel_gradient(image)

            # Save Sobel gradient image
            sobel_output_path = os.path.join(output_dir, f'{year}_sobel_gradient_with_ROI.png')
            save_sobel_gradient_image(gradient_image, sobel_output_path, roi_coords)

# Define the coordinates for the ROI (Region of Interest)
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

# Base directory containing yearly average TIFF images
base_directory = r"C:\Users\Wajeeha\Desktop\NP\yeraly averages"

# Process and save Sobel gradient images with ROI outline
process_yearly_images(base_directory, roi_coords)
```
## 4. Thresholding for monthly images 
```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

def process_and_save_thresholded_image(image_path, threshold, save_folder):
    # Load the TIFF image
    image_data = tf.imread(image_path)
    
    # Create subplots for the original and thresholded images
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Original image plot
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Initialize an empty binary image (3D array for RGB representation)
    binary_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
    white_pixels = 0
    nan_pixels = 0

    # Iterate through each pixel and apply the threshold condition
    for row in range(image_data.shape[0]):
        for col in range(image_data.shape[1]):
            value = image_data[row, col]

            # Check if the value is NaN or an array containing NaN
            if np.isnan(value).any():
                # Set NaN values to red (255, 0, 0)
                binary_image[row, col] = [255, 0, 0]
                nan_pixels += 1  # Increment NaN pixel count
            elif np.isscalar(value) and value > threshold:
                # If it's a scalar value above the threshold, set it to white (255, 255, 255)
                binary_image[row, col] = [255, 255, 255]
                white_pixels += 1  # Increment white pixel count
            elif np.isscalar(value) and value <= threshold:
                # If it's a scalar value below or equal to the threshold, set it to black (0, 0, 0)
                binary_image[row, col] = [0, 0, 0]
            else:
                # If the value is an array and doesn't meet the threshold, treat it as below the threshold
                binary_image[row, col] = [0, 0, 0]

    # Calculate total valid pixels (excluding NaN)
    total_valid_pixels = image_data.size - nan_pixels

    # Calculate white pixel ratio (white pixels / total valid pixels)
    white_pixel_ratio = (white_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

    # Plot the thresholded image
    axs[1].imshow(binary_image)
    axs[1].set_title(f'Threshold = {threshold}\nWhite Pixel Ratio: {white_pixel_ratio:.2f}%')
    axs[1].axis('off')

    # Generate the file name and save the figure
    file_name = os.path.basename(image_path).replace('.tif', '_thresholded.png')
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)
    plt.close()

def process_images_in_directory(main_directory, threshold):
    for year_folder in os.listdir(main_directory):
        year_path = os.path.join(main_directory, year_folder)
        if os.path.isdir(year_path):
            # Create a folder to save the thresholded images for this year
            save_folder = os.path.join(year_path, f"thresholded_{year_folder}")
            os.makedirs(save_folder, exist_ok=True)

            for month_file in os.listdir(year_path):
                if month_file.endswith('.tif'):
                    image_path = os.path.join(year_path, month_file)
                    process_and_save_thresholded_image(image_path, threshold, save_folder)

# Define the main directory containing the subfolders
main_directory = r"C:\Users\Wajeeha\Desktop\NP"

# Define the methane concentration threshold value
threshold = 1_000_000

# Process and save thresholded images for all years in the directory
process_images_in_directory(main_directory, threshold)

```

## 4. Thresholding for overall yearly 

```console

import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

def process_and_save_thresholded_image(image_path, threshold, save_folder):
    # Load the TIFF image
    image_data = tf.imread(image_path)
    
    # Create subplots for the original and thresholded images
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Original image plot
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Initialize an empty binary image (3D array for RGB representation)
    binary_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
    white_pixels = 0
    nan_pixels = 0

    # Iterate through each pixel and apply the threshold condition
    for row in range(image_data.shape[0]):
        for col in range(image_data.shape[1]):
            value = image_data[row, col]

            # Check if the value is NaN
            if np.isnan(value):
                # Set NaN values to red (255, 0, 0)
                binary_image[row, col] = [255, 0, 0]
                nan_pixels += 1  # Increment NaN pixel count
            elif value > threshold:
                # If the value is above the threshold, set it to white (255, 255, 255)
                binary_image[row, col] = [255, 255, 255]
                white_pixels += 1  # Increment white pixel count
            else:
                # If the value is below or equal to the threshold, set it to black (0, 0, 0)
                binary_image[row, col] = [0, 0, 0]

    # Calculate total valid pixels (excluding NaN)
    total_valid_pixels = image_data.size - nan_pixels

    # Calculate white pixel ratio (white pixels / total valid pixels)
    white_pixel_ratio = (white_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

    # Plot the thresholded image
    axs[1].imshow(binary_image)
    axs[1].set_title(f'Threshold = {threshold}\nWhite Pixel Ratio: {white_pixel_ratio:.2f}%')
    axs[1].axis('off')

    # Generate the file name and save the figure
    file_name = os.path.basename(image_path).replace('.tif', '_thresholded.png')
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)
    plt.close()

# Path to the TIFF image
image_path = r"C:\Users\Wajeeha\Desktop\overallyearly\overall_average_image.tif"

# Define the methane concentration threshold value
threshold = 1_000_000

# Define the save folder for the thresholded image
save_folder = r"C:\Users\Wajeeha\Desktop\overallyearly\thresholded"
os.makedirs(save_folder, exist_ok=True)

# Process and save the thresholded image
process_and_save_thresholded_image(image_path, threshold, save_folder)
```

### 5. Linear regression

```console
import os
import numpy as np
import tifffile as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def clip_image(image):
    """Clip the image to ensure values are non-negative."""
    return np.clip(image, a_min=0, a_max=None)  # Clip values to be non-negative

def load_tiff_images(directory, year_range, exclude_image):
    """Load TIFF images for the specified year range, excluding a specific image."""
    images = []
    for year in year_range:
        year_dir = os.path.join(directory, str(year))
        for month in range(1, 13):
            file_path = os.path.join(year_dir, f"{year}-{month:02d}.tif")
            if file_path == exclude_image:
                print(f"Excluding file: {file_path}")
                continue
            if os.path.exists(file_path):
                image = tf.imread(file_path)
                images.append((year, month, image))
            else:
                print(f"File not found: {file_path}")
                images.append((year, month, None))
    return images

def retrieve_pixel_values(images, month, x, y):
    """Retrieve pixel values from the images at position (x, y) for a specific month."""
    pixel_values = []
    for year, image_month, image in images:
        if image_month == month and image is not None and 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            pixel_value = image[x, y]
            if np.issubdtype(type(pixel_value), np.number) and not np.isnan(pixel_value):
                pixel_values.append(pixel_value)
    return pixel_values

def compute_linear_regression(pixel_values):
    """Predict the next pixel value using linear regression."""
    if not pixel_values:
        return np.nan

    X = np.arange(len(pixel_values)).reshape(-1, 1)  # Reshape for scikit-learn
    y = np.array(pixel_values)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the value following the last data point
    next_index = len(pixel_values)
    next_value = model.predict([[next_index]])[0]

    return next_value

def predict_images_for_year(images, year, months, output_directory):
    """Predict images for specified months of a year using the historical data."""
    predictions = {}
    all_actual_values = []
    all_predicted_values = []

    for month in months:
        monthly_images = valid_images_by_month.get(month, [])
        if monthly_images:
            height, width = monthly_images[0].shape
            predicted_image = np.empty((height, width))

            for x in range(height):  # Iterate over rows
                for y in range(width):  # Iterate over columns
                    pixel_values = retrieve_pixel_values(images, month, x, y)
                    next_value = compute_linear_regression(pixel_values)
                    predicted_image[x, y] = next_value

                    # Collect the actual and predicted values
                    if len(pixel_values) > 0:
                        all_actual_values.append(pixel_values[-1])  # Actual value (last known value)
                        all_predicted_values.append(next_value)  # Predicted value

            # Clip the predicted image to ensure non-negative values
            predicted_image = clip_image(predicted_image)

            # Store predicted image
            predictions[month] = predicted_image

            # Save the predicted image
            output_path = os.path.join(output_directory, f"predicted_{year}_{month:02d}.tif")
            tf.imwrite(output_path, predicted_image)

    return predictions, all_actual_values, all_predicted_values

# Directories
historical_directory = r"C:\Users\Wajeeha\Desktop\NP"
predicted_directory = r"C:\Users\Wajeeha\Desktop\ann"

# Exclude this image (not needed in the current context)
exclude_image_path = None

# Load historical data from 2019 to June 2024
year_range = range(2019, 2024)
images = load_tiff_images(historical_directory, year_range, exclude_image_path)

# Add images for 2024 up to June
for month in range(1, 7):
    file_path = os.path.join(historical_directory, '2024', f"2024-{month:02d}.tif")
    if os.path.exists(file_path):
        image = tf.imread(file_path)
        images.append((2024, month, image))
    else:
        print(f"File not found: {file_path}")
        images.append((2024, month, None))

# Initialize dictionary to group images by month
valid_images_by_month = {month: [] for month in range(1, 13)}

# Group images by month
for year, month, image in images:
    if image is not None:
        valid_images_by_month[month].append(image)

# Predict images for July to December 2024 using data from 2019-June 2024
months_to_predict_2024 = [7, 8, 9, 10, 11, 12]
predictions_2024, actual_values_2024, predicted_values_2024 = predict_images_for_year(images, 2024, months_to_predict_2024, predicted_directory)

# Update images list to include 2024 predictions
for month, predicted_image in predictions_2024.items():
    images.append((2024, month, predicted_image))

# Predict images for the entire year of 2025 using updated data
predictions_2025, actual_values_2025, predicted_values_2025 = predict_images_for_year(images, 2025, range(1, 13), predicted_directory)

# Combine actual and predicted values for error metrics
all_actual_values = actual_values_2024 + actual_values_2025
all_predicted_values = predicted_values_2024 + predicted_values_2025

# Ensure that actual and predicted values are not empty
if all_actual_values and all_predicted_values:
    # Calculate MSE, MAE, R², and RMSE
    mse = mean_squared_error(all_actual_values, all_predicted_values)
    mae = mean_absolute_error(all_actual_values, all_predicted_values)
    r2 = r2_score(all_actual_values, all_predicted_values)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Fit a linear regression model for plotting the regression line
    model = LinearRegression()
    X = np.array(all_actual_values).reshape(-1, 1)
    y = np.array(all_predicted_values)
    model.fit(X, y)
    regression_line = model.predict(X)

    # Plot the actual values vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_actual_values, all_predicted_values, color='blue', label='Predicted vs Actual')
    plt.plot(all_actual_values, regression_line, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Actual CH4_column_volume_mixing_ratio_dry_air (ppb)')
    plt.ylabel('Predicted CH4_column_volume_mixing_ratio_dry_air (ppb)')
    plt.title('Regression Line vs Predicted Values')
    plt.legend()
    plt.show()

else:
    print("Error: Actual or predicted values are empty.")


```
 
## 6. regression validation 
**6.1 to calculate the Root mean squraed error (RMSE) against the real images of 2024**

```console
import numpy as np
import tifffile as tf
from sklearn.metrics import mean_squared_error

def calculate_metrics(image_path1, image_path2):
    """Calculate RMSE, mean, minimum, and maximum pixel values between two TIFF images."""
    # Load the images
    image1 = tf.imread(image_path1)
    image2 = tf.imread(image_path2)

    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("The images do not have the same dimensions.")

    # Flatten the images to 1D arrays for comparison
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Calculate the RMSE
    mse = mean_squared_error(image1_flat, image2_flat)
    rmse = np.sqrt(mse)

    # Calculate the mean pixel values
    mean_image1 = np.mean(image1_flat)
    mean_image2 = np.mean(image2_flat)

    # Calculate the minimum and maximum pixel values
    min_image1 = np.min(image1_flat)
    max_image1 = np.max(image1_flat)
    min_image2 = np.min(image2_flat)
    max_image2 = np.max(image2_flat)

    return {
        "rmse": rmse,
        "mean_predicted": mean_image1,
        "mean_actual": mean_image2,
        "min_predicted": min_image1,
        "max_predicted": max_image1,
        "min_actual": min_image2,
        "max_actual": max_image2
    }

# Paths to the images
predicted_image_path = r"C:\Users\Wajeeha\Desktop\ann\predicted_2024_07.tif"
actual_image_path = r"C:\Users\Wajeeha\Desktop\july_average\2024-07_average.tif"

# Calculate metrics
metrics = calculate_metrics(predicted_image_path, actual_image_path)

# Display the results
print(f"Root Mean Squared Error (RMSE) between the images: {metrics['rmse']}")
print(f"Mean pixel value (mean methane column volume mixing ratio in dry air) for predicted image: {metrics['mean_predicted']}")
print(f"Mean pixel value (mean methane column volume mixing ratio in dry air) for actual image: {metrics['mean_actual']}")
print(f"Minimum pixel value for predicted image: {metrics['min_predicted']}")
print(f"Maximum pixel value for predicted image: {metrics['max_predicted']}")
print(f"Minimum pixel value for actual image: {metrics['min_actual']}")
print(f"Maximum pixel value for actual image: {metrics['max_actual']}")

```

## 7. Extra Codes
*7.1 graphical representation of deviations in methane Concentrartion using scatter plot graph. 

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

def load_images(directory, year_range):
    """Load TIFF images for the specified year range, excluding July 2024."""
    images = []
    for year in year_range:
        year_dir = os.path.join(directory, str(year))
        for month in range(1, 13):
            if year == 2024 and month == 7:
                continue  # Skip July 2024
            file_path = os.path.join(year_dir, f"{year}-{month:02d}.tif")
            if os.path.exists(file_path):
                print(f"Loading image for {year}-{month:02d}")  # Debugging print
                image = tf.imread(file_path)
                images.append((year, month, image))
            else:
                print(f"Image not found for {year}-{month:02d}")  # Debugging print
                images.append((year, month, None))
    return images

def calculate_monthly_means(images):
    """Calculate the mean CH4 density for each month across all years."""
    monthly_means = {month: [] for month in range(1, 13)}
    for year, month, image in images:
        if image is not None:
            mean_ch4_density = np.nanmean(image)  # Use np.nanmean to handle NaN values
            print(f"Mean CH4 for {year}-{month:02d}: {mean_ch4_density}")  # Print mean value
            monthly_means[month].append(mean_ch4_density)
        else:
            print(f"No data for {year}-{month:02d}")  # Debugging print
            monthly_means[month].append(np.nan)
    return monthly_means

def plot_monthly_trends(monthly_means, year_range):
    """Plot the trends of mean CH4 density for each month, excluding July 2024."""
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'August', 'September', 'October', 'November', 'December']  # Exclude July
    
    # Prepare data for plotting
    all_year_means = {year: [] for year in year_range}

    for month in range(1, 13):  # Include all months in calculations
        if month == 7:
            continue  # Skip July
        means = monthly_means[month]
        for index, year in enumerate(year_range):
            if len(means) > index:
                mean_value = means[index]
                all_year_means[year].append(mean_value)
            else:
                all_year_means[year].append(np.nan)  # Handle months with no data

    # Plotting
    plt.figure(figsize=(15, 8))
    for year in year_range:
        plt.plot(months, all_year_means[year], marker='o', label=f'Year {year}')

    plt.title('Mean CH4 column volume mixing ratio in dry air, Trends from 2019 to 2024 (Excluding July 2024)')
    plt.xlabel('Month')
    plt.ylabel('CH4 column volume mixing ratio dry air (as parts-per-billion)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Directories
main_directory = r"C:\Users\Wajeeha\Desktop\NP"

# Load all historical images (2019-2024)
year_range = range(2019, 2025)
images = load_images(main_directory, year_range)

# Calculate monthly means
monthly_means = calculate_monthly_means(images)

# Plot the trends
plot_monthly_trends(monthly_means, year_range)
```

  *7.1 graphical representation of deviations in methane Concentrartion using scatter plot graph for yearly.
```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

def load_yearly_images(directory, year_range):
    """Load yearly TIFF images for the specified year range."""
    yearly_means = {}
    for year in year_range:
        file_path = os.path.join(directory, f"{year}_yearly_average.tif")
        if os.path.exists(file_path):
            print(f"Loading yearly image for {year}")  # Debugging print
            image = tf.imread(file_path)
            mean_ch4_density = np.nanmean(image)  # Use np.nanmean to handle NaN values
            yearly_means[year] = mean_ch4_density
            print(f"Mean CH4 for {year}: {mean_ch4_density}")  # Print mean value
        else:
            print(f"Image not found for {year}")  # Debugging print
            yearly_means[year] = None
    return yearly_means

def plot_yearly_trends(yearly_means):
    """Plot the trends of mean CH4 density for each year."""
    years = list(yearly_means.keys())
    means = [yearly_means[year] for year in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, means, marker='o', linestyle='-', color='b')
    plt.title('Mean CH4 Column Volume Mixing Ratio in Dry Air (2019-2024)')
    plt.xlabel('Year')
    plt.ylabel('CH4 Column Volume Mixing Ratio Dry Air (parts-per-billion)')
    plt.grid(True)
    plt.show()

# Directory containing the yearly average images
main_directory = r"C:\Users\Wajeeha\Desktop\NP\yeraly averages"

# Year range from 2019 to 2024
year_range = range(2019, 2025)

# Load all yearly average images and calculate mean CH4 density
yearly_means = load_yearly_images(main_directory, year_range)

# Plot the yearly trends
plot_yearly_trends(yearly_means)

# Print the mean CH4 densities for each year
print("Mean CH4 Densities for Each Year:")
for year, mean_density in yearly_means.items():
    print(f"{year}: {mean_density}")
```

**7.2 generating missing dates**

```console
import ee
import datetime

# Initialize Earth Engine
ee.Initialize()

# Define a region of interest (ROI)
roi = ee.Geometry.Polygon(
    [[[50.717442, 24.455254],
      [51.693314, 24.455254],
      [50.717442, 26.226590],
      [51.693314, 26.226590]]])

def check_dataset_availability(start_date, end_date, extra_start_date=None, extra_end_date=None):
    # Convert start and end dates to datetime objects
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # List to store missing dates
    missing_dates = []

    # Function to check missing dates for a given period
    def check_period(start_dt, end_dt):
        current_date = start_dt
        while current_date < end_dt:
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Sentinel-5P CH4 dataset (Offline)
            collection = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
                          .filterBounds(roi)
                          .filterDate(ee.Date(current_date_str), ee.Date(current_date_str).advance(1, 'day'))
                          .select('CH4_column_volume_mixing_ratio_dry_air'))

            # Check if collection is empty
            if collection.size().getInfo() == 0:
                missing_dates.append(current_date_str)

            # Move to the next day
            current_date += datetime.timedelta(days=1)

    # Check for missing dates within the specified period
    check_period(start_datetime, end_datetime)

    # If extra periods are provided, check those as well
    if extra_start_date and extra_end_date:
        extra_start_datetime = datetime.datetime.strptime(extra_start_date, '%Y-%m-%d')
        extra_end_datetime = datetime.datetime.strptime(extra_end_date, '%Y-%m-%d')
        check_period(extra_start_datetime, extra_end_datetime)

    return missing_dates

# Define the main time range
start_date = '2019-02-08'
end_date = '2024-07-30'

# Define an additional period to check for missing data outside the main period
extra_start_date = '2024-08-01'
extra_end_date = '2024-08-31'

# Check dataset availability
missing_dates = check_dataset_availability(start_date, end_date, extra_start_date, extra_end_date)

# Print missing dates
if missing_dates:
    print("Missing dates:")
    for date in missing_dates:
        print(date)
else:
    print("No missing dates.")

```
