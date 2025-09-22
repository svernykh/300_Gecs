import os
import shutil
import pandas as pd
from icrawler.builtin import BingImageCrawler

# List of Indonesian food to search
food_list = [
    'gado gado ',
    'nasi goreng ',
    'soto ayam ',
    'bakso ',
    'rendang '
    
]

# Create test directory if it doesn't exist
test_dir = './test'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

def download_and_organize_images():
    # Number of images to download per category
    num_images = 60
    
    # List to store image data for CSV
    image_data = []

    for food in food_list:
        print(f"\nDownloading images for: {food}")
        
        # Extract the main food name (first two words)
        food_name = '_'.join(food.split()[:2])
        # Get the label (without 'indonesian food')
        label = ' '.join(food.split()[:2])
        
        # Create temporary directory for this food
        temp_dir = f'temp_downloads/{food_name}'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Download images using BingImageCrawler
        crawler = BingImageCrawler(
            storage={'root_dir': temp_dir}
        )
        crawler.crawl(keyword=food, max_num=num_images)
        
        # Move images to test directory with proper naming
        source_dir = temp_dir
        if os.path.exists(source_dir):
            for i, filename in enumerate(os.listdir(source_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Create new filename
                    extension = os.path.splitext(filename)[1]
                    new_filename = f"{food_name}_{i+1}{extension}"
                    
                    # Move and rename file
                    source_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(test_dir, new_filename)
                    shutil.copy2(source_path, dest_path)
                    print(f"Saved: {new_filename}")
                    
                    # Add to image data
                    image_data.append({
                        'filename': new_filename,
                        'label': label
                    })

    # Clean up temporary downloads
    if os.path.exists('temp_downloads'):
        shutil.rmtree('temp_downloads')
        
    # Create CSV file
    df = pd.DataFrame(image_data)
    csv_path = 'test.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV file created: {csv_path}")
    print(f"Total images: {len(image_data)}")

if __name__ == "__main__":
    print("Starting image download for Indonesian food dataset...")
    download_and_organize_images()
    print("\nDownload completed! Images are saved in the 'test' directory.")