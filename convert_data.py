from PIL import Image, ImageDraw
import os

def resize_with_baseline(image_path, target_width=50, target_height=150, background_color='white', line_color='black'):
    try:
        with Image.open(image_path) as img:
            # Convert image to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate aspect ratio
            aspect = img.width / img.height
            
            # Calculate new dimensions while maintaining aspect ratio
            if aspect > target_width/target_height:
                new_width = target_width
                new_height = int(new_width / aspect)
            else:
                new_height = target_height
                new_width = int(new_height * aspect)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with white background
            new_img = Image.new('RGB', (target_width, target_height), background_color)
            
            # Calculate position to paste (center)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            
            # Paste resized image onto white background
            new_img.paste(resized_img, (paste_x, paste_y))
            
            
            # Create new output path in data1 directory
            relative_path = os.path.relpath(image_path, "../data")
            output_path = os.path.join("../data_new", relative_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the image
            new_img.save(output_path)
            
            return {
                "success": True,
                "output_path": output_path,
                "original_size": img.size,
                "new_size": new_img.size
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
def process_data_folder(data_dir):
    """
    Process all images in all subfolders of the data directory
    """
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Get full path of the image
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                
                try:
                    # Process the image
                    result = resize_with_baseline(image_path)
                    
                    if result["success"]:
                        print(f"Successfully processed: {file}")
                    else:
                        print(f"Failed to process {file}: {result['error']}")
                        
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    
if __name__ == "__main__":
    # Path to your data directory
    data_dir = "../data"
    # Process all images
    process_data_folder(data_dir)
    print("Processing complete!")                   

# # Example usage
# if __name__ == "__main__":
#     image_path = "/Volumes/data/workspace/fast-api/pic_100.jpg"
#     result = resize_with_baseline(image_path)
    
#     if result["success"]:
#         print(f"Original size: {result['original_size']}")
#         print(f"New size: {result['new_size']}")
#         print(f"Saved to: {result['output_path']}")
#     else:
#         print(f"Error: {result['error']}")