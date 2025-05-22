import os
import argparse
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path, output_dir=None, dpi=300, output_format='jpg'):
    """
    Convert a PDF file to high-resolution images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the output images. Defaults to same directory as PDF.
        dpi (int, optional): Resolution in DPI. Higher values mean higher resolution. Defaults to 300.
        output_format (str, optional): Format for output images. Defaults to 'png'.
    
    Returns:
        list: Paths to the saved images
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the PDF filename without extension
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Convert PDF to images
    print(f"Converting '{pdf_path}' to {output_format} images at {dpi} DPI...")
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt=output_format,
        thread_count=os.cpu_count()
    )
    
    # Save images
    image_paths = []
    for i, image in enumerate(images):
        # Create filename for each page
        if len(images) == 1:
            image_path = os.path.join(output_dir, f"{pdf_filename}.{output_format}")
        else:
            image_path = os.path.join(output_dir, f"{pdf_filename}_page_{i+1}.{output_format}")
        
        # Save image
        image.save(image_path)
        image_paths.append(image_path)
        print(f"Saved: {image_path}")
    
    return image_paths

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Convert PDF to high-resolution images')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output-dir', '-o', help='Output directory for images')
    parser.add_argument('--dpi', '-d', type=int, default=300, 
                        help='Resolution in DPI (higher values = higher quality)')
    parser.add_argument('--format', '-f', default='jpg', choices=['png', 'jpg', 'jpeg', 'tiff'],
                        help='Output image format')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert PDF to images
    image_paths = convert_pdf_to_images(
        args.pdf_path,
        args.output_dir,
        args.dpi,
        args.format
    )
    
    print(f"\nConversion complete! {len(image_paths)} images saved.")

if __name__ == "__main__":
    main()