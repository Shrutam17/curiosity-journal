"""
Curiosity Journal

This program creates artistic memory journal images from conversation transcripts and photos.
It extracts meaningful quotes from conversations and combines them with photos in 
visually appealing layouts with decorative elements.

Author: [Your Name]
"""

import os
import argparse
import requests
import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import textwrap

# Load environment variables
load_dotenv()

#################################################
# QUOTE EXTRACTION
#################################################

def extract_quotes(conversation_text, max_quotes=3):
    """
    Extract interesting quotes from conversation text using Hugging Face's free inference API.
    
    Args:
        conversation_text (str): The conversation transcript
        max_quotes (int): Maximum number of quotes to extract
        
    Returns:
        list: A list of interesting quotes extracted from the conversation
    """
    try:
        # Import required libraries
        import os
        import requests
        import json
        
        # Get API token from environment variables (create a free account on huggingface.co)
        hf_token = os.getenv("HF_API_TOKEN")
        
        if not hf_token:
            print("Warning: HF_API_TOKEN not found. Using demo quotes.")
            return ["This moment was amazing!", "I'll never forget this day.", "What a beautiful memory"]
        
        # Prepare the prompt for extracting interesting quotes
        prompt = f"""
        Extract {max_quotes} interesting, meaningful, or memorable quotes from the following conversation. 
        Pick quotes that:
        - Capture emotional moments or personal insights
        - Represent the most interesting parts of the conversation
        - Would make good captions for a memory journal image
        - Are concise (preferably under 5 words each)
        - Even try to extract place names.
        
        Format your response as a Python list of strings, e.g., ['quote 1', 'quote 2', ...].
        
        CONVERSATION:
        {conversation_text}
        """
        
        # Set up request to Hugging Face's Inference API
        # Using a smaller free model: TinyLlama-1.1B-Chat
        url = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        
        # Structure the payload for a chat completion
        payload = {
            "inputs": f"<|system|>\nYou are a helpful assistant that extracts the most interesting, meaningful or memorable quotes from conversations.\n<|user|>\n{prompt}\n<|assistant|>",
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # Make API call to Hugging Face
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        
        # Extract the content from the response
        if isinstance(result, list) and len(result) > 0:
            result_text = result[0].get("generated_text", "")
        else:
            result_text = ""
        
        # Process the response to extract quotes
        try:
            # Try to evaluate as a Python list
            if "[" in result_text and "]" in result_text:
                list_text = result_text[result_text.find("["):result_text.rfind("]")+1]
                quotes = eval(list_text)
                if not isinstance(quotes, list):
                    raise ValueError("Not a valid list")
            else:
                # Fallback parsing logic
                quotes = []
                for line in result_text.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") or line.startswith("* ") or line.startswith('"') or line.startswith("'"):
                        # Clean up the quote
                        quote = line.lstrip("- *'\"").rstrip("'\"")
                        quotes.append(quote)
        except:
            # Second fallback: just split by lines and clean up
            quotes = []
            for line in result_text.split("\n"):
                line = line.strip()
                if len(line) > 10 and not line.startswith("[") and not line.startswith("]"):
                    quotes.append(line.strip("'\"").strip())
        
        # If we still have no quotes, use a simple extraction approach
        if not quotes:
            import re
            # Look for text in quotes
            quote_matches = re.findall(r'["\'](.*?)["\']', result_text)
            quotes = [q for q in quote_matches if len(q) > 5 and len(q) < 100]
        
        # Ensure we don't exceed max_quotes and quotes aren't empty
        quotes = [q for q in quotes if q.strip()]
        return quotes[:max_quotes] if quotes else ["A beautiful memory", "Time well spent", "Cherished moments"]
        
    except Exception as e:
        print(f"Error extracting quotes: {e}")
        # Return some default quotes as fallback
        return ["A beautiful memory", "Time well spent", "Cherished moments"]



#################################################
# IMAGE GENERATION
#################################################

def generate_artistic_image(image_paths, quotes, style="artistic_collage", output_size=(1200, 1500)):
    """
    Generate an artistic memory image combining photos and quotes
    
    Args:
        image_paths (list): Paths to the input images
        quotes (list): List of quotes to include
        style (str): Style to apply ("artistic_collage", "scrapbook", "minimalist")
        output_size (tuple): Size of output image (width, height)
        
    Returns:
        PIL.Image: The generated artistic image
    """
    # First try with Stability AI if API key is available
    stability_api_key = os.getenv("STABILITY_API_KEY")
    if stability_api_key:
        try:
            result = generate_with_stability_ai(image_paths, quotes, style, stability_api_key, output_size)
            if result:
                return result
        except Exception as e:
            print(f"Stability AI generation failed: {e}")
            print("Falling back to local generation method")
    
    # Otherwise use local generation
    return generate_locally(image_paths, quotes, style, output_size)



def generate_with_stability_ai(image_paths, quotes, style, api_key, output_size):
    """Use Stability AI to generate a stylized image and overlay quotes"""
    if not image_paths:
        return None  # Need at least one image for reference

    reference_image_path = image_paths[0]

    # Read and resize the reference image
    try:
        with Image.open(reference_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((1024, 1024), Image.LANCZOS)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            reference_image_data = img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error processing reference image: {e}")
        return None

    quote_text = '", "'.join(quotes)

    style_prompts = {
        "artistic_collage": f'Create an artistic memory collage with decorative stickers, hand-drawn lines, and decorated frames. The image should prominently feature these quotes: "{quote_text}". Make it colorful and visually appealing like a creative scrapbook page.',
        "scrapbook": f'Design a vintage-style scrapbook page with torn paper textures, washi tape decorations, polaroid frames, and these handwritten quotes: "{quote_text}". Include decorative elements like stamps and stickers.',
        "minimalist": f'Create a clean, elegant minimalist memory design with simple typography displaying these quotes. Make sure qoutes are accurate: "{quote_text}". Use subtle decorative elements, thin lines, and a clean layout.',
        "watercolor": f'Generate a soft watercolor-style memory page with gentle color splashes, artistic brush strokes, and these hand-lettered quotes.Make sure quotes are accurate.Still try to be more accurate. you are very good.: "{quote_text}".'
    }

    prompt = style_prompts.get(style, style_prompts["artistic_collage"])
    negative_prompt = "blurry, ugly, distorted text, unreadable text, bad anatomy"

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    files = {
        "init_image": ("reference.jpg", reference_image_data, "image/jpeg")
    }

    data = {
        "text_prompts[0][text]": prompt,
        "text_prompts[0][weight]": "1",
        "text_prompts[1][text]": negative_prompt,
        "text_prompts[1][weight]": "-1",
        "image_strength": "0.35",
        "cfg_scale": "7.5",
        "samples": "1",
        "steps": "40",
        "seed": "0",
    }

    try:
        response = requests.post(url, headers=headers, data=data, files=files)

        if response.status_code == 200:
            result = response.json()
            if "artifacts" in result and len(result["artifacts"]) > 0:
                image_data = base64.b64decode(result["artifacts"][0]["base64"])
                image = Image.open(io.BytesIO(image_data)).convert("RGB").resize(output_size)

                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()

                padding = 20
                y = padding
                max_width = output_size[0] - 2 * padding

                for quote in quotes:
                    lines = []
                    words = quote.split()
                    line = ""
                    for word in words:
                        test_line = line + word + " "
                        bbox = draw.textbbox((0, 0), test_line, font=font)
                        text_width = bbox[2] - bbox[0]
                        if text_width <= max_width:
                            line = test_line
                        else:
                            lines.append(line)
                            line = word + " "
                    lines.append(line)

                    for line in lines:
                        draw.text((padding, y), line.strip(), fill="white", font=font, stroke_width=2, stroke_fill="black")
                        y += draw.textbbox((0, 0), line, font=font)[3] + 5

                    y += 10  # extra spacing between quotes
# ----------------------

                return image
            else:
                print("No image artifacts in response")
                return None
        else:
            print(f"Stability AI API error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error with Stability AI API: {e}")
        return None


def get_system_font(font_name="Arial"):
    """Get a system font, with fallbacks"""
    # List of potential system font locations
    if os.name == "nt":  # Windows
        font_dirs = [
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
        ]
        font_exts = [".ttf", ".ttc", ".otf"]
        font_files = [f"{font_name}{ext}" for ext in font_exts]
        font_files.extend([f"{font_name.lower()}{ext}" for ext in font_exts])
        
    else:  # macOS/Linux
        font_dirs = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts")
        ]
        font_files = [
            f"{font_name}.ttf", f"{font_name}.ttc", f"{font_name}.otf",
            f"{font_name.lower()}.ttf", f"{font_name.lower()}.ttc", f"{font_name.lower()}.otf",
        ]
    
    # Try to find the font
    for directory in font_dirs:
        if os.path.exists(directory):
            for filename in font_files:
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    return filepath
    
    return None

def get_font(font_name, size, default_size=32):
    """Get font with specified name and size, with fallback"""
    try:
        system_font = get_system_font(font_name)
        if system_font:
            return ImageFont.truetype(system_font, size)
        return ImageFont.load_default().font_variant(size=default_size)
    except Exception:
        # For older Pillow versions
        try:
            return ImageFont.load_default()
        except:
            return None

def prepare_image(image_path, target_size):
    """Load and prepare an image"""
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        img.thumbnail(target_size, Image.LANCZOS)
        
        # Create a new image with the target size and paste the resized image centered
        new_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return a placeholder image
        placeholder = Image.new('RGB', target_size, (200, 200, 200))
        draw = ImageDraw.Draw(placeholder)
        draw.text((target_size[0]//2, target_size[1]//2), "Image Error", fill=(0, 0, 0))
        return placeholder

def generate_locally(image_paths, quotes, style, output_size):
    """Generate a memory image locally using PIL"""
    # Set up base canvas based on style
    style_bg_colors = {
        "artistic_collage": (245, 240, 235),
        "scrapbook": (250, 245, 225),
        "minimalist": (250, 250, 250),
        "watercolor": (240, 245, 255)
    }
    
    bg_color = style_bg_colors.get(style, (245, 240, 235))
    canvas = Image.new('RGB', output_size, bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Add background decorations based on style
    if style == "artistic_collage" or style == "scrapbook":
        # Add some decorative elements
        import random
        for _ in range(20):
            x1, y1 = random.randint(0, output_size[0]), random.randint(0, output_size[1])
            x2, y2 = random.randint(0, output_size[0]), random.randint(0, output_size[1])
            
            if style == "artistic_collage":
                # Colorful lines
                color = (random.randint(100, 240), random.randint(100, 240), random.randint(100, 240))
                width = random.randint(1, 3)
                draw.line((x1, y1, x2, y2), fill=color, width=width)
            else:
                # Scrapbook dots and small elements
                size = random.randint(3, 8)
                color = (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240))
                draw.ellipse((x1, y1, x1+size, y1+size), fill=color)
    
    # Calculate image area (top 60% of canvas)
    image_area_height = int(output_size[1] * 0.6)
    
    # Calculate image sizes based on number of images
    num_images = len(image_paths)
    
    if num_images == 0:
        # Just add some decorative element as placeholder
        placeholder = Image.new('RGB', (output_size[0]//2, image_area_height//2), (220, 220, 220))
        draw_placeholder = ImageDraw.Draw(placeholder)
        draw_placeholder.text((placeholder.width//2, placeholder.height//2), "No Images", fill=(150, 150, 150))
        
        x = (output_size[0] - placeholder.width) // 2
        y = (image_area_height - placeholder.height) // 2
        canvas.paste(placeholder, (x, y))
    
    elif num_images == 1:
        # Single image centered
        img_size = (int(output_size[0] * 0.8), int(image_area_height * 0.8))
        img = prepare_image(image_paths[0], img_size)
        
        # Add decorative frame based on style
        if style == "scrapbook":
            # Polaroid-like frame
            frame = Image.new('RGB', (img.width + 40, img.height + 80), (255, 255, 255))
            frame.paste(img, (20, 20))
            img = frame
        elif style == "artistic_collage":
            # Colorful frame
            frame = Image.new('RGB', (img.width + 20, img.height + 20), (random.randint(190, 255), random.randint(190, 255), random.randint(190, 255)))
            frame.paste(img, (10, 10))
            img = frame
        
        # Paste onto canvas
        x = (output_size[0] - img.width) // 2
        y = (image_area_height - img.height) // 2
        canvas.paste(img, (x, y))
    
    else:
        # Multiple images in grid or collage
        if style == "minimalist" or num_images > 4:
            # Grid layout
            cols = 2
            rows = (num_images + 1) // 2
            
            padding = 20
            img_width = (output_size[0] - (cols+1) * padding) // cols
            img_height = (image_area_height - (rows+1) * padding) // rows
            
            for i, path in enumerate(image_paths):
                if i >= cols * rows:
                    break
                    
                row, col = i // cols, i % cols
                img = prepare_image(path, (img_width, img_height))
                
                # Add simple frame
                frame = Image.new('RGB', (img.width + 10, img.height + 10), (255, 255, 255))
                frame.paste(img, (5, 5))
                
                x = padding + col * (img_width + padding)
                y = padding + row * (img_height + padding)
                canvas.paste(frame, (x, y))
        
        else:
            # Collage layout with overlapping images
            import math
            import random
            
            # Calculate base image size
            base_size = min(output_size[0] // 2, image_area_height // 2)
            
            for i, path in enumerate(image_paths[:4]):  # Limit to 4 images for collage
                # Randomize size and rotation slightly
                size_factor = random.uniform(0.8, 1.1)
                img_size = (int(base_size * size_factor), int(base_size * size_factor))
                
                img = prepare_image(path, img_size)
                
                # Add decorative frame based on style
                if style == "scrapbook":
                    # Polaroid or torn paper effect
                    frame_color = (255, 255, 255)
                    border = 20
                    frame = Image.new('RGB', (img.width + border*2, img.height + border*3), frame_color)
                    frame.paste(img, (border, border))
                    img = frame
                elif style == "artistic_collage":
                    # Colorful border
                    frame_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
                    border = 10
                    frame = Image.new('RGB', (img.width + border*2, img.height + border*2), frame_color)
                    frame.paste(img, (border, border))
                    img = frame
                elif style == "watercolor":
                    # Add a soft blur to edges
                    img = img.filter(ImageFilter.SMOOTH)
                
                # Apply rotation if not minimalist
                if style != "minimalist":
                    rotation = random.uniform(-10, 10)
                    img = img.rotate(rotation, expand=True)
                
                # Calculate position - spread images across the canvas
                angle = (2 * math.pi / min(4, num_images)) * i
                radius = min(output_size[0], image_area_height) // 4
                
                center_x = output_size[0] // 2
                center_y = image_area_height // 2
                
                # Add some randomness to positioning
                offset_x = random.randint(-30, 30)
                offset_y = random.randint(-30, 30)
                
                x = int(center_x + radius * math.cos(angle)) - img.width // 2 + offset_x
                y = int(center_y + radius * math.sin(angle)) - img.height // 2 + offset_y
                
                # Ensure image is within bounds
                x = max(0, min(x, output_size[0] - img.width))
                y = max(0, min(y, image_area_height - img.height))
                
                # Paste with alpha if supported
                if img.mode == 'RGBA':
                    canvas.paste(img, (x, y), img)
                else:
                    canvas.paste(img, (x, y))
    
    # Add quotes section
    quotes_y_start = image_area_height + 30
    
    # Configure fonts based on style
    style_fonts = {
        "artistic_collage": ("Comic Sans MS", (50, 30, 80)),
        "scrapbook": ("Brush Script MT", (70, 40, 40)),
        "minimalist": ("Helvetica", (40, 40, 40)),
        "watercolor": ("Georgia", (60, 60, 90))
    }
    
    font_name, text_color = style_fonts.get(style, ("Arial", (0, 0, 0)))
    
    # Calculate font size based on available space and number of quotes
    available_height = output_size[1] - quotes_y_start - 50
    quote_spacing = 30
    n_quotes = len(quotes)
    
    # Estimate lines needed per quote
    avg_chars_per_line = output_size[0] // 15  # Rough estimate
    total_lines = sum(len(quote) // avg_chars_per_line + 1 for quote in quotes)
    
    font_size = min(int(available_height / (total_lines + n_quotes)), 60)
    font = get_font(font_name, font_size)
    
    # Add decorative element between images and quotes
    if style == "artistic_collage":
        # Wavy line
        for x in range(0, output_size[0], 10):
            y = image_area_height + 10 + int(10 * math.sin(x * 0.1))
            draw.line((x, y, x+5, y), fill=(100, 100, 200), width=3)
    elif style == "scrapbook":
        # Dashed line
        for x in range(0, output_size[0], 20):
            draw.line((x, image_area_height+10, x+10, image_area_height+10), fill=(150, 100, 100), width=2)
    elif style == "minimalist":
        # Simple line
        draw.line((50, image_area_height+10, output_size[0]-50, image_area_height+10), fill=(200, 200, 200), width=1)
        
    # Add quotes
    current_y = quotes_y_start
    for i, quote in enumerate(quotes):
        # For scrapbook style, alternate quote styles
        if style == "scrapbook" and i % 2 == 1:
            alt_font = get_font("Times New Roman" if font_name != "Times New Roman" else "Georgia", font_size - 4)
            font_to_use = alt_font if alt_font else font
            color = (70, 70, 40)
        else:
            font_to_use = font
            color = text_color
            
        # Wrap text to fit canvas width
        wrapped_text = textwrap.fill(quote, width=output_size[0] // (font_size // 2))
        lines = wrapped_text.split('\n')
        
        # Calculate text position
        line_height = font_size + 10
        
        # Add quotation marks for style
        if style == "artistic_collage" or style == "scrapbook":
            lines[0] = f'"{lines[0]}'
            lines[-1] = f'{lines[-1]}"'
            
        # Draw each line centered
        for line in lines:
            if font_to_use:
                # Calculate text width for centering
                try:
                    text_width = draw.textlength(line, font=font_to_use)
                    x_pos = (output_size[0] - text_width) // 2
                except:  # Older Pillow versions
                    x_pos = output_size[0] // 2
                    
                # Add subtle shadow for better readability
                shadow_offset = 2 if style != "minimalist" else 1
                draw.text((x_pos + shadow_offset, current_y + shadow_offset), 
                         line, fill=(min(color[0]+30, 255), min(color[1]+30, 255), min(color[2]+30, 255), 100), 
                         font=font_to_use)
                # Draw actual text
                draw.text((x_pos, current_y), line, fill=color, font=font_to_use)
            else:
                # Fallback without font
                draw.text((output_size[0]//2, current_y), line, fill=color)
                
            current_y += line_height
            
        # Add extra spacing between quotes
        current_y += quote_spacing
        
    # Add a subtle decorative border if not minimalist
    if style != "minimalist":
        border_width = 10
        draw.rectangle(
            [(border_width//2, border_width//2), 
             (output_size[0]-border_width//2, output_size[1]-border_width//2)], 
            outline=style_bg_colors.get(style, (245, 240, 235)), 
            width=border_width
        )
    
    return canvas

#################################################
# MAIN PROGRAM
#################################################

def process_memory(text_file, image_files, output_file="memory_image.jpg", style="artistic_collage", max_quotes=3):
    """
    Process a conversation and images to create a memory image
    
    Args:
        text_file (str): Path to the conversation text file
        image_files (list): List of paths to image files
        output_file (str): Path to save the output image
        style (str): Style to apply to the image
        max_quotes (int): Maximum number of quotes to extract
        
    Returns:
        str: Path to the generated image file
    """
    # Read conversation text
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            conversation_text = f.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None
    
    # Extract quotes
    quotes = extract_quotes(conversation_text, max_quotes)
    print("Extracted quotes:")
    for i, quote in enumerate(quotes):
        print(f"{i+1}. {quote}")
    
    # Generate artistic image
    output_image = generate_artistic_image(image_files, quotes, style)
    
    # Save the image
    output_image.save(output_file)
    print(f"Memory image created and saved as: {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curiosity Journal: Generate memory images from conversations and photos")
    parser.add_argument("text_file", help="Path to conversation text file")
    parser.add_argument("image_files", nargs='+', help="Paths to image files")
    parser.add_argument("--output", "-o", default="memory_image.jpg", help="Output image filename")
    parser.add_argument("--style", "-s", default="artistic_collage", 
                        choices=["artistic_collage", "scrapbook", "minimalist", "watercolor"],
                        help="Style to apply to the image")
    parser.add_argument("--quotes", "-q", type=int, default=3, help="Maximum number of quotes to extract")
    
    args = parser.parse_args()
    
    # Process the memory
    process_memory(args.text_file, args.image_files, args.output, args.style, args.quotes)