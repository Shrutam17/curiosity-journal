<!-- ## Memory Journal Creator

A creative tool that combines conversation quotes and images to create memorable digital journal entries.

## Features

- Extract meaningful quotes from conversations using AI
- Combine multiple images in various layouts
- Add decorative elements and text styling
- Support for different output styles

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/memory-journal.git
cd memory-journal
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your OpenAI API key: -->



# Curiosity Journal

A creative application that generates memory journal images from conversations and photos.

## Overview

Curiosity Journal is a Python application that:
1. Takes a conversation transcript and photos as input
2. Uses AI to extract interesting quotes from the conversation
3. Combines the quotes and photos into a creative, visually appealing image
4. Offers multiple layout styles and customization options

## Features

- Extract meaningful quotes from conversations using OpenAI's API
- Process and enhance images with decorative frames and effects
- Create visually appealing layouts combining quotes and photos
- Apply different style options (modern, vintage, artistic, minimal)
- Optional AI image enhancement with Stable Diffusion
- Available as both a web application and a command-line tool

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository or unzip the provided archive:
   ```
   git clone https://github.com/yourusername/curiosity-journal.git
   cd curiosity-journal
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Web Interface

1. Start the web server:
   ```
   python main.py
   ```
   Or:
   ```
   export FLASK_ENV=development
   python -m flask run
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload a conversation file (txt) and photos through the web interface

4. Select your preferred style

5. Click "Create Memory Journal" to generate your image

### Command Line

Generate a memory journal from the command line:

```
python main.py path/to/conversation.txt path/to/image1.jpg path/to/image2.jpg --output output.png --style vintage
```

Options:
- `--output` or `-o`: Output file path (default: memory_journal.png)
- `--style` or `-s`: Layout style (choices: modern, vintage, minimal, artistic; default: modern)

## AI Integration

### Quote Extraction

This application uses OpenAI's GPT models to extract meaningful quotes from conversations. The API calls are made in the `quote_extractor.py` file, specifically in the `extract_quotes()` function.

### Image Enhancement (Optional)

For advanced image enhancement, the application can use Hugging Face's Stable Diffusion model. This functionality is implemented in the `enhance_with_ai()` function in `image_processor.py`.

## Development Notes

- The application is structured to separate concerns: quote extraction, layout management, and image processing.
- If you encounter CUDA/GPU issues with Stable Diffusion, the application will fall back to basic image processing.
- Font availability depends on your operating system - the application includes fallbacks if specific fonts aren't available.

## Example

Included in the `examples` directory:
- `sample_conversation.txt`: A sample conversation file
- `images/`: Sample images to use

Run the example:
```
python main.py examples/sample_conversation.txt examples/images/image1.png examples/images/image2.png
```

## License

This project is licensed under the MIT License.