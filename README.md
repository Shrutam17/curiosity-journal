 ## Curiosity-Journal

This project creates a beautiful artistic memory image by combining quotes extracted from a conversation and a set of user-provided images. It can optionally use Stability AI to stylize the image or fall back to a local generator.

## Features

- Extract meaningful, short quotes using Hugging Face LLM (TinyLlama)
- Generate a styled collage using Stability AI (or fallback to PIL if no API)
- Supports multiple visual styles: `artistic_collage`, `scrapbook`, `minimalist`, `watercolor`
- Quotes are overlaid on the generated image

## Setup Instructions



1.  Clone or Download
```Clone the repo```

2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your API keys
HF_API_TOKEN=your_huggingface_token_here
STABILITY_API_KEY=your_stability_ai_key_here

## How to Run
python curiosity_journal.py examples/sample_conversation.txt examples/images/image1.jpg examples/images/image2.jpg -o my_journal.png -s artistic_collage

Change -s to scrapbook, minimalist, or watercolor to try other styles.




1. AI API Usage Locations
extract_quotes() in curiosity_journal.py: uses Hugging Face API for quote extraction.
generate_with_stability_ai(): uses Stability AI image-to-image endpoint to stylize image.

2. Output :
A final image like my_journal.png will be created.
Quotes are beautifully overlaid with chosen style and layout.

## Demo
[Watch the demo video](https://www.loom.com/share/5e6ec00639744eeca239024541224145?sid=c06d86f9-f115-4180-a4f2-2a63a7e25c34)


This project is licensed under the MIT License.
