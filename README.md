# Beyond Salon

## Project Objective:

Beyond Salon is a web application that allows users to upload an image and select a hair color to see how the color suits them. The application uses Gradio for the user interface and Hugging Face Spaces for deployment.

## Project Goals
- The Goal is to either create or use existing models to figure out the hair masking in each image and then apply the hair transformation to the hair.
- Used existing models from segment_anything and diffusers
  
## Files
* [Gradio_UI.ipynb] - User Interface for handling the input and applying the transformation 
* [hair.py] - main class that creates the hair mask and applies transformation
* [/lib](https://github.com/mcalabrese98/project2/tree/main/lib) - Houses the python library files
* [/source_data](https://github.com/mcalabrese98/project2/tree/main/source_data) - All the source data files are stored here

## Features

- Upload an image of yourself.
- Select a hair color from a dropdown menu.
- See the transformed image with the selected hair color.

## Requirements

- Python 3.7+
- Gradio
- Pillow
- OpenCV
- NumPy
- Webcolors

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/beyond-salon.git
    cd beyond-salon
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application locally:
    ```bash
    python app.py
    ```

2. Open your web browser and go to `http://localhost:7860` to use the application.

## Deployment

To deploy the application to Hugging Face Spaces for free permanent hosting and GPU upgrades, follow these steps:

1. Install the Gradio CLI:
    ```bash
    pip install gradio
    ```

2. Run the deploy command:
    ```bash
    gradio deploy
    ```

3. Follow the prompts:
    - Enter Spaces app title: `Beyond_Salon`
    - Enter Gradio app file: [app.py](http://_vscodecontentref_/1)
    - Enter Spaces hardware: `t4-small`
