import gradio as gr
import torch
import os
import numpy as np
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig

# Environment variable configurations with defaults
PYTORCH_CUDA_ALLOC = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC

# Model configuration
DEFAULT_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct"
model_name = os.environ.get('MODEL_NAME', DEFAULT_MODEL)

# Generation parameters
DEFAULT_TEMP = 0.6
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048

TEMPERATURE = float(os.environ.get('MODEL_TEMPERATURE', DEFAULT_TEMP))
TOP_K = int(os.environ.get('MODEL_TOP_K', DEFAULT_TOP_K))
TOP_P = float(os.environ.get('MODEL_TOP_P', DEFAULT_TOP_P))
MAX_OUTPUT_TOKENS = int(os.environ.get('MAX_OUTPUT_TOKENS', DEFAULT_MAX_TOKENS))

# Image configuration
DEFAULT_MAX_IMAGE_SIZE = (1120, 1120)
MAX_IMAGE_WIDTH = int(os.environ.get('MAX_IMAGE_WIDTH', DEFAULT_MAX_IMAGE_SIZE[0]))
MAX_IMAGE_HEIGHT = int(os.environ.get('MAX_IMAGE_HEIGHT', DEFAULT_MAX_IMAGE_SIZE[1]))
MAX_IMAGE_SIZE = (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)

# Server configuration
SERVER_NAME = os.environ.get('SERVER_NAME', '0.0.0.0')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 7860))

print(f"Loading model: {model_name}")
print(f"Server config: {SERVER_NAME}:{SERVER_PORT}")
print(f"Model parameters: temp={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, max_tokens={MAX_OUTPUT_TOKENS}")
print(f"Image size: {MAX_IMAGE_SIZE}")

# Model configuration
model = MllamaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)
# Tie weights and set to eval mode
model.tie_weights()
model.eval()
print("Model device:", next(model.parameters()).device)
print("Available device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Visual theme
visual_theme = gr.themes.Default()

# Function to process the image and generate a description
def describe_image(image, user_prompt, temperature, top_k, top_p, max_tokens, history):
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        
        # Format messages exactly as in the example
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        # Process text using chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # Print debug information
        print("Input shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")

        print("Starting generation...")
        print(f"GPU Memory before generation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(256, max_tokens),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
            
        print(f"GPU Memory after generation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
        print("Generation complete!")
        
        # Decode the output
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        # Clean up the output if needed
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text):].strip()

        history.append((user_prompt, generated_text))
        return history

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Detailed error: {e}")
        import traceback
        traceback.print_exc()
        history.append((user_prompt, error_message))
        return history

def clear_chat():
    return []

def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    Clean-UI
    </h1>
    <p style='text-align: center'>
    Current Model: {}
    </p>
    """.format(model_name))
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=MAX_IMAGE_HEIGHT,
                    width=MAX_IMAGE_WIDTH
                )
                
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=TEMPERATURE, step=0.1, interactive=True)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=TOP_K, step=1, interactive=True)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=TOP_P, step=0.1, interactive=True)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=MAX_OUTPUT_TOKENS, step=50, interactive=True)

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512)

                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

                generate_button.click(
                    fn=describe_image, 
                    inputs=[image_input, user_prompt, temperature, top_k, top_p, max_tokens, chat_history],
                    outputs=[chat_history]
                )

                clear_button.click(
                    fn=clear_chat,
                    inputs=[],
                    outputs=[chat_history]
                )

    return demo

# Launch the interface
demo = gradio_interface()
demo.launch(
    server_name=SERVER_NAME,
    server_port=SERVER_PORT,
    share=False  # Disable share link for security
)
