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
# Try to tie weights if the model supports it
try:
    model.tie_weights()
except AttributeError:
    print("Model does not support weight tying or weights are already tied")
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
            max_new_tokens=max_tokens,
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

def describe_image_with_debug(image, user_prompt, temp, top_k, top_p, max_tokens, history, width, height):
    global MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT
    MAX_IMAGE_WIDTH = int(width)
    MAX_IMAGE_HEIGHT = int(height)
    
    debug_data = {
        "Status": "Starting processing...",
        "GPU Memory": f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
        "Image Info": {},
        "Model Info": {},
        "Generation Info": {}
    }
    
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        debug_data["Image Info"].update({
            "Size": f"{image.size}",
            "Mode": image.mode
        })
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        debug_data["Status"] = "Processing inputs..."
        
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        debug_data["Model Info"]["Input Shapes"] = {
            k: str(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        
        debug_data["Status"] = "Generating..."
        debug_data["Generation Info"]["GPU Memory Before"] = f"{torch.cuda.memory_allocated()/1024**2:.2f}MB"
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(256, max_tokens),
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        
        debug_data["Generation Info"]["GPU Memory After"] = f"{torch.cuda.memory_allocated()/1024**2:.2f}MB"
        debug_data["Status"] = "Generation complete!"
        
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text):].strip()

        history.append((user_prompt, generated_text))
        
    except Exception as e:
        debug_data["Status"] = f"Error: {str(e)}"
        history.append((user_prompt, f"Error: {str(e)}"))
        import traceback
        debug_data["Error Traceback"] = traceback.format_exc()
    
    return history, debug_data

def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        # Title only at the top
        gr.HTML(
        """
        <h1 style='text-align: center'>Clean-UI</h1>
        """)
        
        with gr.Row():
            # Left column
            with gr.Column(scale=1):
                # Image upload first
                image_input = gr.Image(
                    label="Image", 
                    type="pil", 
                    image_mode="RGB", 
                    height=272,
                    width=484,
                    sources=["upload", "webcam", "clipboard"]  # Enable upload, webcam, and clipboard
                )

                # Image dimensions second
                with gr.Row():
                    image_width = gr.Number(
                        label="Max Image Width",
                        value=MAX_IMAGE_WIDTH,
                        interactive=True,
                        minimum=224,
                        maximum=2048
                    )
                    image_height = gr.Number(
                        label="Max Image Height",
                        value=MAX_IMAGE_HEIGHT,
                        interactive=True,
                        minimum=224,
                        maximum=2048
                    )
                gr.HTML(
                """
                <p style='text-align: center'>Current Model: {}</p>
                """.format(model_name))
                # Model parameters third
                gr.Markdown("### Model Parameters")
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=2.0, value=TEMPERATURE, step=0.1)
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=TOP_K, step=1)
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=TOP_P, step=0.1)
                max_tokens = gr.Slider(
                    label="Max Tokens", minimum=50, maximum=MAX_OUTPUT_TOKENS, value=MAX_OUTPUT_TOKENS, step=50)

            # Right column
            with gr.Column(scale=2):
                # Chat interface first
                chat_history = gr.Chatbot(
                    label="Chat",
                    height=400
                )

                user_prompt = gr.Textbox(
                    show_label=False,
                    container=False,
                    placeholder="Enter your prompt", 
                    lines=2
                )

                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")
                
                # Processing info second
                process_info = gr.JSON(
                    label="Processing Information",
                    value={},
                )
                
                # Debug info last
                debug_info = gr.TextArea(
                    label="Debug Information",
                    value=f"Model device: {next(model.parameters()).device}\n"
                          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
                          f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB",
                    interactive=False,
                    lines=4
                )

        # Wire up the UI components
        generate_button.click(
            fn=describe_image_with_debug,
            inputs=[
                image_input, user_prompt, temperature, 
                top_k, top_p, max_tokens, chat_history,
                image_width, image_height
            ],
            outputs=[chat_history, process_info]
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