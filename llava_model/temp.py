from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

# Path to your locally downloaded model
local_model_path = r"D:\Melbin\VLM-Examples\llava_model\llava-v1.6-mistral-7b-hf"  # Replace with your actual local path

# Load model and processor from local path
processor = LlavaNextProcessor.from_pretrained(local_model_path)
model = LlavaNextForConditionalGeneration.from_pretrained(
    local_model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
) 
model.to("cuda:0")

# Prepare image and text prompt
img = r"D:\Melbin\VLM-Examples\static\ex02.png"
image = Image.open(img)

# Define chat history and format with template
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "what is the area of room 207?"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

# Generate response
output = model.generate(**inputs, max_new_tokens=10000)
print(processor.decode(output[0], skip_special_tokens=True))