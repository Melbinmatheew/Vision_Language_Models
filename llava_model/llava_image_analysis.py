from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import os
import argparse

# === Configuration: set your model path and default instruction here ===
DEFAULT_MODEL_PATH = r"D:\Melbin\VLM-Examples\llava_model\llava-v1.6-mistral-7b-hf"
DEFAULT_INSTRUCTION = (
    "You are an expert at analyzing images. "
    "Provide detailed, accurate answers about the given image."
    # "If you don't know the answer, say 'I don't know'."
    # "If the image is not clear, say 'The image is not clear'."
    # "If the image is not related to the question, say 'The image is not related to the question'."
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLaVA Image Analysis')
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to the local LLaVA model (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the image for analysis'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Initial prompt/question about the image (optional)'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default=DEFAULT_INSTRUCTION,
        help='System instruction for the model'
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file {args.image_path} does not exist")

    # Load model
    print(f"Loading model from {args.model_path}...")
    processor = LlavaNextProcessor.from_pretrained(args.model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Load image
    print(f"Loading image from {args.image_path}...")
    image = Image.open(args.image_path).convert("RGB")
    print("Image loaded successfully!")

    # System instruction: wrap content as list of dicts
    system_message = {"role": "system", "content": [{"type": "text", "text": args.instruction}]}  

    # Single-shot prompt mode
    if args.prompt:
        print(f"\nProcessing prompt: {args.prompt}")
        conversation = [
            system_message,
            {"role": "user", "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image"},
            ]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50000)
        response = processor.decode(output[0], skip_special_tokens=True)
        print("\n=== Analysis Result ===")
        print(response)
        print("======================")
        return

    # Interactive mode
    print("\n===  Image Analysis Interactive Session ===")
    print("Type your prompts to analyze the image. Type 'exit' or 'quit' to end the session.\n")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() in ['exit', 'quit']:
            print("Ending session. Goodbye!")
            break

        conversation = [
            system_message,
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
            ]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        print("Processing your request...")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50000)
        response = processor.decode(output[0], skip_special_tokens=True)
        print("\n=== Analysis Result ===")
        print(response)
        print("======================")

        # save_option = input("\nDo you want to save this result? (y/n): ")
        # if save_option.lower() == 'y':
        #     filename = input("Enter filename (or press Enter for default 'llava_result.txt'): ")
        #     filename = filename or "llava_result.txt"
        #     with open(filename, "w") as f:
        #         f.write(response)
        #     print(f"Result saved to {filename}")


if __name__ == "__main__":
    main()
