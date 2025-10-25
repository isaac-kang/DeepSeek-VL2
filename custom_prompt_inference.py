#!/usr/bin/env python3
"""
Custom Prompt Inference Script for DeepSeek-VL2
Processes images in a directory and outputs model predictions without ground truth
"""

import os
import glob
import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM
from pathlib import Path

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor


class CustomPromptInference:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny", 
                 device: str = "auto", prompt: str = None, max_new_tokens: int = 50):
        """
        Initialize Custom Prompt Inference
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cuda", "cpu")
            prompt: Custom prompt for inference
            max_new_tokens: Maximum number of tokens to generate (default: 50)
        """
        self.model_name = model_name
        self.device = device
        self.prompt = prompt if prompt else "What is the main text in the image? Output only the text."
        self.max_new_tokens = max_new_tokens
        
        print(f"Loading model: {model_name}")
        print(f"Using prompt: {self.prompt}")
        print(f"Max new tokens: {self.max_new_tokens}")
        
        # Load model and processor
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.vl_gpt = self.vl_gpt.cuda().eval()
        
        print("Model loaded successfully!")
        
        # Print GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("Using CPU")
    
    def predict_text(self, image_path: str) -> str:
        """
        Predict text from image using DeepSeek-VL2
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted text
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create conversation format for DeepSeek-VL2
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{self.prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Load PIL images
            pil_images = []
            for message in conversation:
                if "images" not in message:
                    continue
                for img_path in message["images"]:
                    pil_img = Image.open(img_path)
                    pil_img = pil_img.convert("RGB")
                    pil_images.append(pil_img)
            
            # Prepare inputs
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(self.vl_gpt.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.vl_gpt.generate(
                    **prepare_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode prediction
            answer = self.tokenizer.decode(
                outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), 
                skip_special_tokens=True
            )
            
            return answer
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""
    
    def process_directory(self, directory_path: str, output_file: str = None):
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to output file for results
        """
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
            image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return
        
        # Sort by filename
        image_files.sort()
        
        print(f"\nFound {len(image_files)} images in {directory_path}")
        print("="*60)
        
        results = []
        
        # Process each image
        for i, image_path in enumerate(image_files):
            image_filename = os.path.basename(image_path)
            image_id = f"{i+1:09d}"  # Generate image ID like 000000001
            
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_filename}")
            
            # Predict text
            predicted_text = self.predict_text(image_path)
            
            # Store result
            result = {
                'image_id': image_id,
                'image_filename': image_filename,
                'predicted_text': predicted_text
            }
            results.append(result)
            
            # Print result
            print(f"  Model Answer: {predicted_text}")
        
        # Save results to file if output_file is specified
        if output_file:
            self.save_results(results, output_file)
            print(f"\nResults saved to: {output_file}")
        
        return results
    
    def save_results(self, results: list, output_file: str):
        """
        Save results to text file
        
        Args:
            results: List of result dictionaries
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*100 + "\n")
            f.write("Custom Prompt Inference Results - DeepSeek-VL2\n")
            f.write("="*100 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompt: {self.prompt}\n")
            f.write("="*100 + "\n\n")
            
            # Results
            for result in results:
                f.write(f"Image ID:       {result['image_id']}\n")
                f.write(f"Image File:     {result['image_filename']}\n")
                f.write(f"Prompt:         {self.prompt}\n")
                f.write(f"Model Answer:   {result['predicted_text']}\n")
                f.write("\n\n")
            
            # Summary
            f.write("="*100 + "\n")
            f.write(f"Total Images Processed: {len(results)}\n")
            f.write("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Custom Prompt Inference with DeepSeek-VL2')
    parser.add_argument('--image_dir', type=str, default='error_images',
                       help='Path to directory containing images (default: error_images)')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-vl2-tiny',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--prompt', type=str, default='What is the main text in the image? Output only the text.',
                       help='Custom prompt for inference')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum number of tokens to generate (default: 50)')
    parser.add_argument('--output_file', type=str, default='custom_inference_results.txt',
                       help='Output file for results (default: custom_inference_results.txt)')
    
    args = parser.parse_args()
    
    # Print evaluation header
    print("Starting Custom Prompt Inference with DeepSeek-VL2")
    print(f"Image directory: {args.image_dir}")
    print(f"Prompt: {args.prompt}")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist!")
        return
    
    # Initialize inference
    inference = CustomPromptInference(
        model_name=args.model_name, 
        device=args.device, 
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens
    )
    
    # Process images
    results = inference.process_directory(args.image_dir, args.output_file)
    
    print("\n" + "="*60)
    print("Custom Prompt Inference completed!")
    print(f"Total images processed: {len(results) if results else 0}")
    print("="*60)


if __name__ == "__main__":
    main()

