#!/usr/bin/env python3
"""
OCR Test Evaluation Script using DeepSeek-VL2
Uses example_custom_dataset with DeepSeek-VL2 models
Based on the Qwen3-VL pattern but adapted for DeepSeek-VL2
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import argparse

# Import DeepSeek-VL2 components
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class OCRTestEvaluator:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny", device: str = "auto", 
                 prompt: str = None, case_sensitive: bool = False, 
                 ignore_punctuation: bool = True, ignore_spaces: bool = True,
                 chunk_size: int = -1):
        """
        Initialize OCR Test Evaluator
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to use ("auto", "cpu", "cuda")
            prompt: Custom prompt for text recognition
            case_sensitive: Whether to preserve case in evaluation
            ignore_punctuation: Whether to ignore punctuation in evaluation
            ignore_spaces: Whether to ignore spaces in evaluation
            chunk_size: Chunk size for incremental prefilling (-1 to disable)
        """
        print(f"Loading DeepSeek-VL2 model: {model_name}")
        
        # Load processor and tokenizer
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load model
        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.vl_gpt = self.vl_gpt.cuda().eval()
        
        # Set evaluation parameters
        self.prompt = prompt if prompt else "What is the main text in the image? Output only the text."
        self.case_sensitive = case_sensitive
        self.ignore_punctuation = ignore_punctuation
        self.ignore_spaces = ignore_spaces
        self.model_name = model_name
        self.chunk_size = chunk_size
        
        print(f"Using prompt: {self.prompt}")
        print(f"Case sensitive: {self.case_sensitive}")
        print(f"Ignore punctuation: {self.ignore_punctuation}")
        print(f"Ignore spaces: {self.ignore_spaces}")
        print(f"Chunk size: {self.chunk_size}")
        print("Model loaded successfully!")
        
        # Print GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("Using CPU")
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load dataset from labels.json
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of dataset items
        """
        labels_path = os.path.join(dataset_path, "labels.json")
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {labels_path}")
        return data
    
    def preprocess_text(self, text: str, case_sensitive: bool = False, 
                       ignore_punctuation: bool = True, ignore_spaces: bool = True) -> str:
        """
        Preprocess text for evaluation
        
        Args:
            text: Raw text
            case_sensitive: Whether to preserve case
            ignore_punctuation: Whether to remove punctuation
            ignore_spaces: Whether to remove spaces
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        if not case_sensitive:
            text = text.upper()
        
        if ignore_punctuation:
            # Remove punctuation but keep alphanumeric characters
            text = re.sub(r'[^\w\s]', '', text)
        
        if ignore_spaces:
            # Remove all spaces
            text = re.sub(r'\s+', '', text)
        
        return text
    
    def predict_text(self, image_path: str) -> str:
        """
        Predict text from image using DeepSeek-VL2
        
        Args:
            image_path: Path to image file
            
        Returns:
            Predicted text
        """
        try:
            # Create conversation format for DeepSeek-VL2
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{self.prompt}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(self.vl_gpt.device)
            
            with torch.no_grad():
                if self.chunk_size == -1:
                    # Standard inference
                    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
                    past_key_values = None
                else:
                    # Incremental prefilling for memory efficiency
                    inputs_embeds, past_key_values = self.vl_gpt.incremental_prefilling(
                        input_ids=prepare_inputs.input_ids,
                        images=prepare_inputs.images,
                        images_seq_mask=prepare_inputs.images_seq_mask,
                        images_spatial_crop=prepare_inputs.images_spatial_crop,
                        attention_mask=prepare_inputs.attention_mask,
                        chunk_size=self.chunk_size
                    )
                
                # Generate prediction
                outputs = self.vl_gpt.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    past_key_values=past_key_values,
                    
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=50,
                    
                    do_sample=False,
                    use_cache=True,
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
    
    def calculate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate exact match accuracy
        
        Args:
            predictions: List of predicted texts
            ground_truths: List of ground truth texts
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            if pred == gt:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_dataset(self, dataset_path: str, max_samples: int = None) -> Dict:
        """
        Evaluate on the dataset
        
        Args:
            dataset_path: Path to dataset directory
            max_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Evaluation results dictionary
        """
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        predictions = []
        ground_truths = []
        samples = []  # Store original texts for display
        
        print(f"Evaluating {len(dataset)} samples...")
        
        for i, item in enumerate(dataset):
            image_filename = item['image_filename']
            image_path = os.path.join(dataset_path, image_filename)
            
            # Keep original ground truth for display
            original_gt = item['text']
            
            print(f"Processing {i+1}/{len(dataset)}: {image_filename}")
            
            # Predict text
            predicted_text = self.predict_text(image_path)
            
            # Preprocess both for comparison
            processed_gt = self.preprocess_text(original_gt, self.case_sensitive, 
                                               self.ignore_punctuation, self.ignore_spaces)
            processed_pred = self.preprocess_text(predicted_text, self.case_sensitive, 
                                                self.ignore_punctuation, self.ignore_spaces)
            
            predictions.append(processed_pred)
            ground_truths.append(processed_gt)
            
            # Store original texts for display
            samples.append({
                'image_filename': image_filename,
                'image_id': item.get('image_id', i + 1),
                'predicted_text': predicted_text,  # Original prediction
                'ground_truth': original_gt,  # Original ground truth
                'correct': processed_pred == processed_gt
            })
            
            # Print result with original values for display
            status = "✓" if processed_pred == processed_gt else "✗"
            print(f"  {status} GT: '{original_gt}' | Pred: '{predicted_text}'")
        
        # Calculate metrics
        accuracy = self.calculate_accuracy(predictions, ground_truths)
        
        # Count correct predictions
        correct_count = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        
        results = {
            'total_samples': len(dataset),
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'samples': samples
        }
        
        return results
    
    def print_results(self, results: Dict):
        """
        Print evaluation results
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "="*50)
        print("DEEPSEEK-VL2 OCR TEST EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct predictions: {results['correct_predictions']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*50)
    
    def save_detailed_results(self, results: Dict, output_file: str):
        """
        Save detailed results to text file
        
        Args:
            results: Results dictionary from evaluate_dataset
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*100 + "\n")
            f.write("DeepSeek-VL2 OCR Test Evaluation Results\n")
            f.write("="*100 + "\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompt: {self.prompt}\n")
            f.write(f"Matching - Case-sensitive: {self.case_sensitive}, Ignore punct: {self.ignore_punctuation}, Ignore space: {self.ignore_spaces}\n")
            f.write(f"Chunk size: {self.chunk_size}\n")
            f.write("="*100 + "\n\n")
            
            # Sample results
            for i, sample in enumerate(results['samples']):
                f.write(f"Sample {i+1}/{results['total_samples']}\n")
                f.write("-"*100 + "\n")
                f.write(f"Image:          {sample['image_filename']}\n")
                f.write(f"Image ID:       {sample['image_id']}\n")
                f.write(f"Prompt:         {self.prompt}\n")
                f.write(f"Model Answer:   {sample['predicted_text']}\n")
                f.write(f"Ground Truth:   {sample['ground_truth']}\n")
                f.write(f"Correct:        {'✓' if sample['correct'] else '✗'}\n")
                f.write("\n\n")
            
            # Summary
            f.write("="*100 + "\n")
            f.write(f"Dataset Complete!\n")
            f.write(f"Accuracy: {results['correct_predictions']}/{results['total_samples']} = {results['accuracy']*100:.2f}%\n")
            f.write("="*100 + "\n")
        
        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='OCR Test Evaluation with DeepSeek-VL2')
    parser.add_argument('--dataset_path', type=str, default='example_custom_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-vl2-tiny',
                       help='HuggingFace model name or local path')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--prompt', type=str, default="What is the main text in the image? Output only the text.",
                       help='Custom prompt for text recognition')
    parser.add_argument('--case-sensitive', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable case-sensitive evaluation (default: False)')
    parser.add_argument('--ignore-punctuation', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore punctuation in evaluation (default: True)')
    parser.add_argument('--ignore-spaces', type=lambda x: x.lower() == 'true', default=True,
                       help='Ignore spaces in evaluation (default: True)')
    parser.add_argument('--chunk_size', type=int, default=-1,
                       help='Chunk size for incremental prefilling (-1 to disable)')
    
    args = parser.parse_args()
    
    # Print evaluation header
    print("Starting OCR Test Evaluation with DeepSeek-VL2")
    print(f"Model: {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist!")
        return
    
    # Initialize evaluator
    evaluator = OCRTestEvaluator(
        model_name=args.model_name, 
        device=args.device, 
        prompt=args.prompt,
        case_sensitive=args.case_sensitive,
        ignore_punctuation=args.ignore_punctuation,
        ignore_spaces=args.ignore_spaces,
        chunk_size=args.chunk_size
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset_path, args.max_samples)
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results to text file
    model_name_clean = args.model_name.replace('/', '_').replace('-', '_')
    output_file = f"deepseek_vl2_ocr_results_{model_name_clean}.txt"
    evaluator.save_detailed_results(results, output_file)


if __name__ == "__main__":
    main()
