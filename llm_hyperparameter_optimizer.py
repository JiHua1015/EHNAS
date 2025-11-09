 

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


class LLMHyperparameterOptimizer:
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def load_tips(self, tips_file: str) -> str:
         
        try:
            with open(tips_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def _chunk_text(self, text: str, chunk_size: int = 4000) -> List[str]:
         
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _analyze_chunk(self, chunk: str) -> str:
        
        prompt = f"""Analyze the following text about neural network architecture with a focus on hyperparameter design for CNN and Transformer modules. Summarize in English the core ideas for:

1. CNN module design (ResConv-based):
   - planes (channels): current candidates {16, 32, 64, 128}
   - stride: current candidates {1, 2}
   - cardinality (groups): current candidates {1, 2, 4}
   - base_width: current candidates {32, 64, 128}

2. Transformer module design (Vision_TransformerSuper):
   - num_heads: current candidates {3, 4, 6, 8}
   - mlp_ratio: current candidates {4, 6, 8}
   - qkv_dim: current candidates {192, 256, 384, 512}

3. Combination strategy:
   - Constraint: qkv_dim must be divisible by num_heads
   - Interactions across hyperparameters
   - Trade-offs between complexity and performance

Text:
{chunk}

Summary:"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.2,
            )
            summary = response.choices[0].message.content
            if not isinstance(summary, str):
                summary = str(summary)
            return summary.strip()
        except Exception as e:
            print(f"Error analyzing chunk: {e}")
            return ""

    def analyze_with_llm(self, tips_content: str) -> Dict[str, Any]:
        
        print("Starting LLM hyperparameter design analysis...")
        chunks = self._chunk_text(tips_content)
        print(f"Split content into {len(chunks)} chunks.")

        chunk_summaries_list = []
        with tqdm(total=len(chunks), desc="Analyze the text block") as pbar:
            for chunk in chunks:
                summary = self._analyze_chunk(chunk)
                if summary:
                    chunk_summaries_list.append(summary)
                pbar.update(1)

        # Deduplicate and combine summaries
        unique_summaries = list(set(chunk_summaries_list))
        combined_summary = "\n".join(unique_summaries)
        print("\nPerforming final analysis on combined summaries...")

        final_prompt = f"""
Based on the following summaries about CNN and Transformer hyperparameter design, produce a structured configuration for second-layer search space optimization in valid JSON with the fields:

Background:
- This is the second-layer optimization with a fixed macro skeleton: first 11 CNN blocks + last 5 ViT blocks
- Based on HyperparameterSearchSpace from search_space2.py
- Goal: narrow the search space while maintaining performance

Current search space:
- CNN (ResConv): hyperparameters = planes, stride, cardinality, base_width
- ViT (Vision_TransformerSuper): hyperparameters = window_size, num_heads, mlp_ratio, qkv_dim
- Constraint: qkv_dim % num_heads == 0

Output JSON schema:
1. "optimization_strategy": (String)
2. "cnn_hyperparameters": (Dict)
   - "planes": (List[Int]) candidates, current [16, 32, 64, 128]
   - "stride": (List[Int]) candidates, current [1, 2]
   - "cardinality": (List[Int]) candidates, current [1, 2, 4]
   - "base_width": (List[Int]) candidates, current [32, 64, 128]
   - "optimization_rationale": (String)
3. "transformer_hyperparameters": (Dict)
   - "num_heads": (List[Int]) candidates, current [3, 4, 6, 8]
   - "mlp_ratio": (List[Int]) candidates, current [4, 6, 8]
   - "qkv_dim": (List[Int]) candidates, current [192, 256, 384, 512]
   - "parameter_interactions": (String)
   - "constraint_rules": (String) describe qkv_dim divisibility rule
4. "search_space_reduction": (Dict)
   - "current_space_size": (String)
   - "reduction_recommendations": (List[String])
   - "expected_performance_retention": (String)
5. "implementation_guidance": (String)

Summaries:
{combined_summary}

Return only valid JSON.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': final_prompt}
                ],
                temperature=0.1,
            )
            
            result = response.choices[0].message.content
            if not isinstance(result, str):
                result = str(result)
            try:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    analysis = json.loads(json_str)
                    return analysis
                else:
                    print("No valid JSON format was found. Using fallback analysis...")
                    return self._get_fallback_analysis()
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return self._get_fallback_analysis()
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
         
        print("LLM analysis failed, using fallback analysis based on search_space2.py.")
        return {
            "optimization_strategy": "Based on a fixed network skeleton (first 11 CNN blocks + last 5 ViT blocks), refine hyperparameters of ResConv and Vision_TransformerSuper modules and narrow candidates to improve efficiency.",
            "cnn_hyperparameters": {
                "planes": [16, 32, 64, 128],
                "stride": [1, 2],
                "cardinality": [1, 2, 4],
                "base_width": [32, 64, 128],
                "optimization_rationale": "ResConv hyperparameters are chosen based on empirical performance: planes controls feature dimensions; stride affects spatial resolution; cardinality and base_width adjust model complexity."
            },
            "transformer_hyperparameters": {
                "window_size": [7, 14],
                "num_heads": [3, 4, 6, 8],
                "mlp_ratio": [4, 6, 8],
                "qkv_dim": [192, 256, 384, 512],
                "parameter_interactions": "window_size sets the locality of attention; num_heads controls attention parallelism; mlp_ratio impacts FFN capacity; qkv_dim is the core embedding size.",
                "constraint_rules": "qkv_dim must be divisible by num_heads to ensure each head has an integer dimension; when it is not, adjust to the nearest valid configuration."
            },
            "search_space_reduction": {
                "current_space_size": "Theoretical space: CNN 4×2×3×3=72 combos; ViT 2×4×3×4=96 combos; total ≈6,912 configurations.",
                "reduction_recommendations": [
                    "Prioritize medium-complexity combos with planes=[32,64] and base_width=[64,128]",
                    "Prefer balanced ViT configs with qkv_dim=[256,384] and num_heads=[4,6]",
                    "Avoid extreme pairs like minimal planes with maximal cardinality"
                ],
                "expected_performance_retention": "With reasonable hyperparameter filtering, expect >90% performance coverage of the original space."
            },
            "implementation_guidance": "Use HyperparameterSearchSpace.sample() with heuristics to prioritize stronger configurations and avoid invalid combinations."
        }
    
    def generate_hyperparameter_config(self, analysis: Dict[str, Any], output_file: str):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)
            print(f"LLM-generated hyperparameter configuration saved to {output_file}")
        except Exception as e:
            print(f"Error saving configuration file: {e}")

    def run_hyperparameter_optimization(self, tips_file: str, output_file: str):
        print("=== LLM-Assisted NAS Hyperparameter Optimization (Layer 2) ===")
        print()
        
        print("Step 1: Loading architecture insights...")
        tips_content = self.load_tips(tips_file)
        if not tips_content:
            print("✗ Failed to load tips file. Aborting.")
            return
        print(f"✓ Loaded {len(tips_content)} characters from tips file")
        
        print("\nStep 2: Analyzing hyperparameters with LLM...")
        analysis = self.analyze_with_llm(tips_content)
        
        if not analysis:
            print("✗ LLM analysis returned nothing. Aborting.")
            return

        print(f"  Optimization strategy: {analysis.get('optimization_strategy', 'N/A')}")
        print(f"  CNN candidates: {analysis.get('cnn_hyperparameters', {})}")
        print(f"  Transformer candidates: {analysis.get('transformer_hyperparameters', {})}")
        print(f"  Search space reduction: {analysis.get('search_space_reduction', 'N/A')}")
        
        print("\nStep 3: Generating hyperparameter config file...")
        self.generate_hyperparameter_config(analysis, output_file)
        
        print("\n=== Hyperparameter optimization complete ===")
        print("The config can guide layer-2 hyperparameter sampling and optimization.")


def main():
     
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") 
    BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "")
    
    if not API_KEY or not BASE_URL:
        print("Error: DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL environment variables must be set.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tips_file = os.path.join(script_dir, "tips.txt")
    output_file = os.path.join(script_dir, "llm_hyperparameter_config.json")
    
    optimizer = LLMHyperparameterOptimizer(api_key=API_KEY, base_url=BASE_URL)
    optimizer.run_hyperparameter_optimization(tips_file, output_file)


if __name__ == "__main__":
    main()
