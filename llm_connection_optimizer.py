 

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


class LLMConnectionOptimizer:
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-plus"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def load_tips(self, tips_file: str) -> str:
         
        try:
            with open(tips_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading tips file: {e}")
            return ""

    def _chunk_text(self, text: str, chunk_size: int = 4000) -> List[str]:
         
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _analyze_chunk(self, chunk: str) -> str:
        
        prompt = f"""Analyze the following text about neural network architecture focusing on inter-block connection types and information flow. Summarize in English the key insights for:

1. Connection types (for third-layer connection search):
   - Sequential: standard layer-by-layer forward
     * Best scenarios
     * Advantages
     * Disadvantages

   - Residual: skip connections across blocks
     * Best scenarios
     * Advantages
     * Disadvantages

   - Parallel: multi-branch processing
     * Best scenarios
     * Advantages
     * Disadvantages

2. Position-dependent strategy:
   - Early layers (shallow)
   - Middle layers
   - Late layers (deep)

3. Cross-modal connections:
   - CNN->CNN
   - ViT->ViT
   - CNN->ViT

4. Combination strategies:
   - Effective combinations
   - Patterns to avoid
   - Alternating strategies

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
            print(f"Error during chunk analysis: {e}")
            return ""

    def analyze_with_llm(self, tips_content: str) -> Dict[str, Any]:
        
        print("Starting LLM connection design analysis...")
        chunks = self._chunk_text(tips_content)
        print(f"Split content into {len(chunks)} chunks.")

        chunk_summaries_list = []
        with tqdm(total=len(chunks), desc="Analyzing text blocks") as pbar:
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
Based on the following multiple summaries about connection design in neural architectures, produce a unified structured recommendation for third-layer connection search in valid JSON with fields:

- "optimization_strategy": (String)
- "connection_analysis": (Dict)
    - "sequential": (Dict)
        - "advantages": (List[String])
        - "disadvantages": (List[String])
        - "best_scenarios": (List[String])
    - "residual": (Dict)
    - "parallel": (Dict)
- "position_strategy": (Dict)
    - "early_layers": (Dict)
        - "preferred_connections": (List[String])
        - "reasoning": (String)
    - "middle_layers": (Dict)
    - "late_layers": (Dict)
- "cross_modal_connections": (Dict)
    - "cnn_to_cnn": (String)
    - "vit_to_vit": (String)
    - "cnn_to_vit": (String)
    - "vit_to_cnn": (String)
- "connection_patterns": (Dict)
    - "effective_sequences": (List[List[String]])
    - "patterns_to_avoid": (List[List[String]])
    - "alternating_strategies": (List[String])
- "search_space_reduction": (Dict)
    - "priority_connections": (List[String])
    - "reduction_factor": (String)
    - "performance_retention": (String)
- "implementation_guidance": (String)

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
            
            # 尝试解析JSON
            try:
                analysis = json.loads(result)
                print("✓ LLM successfully generated structured connection analysis")
                return analysis
            except json.JSONDecodeError as e:
                print(f"✗ JSON parse failed: {e}")
                print("Raw LLM output:")
                print(result)
                return self._get_fallback_analysis()
                
        except Exception as e:
            print(f"Error during final analysis: {e}")
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
         
        print("Using fallback connection analysis config...")
        return {
            "optimization_strategy": "Based on network depth and module type, apply a staged connection strategy: shallow layers prefer sequential for stability, middle layers introduce residual to enhance information flow, and deep layers selectively use parallel to increase expressive capacity.",
            "connection_analysis": {
                "sequential": {
                    "advantages": ["High computational efficiency", "Stable training", "Simple to implement", "Low memory footprint"],
                    "disadvantages": ["Potential vanishing gradients", "Information loss", "Limited expressive capacity"],
                    "best_scenarios": ["Shallow layers", "Resource-constrained settings", "Fast inference needed"]
                },
                "residual": {
                    "advantages": ["Mitigates vanishing gradients", "Preserves information flow", "Enables training deep networks", "Often improves performance"],
                    "disadvantages": ["Higher compute cost", "Requires dimension matching", "Potential overfitting"],
                    "best_scenarios": ["Deep networks", "Need to preserve gradient flow", "CNN-to-CNN connections"]
                },
                "parallel": {
                    "advantages": ["Increases model capacity", "Feature diversity", "Multi-scale processing", "Strong expressive power"],
                    "disadvantages": ["High compute cost", "Large memory usage", "Training complexity", "Potential overfitting"],
                    "best_scenarios": ["Late network stages", "Need diverse features", "Ample compute resources"]
                }
            },
            "position_strategy": {
                "early_layers": {
                    "preferred_connections": ["sequential", "residual"],
                    "reasoning": "Early layers extract basic features. Sequential ensures stability; residual can enhance information flow when needed."
                },
                "middle_layers": {
                    "preferred_connections": ["residual", "sequential"],
                    "reasoning": "Middle layers perform feature fusion and transformation. Residual helps retain important information; sequential maintains efficiency."
                },
                "late_layers": {
                    "preferred_connections": ["residual", "parallel"],
                    "reasoning": "Late layers handle high-level semantics. Residual preserves information integrity; parallel increases expressive capacity."
                }
            },
            "cross_modal_connections": {
                "cnn_to_cnn": "Prefer residual to maintain spatial feature continuity and gradient flow.",
                "vit_to_vit": "Recommend sequential or residual to preserve attention effectiveness.",
                "cnn_to_vit": "Use sequential for modality transition to ensure smooth conversion from spatial to sequence features.",
                "vit_to_cnn": "Generally not recommended; if needed, use sequential."
            },
            "connection_patterns": {
                "effective_sequences": [
                    ["sequential", "residual", "sequential"],
                    ["residual", "residual", "parallel"],
                    ["sequential", "sequential", "residual"]
                ],
                "patterns_to_avoid": [
                    ["parallel", "parallel", "parallel"],
                    ["sequential", "parallel", "sequential"],
                    ["parallel", "sequential", "parallel"]
                ],
                "alternating_strategies": [
                    "Alternate sequential and residual to balance stability and performance.",
                    "Use residual at critical positions, and sequential elsewhere.",
                    "Avoid consecutive parallel connections to prevent over-complication."
                ]
            },
            "search_space_reduction": {
                "priority_connections": ["sequential", "residual"],
                "reduction_factor": "With intelligent connection strategies, expect a 60–70% search space reduction.",
                "performance_retention": "Position- and modality-aware connections are expected to retain >90% performance coverage."
            },
            "implementation_guidance": "In ConnectionSearchSpace, implement position- and module-aware connection sampling, prioritize recommended combinations, and avoid unreasonable patterns."
        }
    
    def generate_connection_config(self, analysis: Dict[str, Any], output_file: str):
         
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)
            print(f"✓ LLM-generated connection configuration saved to {output_file}")
        except Exception as e:
            print(f"✗ Error saving configuration file: {e}")

    def run_connection_optimization(self, tips_file: str, output_file: str):
         
        print("=== LLM-Assisted NAS Connection Optimization (Layer 3) ===")
        print()
        
        print("Step 1: Loading architecture insights...")
        tips_content = self.load_tips(tips_file)
        if not tips_content:
            print("✗ Failed to load tips file. Aborting.")
            return
        print(f"✓ Loaded {len(tips_content)} characters from tips file")
        
        print("\nStep 2: Analyzing connection strategy with LLM...")
        analysis = self.analyze_with_llm(tips_content)
        
        if not analysis:
            print("✗ LLM analysis returned nothing. Aborting.")
            return

        print(f"  Optimization strategy: {analysis.get('optimization_strategy', 'N/A')}")
        print(f"  Priority connections: {analysis.get('search_space_reduction', {}).get('priority_connections', 'N/A')}")
        print(f"  Cross-modal strategy: {analysis.get('cross_modal_connections', {}).get('cnn_to_vit', 'N/A')}")
        print(f"  Search space reduction: {analysis.get('search_space_reduction', {}).get('reduction_factor', 'N/A')}")
        
        print("\nStep 3: Generating connection config file...")
        self.generate_connection_config(analysis, output_file)
        
        print("\n=== Connection optimization complete ===")
        print("The config can guide layer-3 connection search sampling and optimization.")
        print("It is recommended to use this with the best layer-2 architecture.")


def main():
     
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") 
    BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "")
    
    if not API_KEY or not BASE_URL:
        print("Error: DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL environment variables must be set.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tips_file = os.path.join(script_dir, "tips.txt")
    output_file = os.path.join(script_dir, "llm_connection_config.json")
    
    optimizer = LLMConnectionOptimizer(api_key=API_KEY, base_url=BASE_URL)
    optimizer.run_connection_optimization(tips_file, output_file)


if __name__ == "__main__":
    main()
