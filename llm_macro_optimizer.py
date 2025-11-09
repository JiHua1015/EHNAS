
import json
import os
import re
from typing import Any, Dict, List
from openai import OpenAI
from tqdm import tqdm


class LLMOptimizer:
    
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
         
        prompt = f"""Analyze the following text about neural network architecture design and summarize the core ideas and key design principles in English. Focus on (but not limited to):
 - How module types (e.g., CNN, Transformer, Mamba) are combined
 - Advantages of hybrid architectures
 - Suitable stages for specific modules (early, middle, late network stages)
 Keep the summary concise and highlight the most valuable design patterns.

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
            # Ensure summary is a string
            if not isinstance(summary, str):
                summary = str(summary)
            return summary.strip()
        except Exception as e:
            print(f"An error occurred during chunk analysis: {e}")
            return ""

    def analyze_with_llm(self, tips_content: str) -> Dict[str, Any]:
         
        print("Starting LLM analysis using 'chunk and summarize' strategy...")
        chunks = self._chunk_text(tips_content)
        print(f"Split content into {len(chunks)} chunks.")

        chunk_summaries_list = []
        with tqdm(total=len(chunks), desc="Analyzing chunks") as pbar:
            for chunk in chunks:
                summary = self._analyze_chunk(chunk)
                if summary:
                    chunk_summaries_list.append(summary)
                pbar.update(1)

        # Deduplicate summaries after collection
        unique_summaries = list(set(chunk_summaries_list))
        combined_summary = "\n".join(unique_summaries)
        print("\nPerforming final analysis on combined summaries...")

        final_prompt = f"""
Based on the following multiple summaries about neural network architecture design, produce a single, structured recommendation for search space optimization in valid JSON format with the fields below:
- "key_insights": (String) One or two sentences capturing the key takeaways.
- "structural_principle": (String) Describe the recommended overall pipeline, e.g., "use CNN for local feature extraction first, then Transformer for global context".
- "stage_preferences": (Dict) A dict with keys "early_stage", "mid_stage", and "late_stage". Each is a dict with:
    - "blocks": (List[String]) Recommended modules for the stage (e.g., ["cnn", "vit"]).
    - "comment": (String) Brief explanation for the recommendation.
- "block_proportions": (Dict) Recommended usage proportions for 'vit', 'mamba', and 'cnn' (floats summing to 1.0).
- "depth_range": (List[int]) Two integers: recommended min and max depth.
- "effective_sequences": (List[List[String]]) Examples of effective module sequences.
- "patterns_to_avoid": (List[List[String]]) Examples of sequences to avoid.

Summaries:
---
{combined_summary}
---

Output only valid JSON with the schema above.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            analysis_result = json.loads(response_text)
            print("✓ LLM analysis successful.")
            # Save analysis to a file for debugging
            with open("llm_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=4, ensure_ascii=False)
            return analysis_result
        except Exception as e:
            print(f"An error occurred during final LLM analysis: {e}")
            print("Attempting to parse JSON from raw response...")
            try:
                # Fallback to regex if strict JSON mode fails
                response_text = response.choices[0].message.content
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group())
                    print("✓ LLM analysis successful (via regex)." )
                    with open("llm_analysis.json", 'w', encoding='utf-8') as f:
                        json.dump(analysis_result, f, indent=4, ensure_ascii=False)
                    return analysis_result
            except Exception as final_e:
                print(f"Final parsing attempt failed: {final_e}")

            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
         
        print("LLM analysis failed. Using fallback analysis.")
        return {
            "key_insights": "The hybrid CNN-Transformer model is effective. CNN is used for extracting local features in the early layers, while Transformer/Mamba is used for handling global dependencies in the later layers。",
            "structural_principle": "Start with a CNN block for effective local feature extraction, and then transition to a ViT or Mamba block to capture global context and remote dependencies.",
            "stage_preferences": {
                "early_stage": {
                    "blocks": ["cnn"],
                    "comment": "Use CNN blocks in the early stages to efficiently capture local spatial features."
                },
                "mid_stage": {
                    "blocks": ["cnn", "vit"],
                    "comment": "Transition to ViT or Mamba blocks to capture global context and remote dependencies."
                },
                "late_stage": {
                    "blocks": ["vit", "mamba"],
                    "comment": "Focus on global context and long-range dependencies in the late stages."
                }
            },
            "depth_range": [10, 20],
            "block_proportions": {
                "cnn": 0.4,
                "vit": 0.6,
                "mamba": 0.0
            },
            "effective_sequences": [["cnn", "cnn", "vit"], ["cnn", "vit", "mamba"]],
            "patterns_to_avoid": [["vit", "cnn"], ["mamba", "cnn"]]
        }
    
    def generate_search_config(self, analysis: Dict[str, Any], output_file: str):
         
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)
            print(f"✓ LLM-generated search configuration saved to {output_file}")
        except Exception as e:
            print(f"✗ Error saving configuration file: {e}")

    def run_optimization(self, tips_file: str, output_file: str):
         
        print("=== LLM-Assisted Neural Architecture Search Space Optimization ===")
        print()
        
        print("Step 1: Loading architecture insights...")
        tips_content = self.load_tips(tips_file)
        if not tips_content:
            print("✗ Failed to load tips file. Aborting.")
            return
        print(f"✓ Loaded {len(tips_content)} characters from tips file")
        
        print("\nStep 2: Analyzing with LLM...")
        analysis = self.analyze_with_llm(tips_content)
        
        if not analysis:
            print("✗ LLM analysis returned nothing. Aborting.")
            return

        print(f"  Key Insights: {analysis.get('key_insights', 'N/A')}")
        print(f"  Structural Principle: {analysis.get('structural_principle', 'N/A')}")
        print(f"  Block Proportions: {analysis.get('block_proportions', 'N/A')}")
        
        print("\nStep 3: Generating search configuration file...")
        self.generate_search_config(analysis, output_file)
        
        print("\n=== Optimization Complete ===")


def main():
     
    API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") 
    BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "")
    
    if not API_KEY or not BASE_URL:
        print("Error: DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL must be set.")
        return


    script_dir = os.path.dirname(os.path.abspath(__file__))
    tips_file = os.path.join(script_dir, "tips.txt")
    output_file = os.path.join(script_dir, "llm_search_params.json")
    
    optimizer = LLMOptimizer(api_key=API_KEY, base_url=BASE_URL)
    optimizer.run_optimization(tips_file, output_file)


if __name__ == "__main__":
    main()
