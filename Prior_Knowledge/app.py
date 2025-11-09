from flask import Flask, Response, request, jsonify, send_from_directory
from urllib.parse import unquote
import os
import json
import pdfplumber
import openai
from datetime import datetime
import traceback
app = Flask(__name__, static_folder=None)
client = openai.Client(
    api_key="",
    base_url="",
)

analysis_prompt = """
You are a top-tier deep learning research scientist. Your task is to conduct a forensic, layer-by-layer analysis of an uploaded paper about CNN–Transformer hybrid architectures. Produce your report strictly following the structure below:
    
**1. Architectural Overview:**
  - **Core Idea:** What are the key innovations of the proposed hybrid architecture? How does it combine the strengths of CNNs and Transformers?
  - **End-to-End Flow:** Describe the complete processing pipeline from input to output.
    
**2. CNN Module Deep Dive:**
  - **Specific Variant:** Which CNN variant is used (e.g., ResNet, VGG, MobileNet)? If applicable, which exact version (e.g., ResNet-50)?
  - **Layer Configuration:** Detail convolution/pooling/activation configurations (kernel sizes, strides, padding, number of layers/filters).
  - **Role and Purpose:** What does the CNN part primarily handle in the hybrid (e.g., local feature extraction, downsampling, spatial encoding)?
    
**3. Transformer Module Deep Dive:**
  - **Specific Variant:** Is it a standard Transformer encoder/decoder or a variant (e.g., ViT, Swin Transformer)?
  - **Core Components:**
    - **Embedding:** How are CNN feature maps converted into a token sequence suitable for the Transformer? (e.g., details of patch embedding)
    - **Multi-Head Self-Attention:** How many heads are used, and what is the dimension per head?
    - **Feed-Forward Network:** What is the structure and activation used?
  - **Role and Purpose:** What does the Transformer part primarily handle (e.g., global dependency modeling, long-range relation capture, sequence feature integration)?
    
**4. Hybridization Strategy & Interface:**
  - **Connection Method:** How are CNN and Transformer connected (e.g., direct feeding of CNN outputs into the Transformer vs. a more sophisticated fusion module)?
  - **Feature Fusion:** If multiple connection points or fusion modules exist, describe their working principles in detail.
    
**5. Design Rationale & Insights:**
  - **Why This Design:** Why did the authors choose this specific combination of CNN and Transformer? What are the roles of early vs. later stages?
  - **Key Hyperparameters:** Which hyperparameters are critical to performance (e.g., token dimension, network depth, number of heads)?
  - **Your Expert Insights:** Based on your expertise, what are the potential strengths, weaknesses, or possible improvements of this architecture?
    
Ensure the analysis is objective, in-depth, and strictly grounded in the paper content. Output must be in Markdown.
"""


def parse_file_content(file_stream, file_name):
    file_ext = os.path.splitext(file_name)[1].lower()
    content = ""
    try:
        if file_ext == '.pdf':
            with pdfplumber.open(file_stream) as pdf:
                for page in pdf.pages:
                    content += page.extract_text() or ""
        elif file_ext == '.json':
            json_data = json.load(file_stream)
            content = json.dumps(json_data, ensure_ascii=False, indent=2)
        else:
            return f"Error: Unsupported file format '{file_ext}'."
    except Exception as e:
        return f"Error: Failed to parse file '{file_name}': {e}"
    return content




@app.route('/')
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, 'app.HTML')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_name = file.filename
    upload_path = os.path.join('uploads', file_name)
    file.save(upload_path)

    def generate_progress():
        try:
            with open(upload_path, 'rb') as saved_file:
                content = parse_file_content(saved_file, file_name)
            
            if content.startswith("Error:"):
                yield f"data: {json.dumps({'error': content})}\n\n"
                return
            yield f"data: {json.dumps({'progress': 5, 'status': 'Preparing analysis engine...'})}\n\n"

            stream = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': analysis_prompt},
                    {'role': 'user', 'content': f"Please analyze the following paper content:\n\n{content[:15000]}"}
                ],
                stream=True
            )

            result_text = ""
            progress = 5
            for chunk in stream:
                chunk_content = chunk.choices[0].delta.content or ""
                result_text += chunk_content
                if progress < 95:
                    progress += 0.1 
                yield f"data: {json.dumps({'progress': round(progress, 1), 'status': 'The model is generating the report...'})}\n\n"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_filename = f"{os.path.splitext(file_name)[0]}_report.txt"
            report_path = os.path.join('reports', report_filename)
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(result_text)

            report_header = f"--- REPORT START ---\nFile: {file.filename}\nAnalyzed at: {timestamp}\n\n"
            report_footer = "\n--- REPORT END ---\n\n"
            with open("tips.txt", "a", encoding='utf-8') as f:
                f.write(report_header + result_text + report_footer)

            report_data = {
                'original_filename': file.filename,
                'report_filename': report_filename,
                'timestamp': timestamp
            }
            yield f"data: {json.dumps({'progress': 100, 'status': 'Analysis completed', 'done': True, 'report': report_data})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': f'Error during analysis: {str(e)}'})}\n\n"

    return Response(generate_progress(), mimetype='text/event-stream')


def parse_reports_from_tips():
    reports = []
    if not os.path.exists('tips.txt'):
        return reports
    
    with open('tips.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    report_blocks = content.split('--- REPORT START ---')
    for block in report_blocks:
        if not block.strip():
            continue
        
        try:
            header_lines = block.strip().split('\n', 2)[:2]
            original_filename = ''
            timestamp = ''
            for line in header_lines:
                # Support both Chinese and English headers
                if line.startswith('文件:') or line.startswith('File:'):
                    original_filename = line.split(':', 1)[1].strip()
                elif line.startswith('分析时间:') or line.startswith('Analyzed at:'):
                    timestamp = line.split(':', 1)[1].strip()
            
            if original_filename and timestamp:
                base_name, _ = os.path.splitext(original_filename)
                report_filename = f"{base_name}_report.txt"
                
                reports.append({
                    'original_filename': original_filename,
                    'report_filename': report_filename,
                    'timestamp': timestamp
                })
        except Exception as e:
            print(f"Error parsing report block: {e}")
            traceback.print_exc()
            continue
            
    return sorted(reports, key=lambda x: x['timestamp'], reverse=True)

@app.route('/reports', methods=['GET'])
def get_reports():
    return jsonify(parse_reports_from_tips())

@app.route('/download/<path:filename>')
def download_report(filename):
    filename = unquote(filename)
    return send_from_directory('reports', filename, as_attachment=True)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    if not os.path.exists('tips.txt'):
        open('tips.txt', 'w').close()
    app.run(host='0.0.0.0', port=5000, debug=True)



