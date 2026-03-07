import ast
import os
import sys

def get_imports(path):
    imports = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=file_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.level == 0:
                                imports.add(node.module.split('.')[0])
                except Exception as e:
                    pass
    return imports

std_libs = set(sys.stdlib_module_names)
app_imports = get_imports('d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app') | get_imports('d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/scripts')
third_party = sorted(list(app_imports - std_libs - {'app', 'scripts', '_frozen_importlib'}))

# Read requirements.txt
req_modules = set()
with open('d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        # Get base package name
        pkg = line.split('==')[0].split('>=')[0].split('<')[0].split('[')[0].strip().lower()
        # Handle some common mappings
        mappings = {
            'python-dotenv': 'dotenv',
            'python-docx': 'docx',
            'python-multipart': 'multipart',
            'pillow': 'PIL',
            'beautifulsoup4': 'bs4',
            'pyyaml': 'yaml',
            'coqui-tts': 'TTS',
            'sentence-transformers': 'sentence_transformers',
            'qdrant-client': 'qdrant_client',
            'langchain-core': 'langchain_core',
            'langchain-text-splitters': 'langchain_text_splitters',
            'prometheus-client': 'prometheus_client'
        }
        req_modules.add(mappings.get(pkg, pkg))

print("Used in code but possibly missing in requirements:")
for mod in third_party:
    if mod.lower() not in req_modules and mod not in req_modules:
        print(f" - {mod}")

print("\\nRequirements check done.")
