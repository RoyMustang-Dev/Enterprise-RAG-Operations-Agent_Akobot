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
                    print(f"Error parsing {file_path}: {e}")
    return imports

std_libs = set(sys.stdlib_module_names)
app_imports = get_imports('d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/app') | get_imports('d:/WorkSpace/Enterprise-RAG-Operations-Agent_POC/scripts')
third_party = app_imports - std_libs - {'app', 'scripts'} # exclude local packages

print("Third-party imports found in code:", sorted(list(third_party)))
