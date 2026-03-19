with open('app/v2/ingestion/pageindex_core/utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace PyPDF2 import with a no-op comment (pymupdf already imported on next line)
content = content.replace('import PyPDF2\n', '# PyPDF2 replaced by pymupdf (fitz) - already imported below\n')

# Replace all PyPDF2.PdfReader(x) calls with equivalent pymupdf.open(x) pattern
import re

def replace_pypdf2_reader(m):
    path_var = m.group(1)
    # Returns a context manager replacement
    return f'_FitzPDFReader({path_var})'

content = re.sub(r'PyPDF2\.PdfReader\(([^)]+)\)', replace_pypdf2_reader, content)

# Inject a thin PyPDF2 compat shim right after the pymupdf import line
shim = '''
class _FitzPDFReader:
    """Thin shim replacing PyPDF2.PdfReader with pymupdf/fitz for PageIndex core."""
    def __init__(self, path):
        self._doc = pymupdf.open(path)
        self.pages = [type("Page", (), {"extract_text": lambda s, i=i: self._doc[i].get_text("text")})() for i in range(len(self._doc))]
        self.num_pages = len(self._doc)

'''
content = content.replace('from io import BytesIO\n', 'from io import BytesIO\n' + shim, 1)

with open('app/v2/ingestion/pageindex_core/utils.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('utils.py patched OK')
