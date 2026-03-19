import logging
from typing import Dict, Any
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool

logger = logging.getLogger(__name__)

class DeterministicPandasEngine:
    """
    A strictly sandboxed execution environment. 
    It prevents the LLM from hallucinating math by forcing it to write query logic as Python string, 
    which is then deterministically computed against the active DataFrame in memory using LangChain's REPL.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Initialize the REPL with the dataframe explicitly injected into its locals dictionary 
        # allowing the LLM to write syntax like `df['Revenue'].sum()` out of the box.
        self.repl = PythonAstREPLTool(locals={"df": self.df, "pd": pd})
        
    def generate_schema_context(self) -> str:
        """
        Extracts a lightweight string blueprint of the dataset (columns, missing values, types) 
        without passing the entire heavy payload to the LLM token window.
        """
        try:
            schema = f"Total Rows: {len(self.df)} | Columns: {list(self.df.columns)}\n"
            missing = self.df.isnull().sum().to_dict()
            dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            
            schema += f"Data Types: {dtypes}\n"
            schema += f"Missing Values Map: {missing}"
            return schema
        except Exception as e:
            logger.error(f"[SCHEMA GEN ERR] {e}")
            return "Unable to parse DataFrame schema."

    def execute_deterministic_math(self, python_code_string: str) -> str:
        """
        The LLM passes physical python code (e.g. `df['revenue'].mean()`).
        This sandbox runs it mathematically and returns the explicit integer/float/string result.
        """
        try:
            logger.info(f"[PANDAS REPL] Executing deterministic math:\n{python_code_string}")
            
            # Sanitize markdown formatting if the LLM hallucinated code blocks
            code = python_code_string.replace("```python", "").replace("```", "").strip()
            # Patch deprecated pandas resample alias
            code = code.replace(".resample('M')", ".resample('ME')")
            code = code.replace('.resample(\"M\")', '.resample(\"ME\")')
            
            # Execute the code against the injected DataFrame mathematically
            result = self.repl.invoke(code)
            
            # Prevent the LLM from blowing out its own context window if it accidentally prints the whole DF
            str_result = str(result)
            if len(str_result) > 2000:
                logger.warning("[PANDAS REPL] Output too massive. Truncating.")
                return str_result[:2000] + "\n... [DATA TRUNCATED: You printed too many rows. Use .head() or .groupby() to aggregate!]"
                
            return str_result
        except Exception as e:
            logger.error(f"[PANDAS REPL ERR] Attempted code: {python_code_string} | Error: {e}")
            return f"Error executing deterministic math: {e}. Check the exact column names in the schema and adjust your Pandas syntax."
