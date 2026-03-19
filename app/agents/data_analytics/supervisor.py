import logging
from typing import Dict, Any, List, TypedDict, Annotated, Optional
import re
import pandas as pd
from pydantic import BaseModel, Field
import os
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from app.agents.data_analytics.pandas_engine import DeterministicPandasEngine
from app.agents.data_analytics.tools.rag_wrapper import enterprise_rag_tool
from app.tools.emailer import send_email_via_composio
from app.integrations.composio_tool_router import ComposioToolRouterClient
from app.agents.data_analytics.tools.third_party import google_analytics_4_tool, salesforce_soql_tool, google_sheets_tool, stripe_financial_tool, power_bi_analytics_tool
from app.infra.model_registry import get_phase_model
from app.infra.llm_client import set_agent_context

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Pydantic Output Generation Schema
# -----------------------------------------------------------------------------
class KPI(BaseModel):
    title: str = Field(description="The primary metric name (e.g., 'Total Revenue').")
    value: str = Field(description="The exact quantitative value calculated.")
    trend: str = Field(description="A short descriptive narrative of the trend (e.g. 'Up 15% WoW').")
    direction: str = Field(default="flat", description="Direction of change: up|down|flat.")
    delta_percent: float = Field(default=0.0, description="Percent delta for the trend.")
    period: str = Field(default="", description="Period for the trend (e.g., day_over_day).")


class StatisticalTest(BaseModel):
    test: str = Field(description="Name of the statistical test executed.")
    result: str = Field(description="Human-readable result string.")
    p_value: Optional[float] = Field(default=None, description="p-value if available.")
    significant: Optional[bool] = Field(default=None, description="Significance flag if available.")


class ForecastingSummary(BaseModel):
    model_used: str = Field(default="", description="Forecast model name (xgboost|prophet|linear_regression).")
    horizon_days: int = Field(default=0, description="Forecast horizon in days.")
    confidence_bounds: str = Field(default="", description="Confidence bounds summary if available.")
    forecast_csv_url: str = Field(default="", description="CSV URL from forecast tool output.")


class AnalyticsMetadata(BaseModel):
    rows_processed: int = Field(default=0, description="Number of rows processed.")
    columns_used: List[str] = Field(default_factory=list, description="Columns used in analysis.")
    date_range: str = Field(default="", description="Date range of the dataset.")
    missingness_percent: float = Field(default=0.0, description="Percent missing values in dataset.")
    merge_note: str = Field(default="", description="Notes about dataset merge/concat behavior.")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Data lineage sources and row counts.")

class ExecutiveDashboard(BaseModel):
    summary_paragraph: str = Field(description="A 3-4 sentence comprehensive business executive summary of the document/analysis.")
    per_region_summaries: List[Dict[str, Any]] = Field(default_factory=list, description="Deterministic per-region summaries for future trends.")
    forecast_table: List[Dict[str, Any]] = Field(default_factory=list, description="Forecast table rows grouped by region/product/date.")
    forecast_table_preview: List[Dict[str, Any]] = Field(default_factory=list, description="Compact preview of the forecast table (first N rows).")
    forecast_table_truncated: bool = Field(default=False, description="True if forecast_table was truncated for payload size.")
    forecast_table_total_rows: int = Field(default=0, description="Total rows available in the forecast table before truncation.")
    forecast_table_preview_rows: int = Field(default=0, description="Number of rows included in the forecast_table_preview.")
    forecast_table_preview: List[Dict[str, Any]] = Field(default_factory=list, description="Compact preview of the forecast table (first N rows).")
    forecast_table_truncated: bool = Field(default=False, description="True if forecast_table was truncated for payload size.")
    forecast_table_total_rows: int = Field(default=0, description="Total rows available in the forecast table before truncation.")
    forecast_table_preview_rows: int = Field(default=0, description="Number of rows included in the forecast_table_preview.")
    kpi_cards: List[KPI] = Field(description="A rigid array of up to 4 core KPIs extracted from the data.")
    risk_alerts: List[Dict[str, Any]] = Field(description="Any critical anomalies in the data, missing dates, dropping trends, etc.")
    inferred_csv_payload: str = Field(
        default="", 
        description="A raw CSV formatted string (with headers separated by commas, entries separated by newlines) containing the explicit data points and underlying metrics deduced during the analysis."
    )
    csv_download_url: str = Field(
        default="",
        description="The physical URL endpoint where the user can click to download the generated CSV file."
    )
    suggested_filename: str = Field(
        default="analytics_export.csv",
        description="A short, URL-safe, lowercase string describing this data natively ending in .csv (e.g., 'marketing_roi_q3.csv')."
    )
    statistical_tests: List[StatisticalTest] = Field(default_factory=list, description="Deterministic statistical tests executed on the dataset.")
    forecasting: ForecastingSummary = Field(default_factory=ForecastingSummary, description="Forecasting summary if used.")
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata, description="Dataset metadata and processing stats.")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Top-performing segments for cohort analysis.")
    time_windows: Dict[str, Any] = Field(default_factory=dict, description="Time-window summary (last 7/30 days) if available.")
    governance_checks: List[Dict[str, Any]] = Field(default_factory=list, description="Deterministic governance and data quality checks.")
    causal_insights: Dict[str, Any] = Field(default_factory=dict, description="Lightweight causal proxy results if available.")
    scenario_simulation: Dict[str, Any] = Field(default_factory=dict, description="Deterministic what-if scenario simulation results.")
    drift_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Model drift monitoring alerts based on numeric distribution shifts.")
    xlsx_download_url: str = Field(default="", description="XLSX report download URL.")
    kpi_explanations: Dict[str, str] = Field(default_factory=dict, description="Deterministic explanations for KPI calculations.")
    forecast_explainability: Dict[str, Any] = Field(default_factory=dict, description="Explainability details for forecast/backtest.")

# -----------------------------------------------------------------------------
# 2. LangGraph State and Orchestrator
# -----------------------------------------------------------------------------
class AnalyticsState(TypedDict):
    session_id: str
    query: str
    df_schema: str
    persona: str
    messages: Annotated[list, add_messages]
    dashboard_json: Optional[Dict[str, Any]]
    deterministic_kpis: List[Dict[str, Any]]
    deterministic_risks: List[Dict[str, Any]]
    deterministic_tests: List[Dict[str, Any]]
    deterministic_metadata: Dict[str, Any]
    deterministic_segments: List[Dict[str, Any]]
    deterministic_time_windows: Dict[str, Any]
    deterministic_governance: List[Dict[str, Any]]
    deterministic_causal: Dict[str, Any]
    deterministic_scenario: Dict[str, Any]
    deterministic_summary: str
    deterministic_forecast_meta: Dict[str, Any]
    deterministic_forecast_table: List[Dict[str, Any]]
    deterministic_per_region_summaries: List[Dict[str, Any]]
    deterministic_drift: List[Dict[str, Any]]
    deterministic_sources: List[Dict[str, Any]]
    deterministic_kpi_explanations: Dict[str, str]
    deterministic_forecast_explain: Dict[str, Any]
    rewritten_query: str
    intent_payload: Dict[str, Any]
    early_exit: bool
    early_reason: str
    groq_rate_limited: bool

class DataAnalyticsSupervisor:
    """
    The LangGraph orchestrator specifically built for the Business Analyst Persona.
    It takes an array of CSV files, generates schema context, and deterministically computes mathematical insights.
    """
    def __init__(self, dataframes: List[pd.DataFrame], sources: Optional[List[Dict[str, Any]]] = None):
        # Merge multiple datasets when schema overlaps; otherwise fallback to first
        self.merge_note = ""
        self.sources = sources or []
        self.samples = dataframes
        self.primary_df = self._merge_dataframes(dataframes)
        from app.agents.data_analytics.tools.deterministic_kpi import auto_date_cutoff
        self.primary_df = auto_date_cutoff(self.primary_df)
        self.pandas_sandbox = DeterministicPandasEngine(self.primary_df)
        
        cfg = get_phase_model("hallucination_verifier")
        
        # Model routing config (prioritize ModelsLab for BA reasoning; Groq for smalltalk)
        self.toolcall_provider = os.getenv("BA_TOOLCALL_PROVIDER", "modelslab").lower()
        self.toolcall_model = os.getenv("BA_TOOLCALL_MODEL", os.getenv("ANALYTICS_MODEL", "qwen-qwen3.5-122b-a10b"))
        self.synthesis_provider = os.getenv("BA_SYNTH_PROVIDER", "modelslab").lower()
        self.synthesis_model = os.getenv("BA_SYNTH_MODEL", self.toolcall_model)
        self.smalltalk_provider = os.getenv("BA_SMALLTALK_PROVIDER", "groq").lower()
        self.smalltalk_model = os.getenv("BA_SMALLTALK_MODEL", "llama-3.1-8b-instant")
        self.toolcall_enabled = os.getenv("BA_TOOLCALL_ENABLED", "true").lower() == "true"
        self.toolcall_max_steps = int(os.getenv("BA_TOOLCALL_MAX_STEPS", "4"))

        # Groq wrapper for fallback tool-calling
        self.llm = ChatGroq(
            model=os.getenv("BA_GROQ_TOOLCALL_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY", "dummy"),
            temperature=0.0,
            max_retries=1,
            timeout=30
        )

        # Preferred analytics synthesis model for Modelslab (when key exists)
        self.modelslab_analytics_model = os.getenv("ANALYTICS_MODEL", "qwen-qwen3.5-122b-a10b")
        
        # Bind the specific Master Toolkit
        # NOTE: LangChain requires tools to be bound for parallel execution capabilities
        # Map Pandas code execution as a direct callable Tool for the LLM
        pandas_tool = StructuredTool.from_function(
            func=self.pandas_sandbox.execute_deterministic_math,
            name="execute_pandas_math",
            description="Use explicit Python Pandas code as string to perform math on the 'df' DataFrame. ALWAYS use aggregations (like .groupby(), .sum(), .head()) to extract insights. NEVER try to print the entire dataframe."
        )
        
        # We define dynamic memory-bound forecasting wrappers so the LLM doesn't have to stringify 10,000 rows.
        # The LLM evaluates a dataframe variable inside the sandbox, and passes that variable's EXACT string name here.
        def _prophet_wrapper(variable_name: str, periods: int = 30) -> str:
            from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_prophet
            try:
                target_df = self.pandas_sandbox.repl.locals.get(variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: The variable '{variable_name}' does not exist as a Pandas DataFrame in the active sandbox. Please use execute_pandas_math to create it first."
                # Pass the physical Pandas DF down to the backend tool.
                # DO NOT run .to_dict() inside this LangChain StructuredTool wrapper. 
                # Doing so forces Langfuse and the ReAct Message History to log the massive 2MB array as tool args!
                return time_series_forecast_prophet(historical_data=target_df, periods=periods)
            except Exception as e:
                return f"Prophet Pipeline Error: {e}"
                
        prophet_tool = StructuredTool.from_function(
            func=_prophet_wrapper,
            name="time_series_forecast_prophet",
            description="Use for baseline time-series forecasting. First run execute_pandas_math to create a Pandas dataframe variable (e.g. `df_grouped`) containing 'ds' (dates) and 'y' (metrics). Then pass that EXACT variable name as a string (e.g. 'df_grouped') to this tool's `variable_name` argument."
        )
        
        def _xgboost_wrapper(variable_name: str, periods: int = 30) -> str:
            from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_xgboost
            try:
                target_df = self.pandas_sandbox.repl.locals.get(variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: The variable '{variable_name}' does not exist as a Pandas DataFrame in the active sandbox. Please use execute_pandas_math to create it first."
                # Pass the physical Pandas DF directly down to prevent ToolNode JSON Truncations
                return time_series_forecast_xgboost(historical_data=target_df, periods=periods)
            except Exception as e:
                return f"XGBoost Pipeline Error: {e}"
                
        xgboost_tool = StructuredTool.from_function(
            func=_xgboost_wrapper,
            name="time_series_forecast_xgboost",
            description="Use exclusively for advanced multi-variable (Panel Data) forecasting (e.g. predicting across multiple regions and categories simultaneously without collapsing them into a flat line). First run execute_pandas_math to create a Pandas dataframe variable containing 'ds' (dates), 'y' (metrics), and 'unique_id' (string labels). Then pass that EXACT variable name as a string (e.g. 'df_project') to this tool's `variable_name` argument."
        )
        
        def _regression_wrapper(historical_variable_name: str, target_column: str, future_feature_values: List[Dict[str, Any]]) -> str:
            from app.agents.data_analytics.tools.predictive_tools import linear_regression_projection
            try:
                target_df = self.pandas_sandbox.repl.locals.get(historical_variable_name)
                if target_df is None or not isinstance(target_df, pd.DataFrame):
                    return f"Fatal Error: Variable '{historical_variable_name}' does not exist."
                # Pass directly down to avoid ToolNode tracing
                result = linear_regression_projection(features_data=target_df, target_column=target_column, future_feature_values=future_feature_values)
                return result
            except Exception as e:
                return f"Regression Pipeline Error: {e}"
                
        regression_tool = StructuredTool.from_function(
            func=_regression_wrapper,
            name="linear_regression_projection",
            description="Use for multi-variate continuous prediction. Run execute_pandas_math to create your feature dataframe first, then pass its variable name here."
        )

        def _email_wrapper(user_id: str, to: List[str], subject: str, body: str) -> Dict[str, Any]:
            return send_email_via_composio(user_id=user_id, to=to, subject=subject, body=body)

        email_tool = StructuredTool.from_function(
            func=_email_wrapper,
            name="send_email",
            description="Send an email via Composio. Provide user_id, list of recipients, subject, and body."
        )

        # Composio tool router wrappers for CRM and other toolkits
        def _composio_search(user_id: str, use_case: str) -> Dict[str, Any]:
            client = ComposioToolRouterClient()
            if not client.is_configured():
                return {"error": "Composio not configured"}
            session = client.create_session(user_id=user_id)
            if not session or not session.get("session_id"):
                return {"error": "Failed to create Composio session"}
            return client.search_tools(session["session_id"], use_case)

        def _composio_manage_connections(user_id: str, toolkit: str) -> Dict[str, Any]:
            client = ComposioToolRouterClient()
            if not client.is_configured():
                return {"error": "Composio not configured"}
            session = client.create_session(user_id=user_id)
            if not session or not session.get("session_id"):
                return {"error": "Failed to create Composio session"}
            return client.create_link(session["session_id"], toolkit)

        def _composio_execute(user_id: str, tool_slug: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            client = ComposioToolRouterClient()
            if not client.is_configured():
                return {"error": "Composio not configured"}
            session = client.create_session(user_id=user_id)
            if not session or not session.get("session_id"):
                return {"error": "Failed to create Composio session"}
            return client.execute_tool(session["session_id"], tool_slug, arguments)

        composio_search_tool = StructuredTool.from_function(
            func=_composio_search,
            name="composio_search_tools",
            description="Search Composio tools by use case. Returns tool suggestions and connection status."
        )
        composio_manage_tool = StructuredTool.from_function(
            func=_composio_manage_connections,
            name="composio_manage_connections",
            description="Generate a connect link for a toolkit (e.g., hubspot, salesforce)."
        )
        composio_execute_tool = StructuredTool.from_function(
            func=_composio_execute,
            name="composio_execute_tool",
            description="Execute a Composio tool by slug with arguments after connection."
        )
        
        self.tools = [
            enterprise_rag_tool, 
            prophet_tool, 
            xgboost_tool,
            regression_tool, 
            pandas_tool,
            google_analytics_4_tool,
            salesforce_soql_tool,
            google_sheets_tool,
            power_bi_analytics_tool,
            stripe_financial_tool,
            email_tool,
            composio_search_tool,
            composio_manage_tool,
            composio_execute_tool
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Pydantic Structural generator (Groq default)
        self.llm_structured = self.llm.with_structured_output(ExecutiveDashboard)
        
        self.graph = self._build_graph()

    def _merge_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            return pd.DataFrame()
        if len(dataframes) == 1:
            return dataframes[0]
        try:
            # Only concat if schemas overlap significantly
            base_cols = set(dataframes[0].columns)
            compatible = [df for df in dataframes if len(base_cols.intersection(df.columns)) >= max(3, int(0.5 * len(base_cols)))]
            if len(compatible) >= 2:
                self.merge_note = f"Concatenated {len(compatible)} datasets with overlapping schema."
                return pd.concat(compatible, ignore_index=True)
            self.merge_note = "Multiple datasets provided, but schemas diverged; using the first dataset."
            return dataframes[0]
        except Exception:
            self.merge_note = "Dataset merge failed; using the first dataset."
            return dataframes[0]

    def _build_graph(self) -> Any:
        workflow = StateGraph(AnalyticsState)
        
        workflow.add_node("smalltalk_gate", self.smalltalk_gate)
        workflow.add_node("prompt_rewriter", self.prompt_rewriter)
        workflow.add_node("prompt_guard", self.prompt_guard)
        workflow.add_node("detect_schema", self.detect_schema)
        workflow.add_node("reasoning_agent", self.reasoning_agent)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("generate_dashboard", self.generate_dashboard)
        workflow.add_node("forecast_meta", self.forecast_meta)
        workflow.add_node("generate_early_exit", self.generate_early_exit)
        
        workflow.add_edge(START, "smalltalk_gate")
        workflow.add_conditional_edges(
            "smalltalk_gate",
            lambda state: "early" if state.get("early_exit") else "continue",
            {"early": "generate_early_exit", "continue": "prompt_rewriter"}
        )
        workflow.add_conditional_edges(
            "prompt_rewriter",
            lambda state: "early" if state.get("early_exit") else "continue",
            {"early": "generate_early_exit", "continue": "prompt_guard"}
        )
        workflow.add_edge("prompt_guard", "detect_schema")
        workflow.add_edge("detect_schema", "reasoning_agent")
        
        # The AI loops: Reason -> Tool -> Reason -> Tool until it stops calling tools
        workflow.add_conditional_edges(
            "reasoning_agent",
            tools_condition,
            {"tools": "tools", "__end__": "generate_dashboard"}
        )
        workflow.add_edge("tools", "forecast_meta")
        workflow.add_edge("forecast_meta", "reasoning_agent")
        workflow.add_edge("generate_dashboard", END)
        workflow.add_edge("generate_early_exit", END)
        
        return workflow.compile()

    def forecast_meta(self, state: AnalyticsState) -> Dict:
        """
        Extract deterministic forecast metadata from tool outputs.
        Looks for forecast tool success strings that include model name and csv path.
        """
        meta = {}
        explain = {}
        try:
            # find last tool message
            last_msg = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    last_msg = msg
                    break
            if last_msg and isinstance(last_msg.content, str):
                content = last_msg.content.lower()
                if "xgboost" in content:
                    meta["model_used"] = "xgboost"
                elif "prophet" in content:
                    meta["model_used"] = "prophet"

                # extract csv path
                match = re.search(r"/api/v1/exports/[^\s']+", last_msg.content)
                if match:
                    meta["forecast_csv_url"] = match.group(0)

                # extract horizon days if mentioned
                horizon_match = re.search(r"Projecting\s+(\d+)\s+days", last_msg.content, re.IGNORECASE)
                if horizon_match:
                    meta["horizon_days"] = int(horizon_match.group(1))

                ci_match = re.search(r"CI=(?:\u00b1|\+/-)([0-9\.]+)", last_msg.content)
                if ci_match:
                    meta["confidence_bounds"] = f"+/-{ci_match.group(1)}"

                mae_match = re.search(r"Backtest MAE=([0-9]+(?:\.[0-9]+)?)", last_msg.content)
                if mae_match:
                    explain["mae"] = float(mae_match.group(1))
        except Exception:
            pass
        return {"deterministic_forecast_meta": meta, "deterministic_forecast_explain": explain}

    def _build_forecast_artifacts(self, state: AnalyticsState) -> Dict[str, Any]:
        """
        Build forecast_table + per_region_summaries deterministically from forecast CSV (if available).
        """
        try:
            from app.agents.data_analytics.tools.deterministic_kpi import (
                load_forecast_dataframe,
                build_forecast_table,
                compute_per_region_summaries,
            )

            forecast_url = (state.get("deterministic_forecast_meta") or {}).get("forecast_csv_url", "")
            if not forecast_url:
                return {"deterministic_forecast_table": [], "deterministic_per_region_summaries": []}

            forecast_df = load_forecast_dataframe(forecast_url)
            metric_label = "units_sold"
            intent_metrics = (state.get("intent_payload") or {}).get("metrics") or []
            if intent_metrics:
                metric_label = intent_metrics[0]
            ci_width = None
            bounds = (state.get("deterministic_forecast_meta") or {}).get("confidence_bounds")
            if isinstance(bounds, str) and bounds.strip():
                try:
                    ci_width = float(bounds.replace("+/-", "").replace("±", "").strip())
                except Exception:
                    ci_width = None
            table = build_forecast_table(forecast_df, metric_label=metric_label, ci_width=ci_width)
            summaries = compute_per_region_summaries(table)
            return {
                "deterministic_forecast_table": table,
                "deterministic_per_region_summaries": summaries
            }
        except Exception:
            return {"deterministic_forecast_table": [], "deterministic_per_region_summaries": []}

    def _forecast_table_limits(self) -> tuple[int, int]:
        """
        Returns (max_rows, preview_rows). max_rows <= 0 disables truncation.
        """
        def _parse_int(value: str, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        max_rows = _parse_int(os.getenv("FORECAST_TABLE_MAX_ROWS", "200"), 200)
        preview_rows = _parse_int(os.getenv("FORECAST_TABLE_PREVIEW_ROWS", "12"), 12)
        if max_rows < 0:
            max_rows = 0
        if preview_rows < 0:
            preview_rows = 0
        return max_rows, preview_rows

    def _build_executive_summary(self, state: AnalyticsState) -> str:
        """
        Build a fuller 3-4 sentence deterministic executive summary.
        """
        parts: List[str] = []
        meta = state.get("deterministic_metadata") or {}
        per_region = state.get("deterministic_per_region_summaries") or []
        forecast_meta = state.get("deterministic_forecast_meta") or {}

        rows = meta.get("rows_processed")
        date_range = meta.get("date_range")
        region_count = None
        product_count = None
        try:
            if not self.primary_df.empty:
                if "Region" in self.primary_df.columns:
                    region_count = int(self.primary_df["Region"].nunique())
                if "Product_Category" in self.primary_df.columns:
                    product_count = int(self.primary_df["Product_Category"].nunique())
        except Exception:
            region_count = None
            product_count = None

        coverage_bits = []
        if isinstance(rows, int) and rows > 0:
            coverage_bits.append(f"{rows} rows")
        if date_range:
            coverage_bits.append(f"date range {date_range}")
        if region_count:
            coverage_bits.append(f"{region_count} regions")
        if product_count:
            coverage_bits.append(f"{product_count} products")
        if coverage_bits:
            parts.append(f"Analyzed {', '.join(coverage_bits)}.")

        model_used = forecast_meta.get("model_used")
        horizon_days = forecast_meta.get("horizon_days")
        if model_used or horizon_days:
            if model_used and horizon_days:
                parts.append(f"Forecasting used {model_used} with a {horizon_days}-day horizon.")
            elif model_used:
                parts.append(f"Forecasting used {model_used} for near-term trend projection.")
            elif horizon_days:
                parts.append(f"Forecast horizon set to {horizon_days} days.")

        if per_region:
            total = len(per_region)
            down = sum(1 for r in per_region if r.get("overall_region_trend") == "down")
            up = sum(1 for r in per_region if r.get("overall_region_trend") == "up")
            flat = total - down - up
            parts.append(f"Overall regional trend: down in {down}/{total}, up in {up}/{total}, flat in {flat}/{total}.")

            region_bits = []
            for r in per_region:
                region = r.get("region", "Unknown")
                top_products = r.get("top_products", [])[:2]
                if top_products:
                    movers = ", ".join([f"{t.get('product')} ({t.get('direction')}, {t.get('forecast_delta_percent', 0):.2f}%)" for t in top_products])
                    region_bits.append(f"{region} - {movers}")
            if region_bits:
                parts.append(f"Top movers by region: {'; '.join(region_bits)}.")

        if not parts:
            fallback = state.get("deterministic_summary") or "Summary generated, but no dominant patterns were detected."
            parts.append(fallback)

        return " ".join(parts[:4]).strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts a JSON object from a string, if present.
        """
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            return None
        return None

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """
        Tool catalog for BA Agent model-orchestrated tool calling.
        """
        return [
            {
                "name": "execute_pandas_math",
                "description": "Run pandas code against the in-memory df (use groupby/agg).",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            },
            {
                "name": "time_series_forecast_xgboost",
                "description": "Panel data forecasting. Input a dataframe variable name with ds, y, unique_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"variable_name": {"type": "string"}, "periods": {"type": "integer"}},
                    "required": ["variable_name"]
                }
            },
            {
                "name": "time_series_forecast_prophet",
                "description": "Baseline time-series forecasting. Input a dataframe variable name with ds, y.",
                "parameters": {
                    "type": "object",
                    "properties": {"variable_name": {"type": "string"}, "periods": {"type": "integer"}},
                    "required": ["variable_name"]
                }
            },
            {
                "name": "linear_regression_projection",
                "description": "Multivariate regression projection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "historical_variable_name": {"type": "string"},
                        "target_column": {"type": "string"},
                        "future_feature_values": {"type": "array"}
                    },
                    "required": ["historical_variable_name", "target_column", "future_feature_values"]
                }
            },
            {
                "name": "compute_kpis",
                "description": "Compute deterministic KPIs from the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_risk_alerts",
                "description": "Compute risk alerts from the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_stat_tests",
                "description": "Compute statistical tests from the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_metadata",
                "description": "Compute metadata for the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_segments",
                "description": "Compute top segments from the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_governance_checks",
                "description": "Compute governance checks for the dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_causal_proxy",
                "description": "Compute causal proxy for dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_scenario_simulation",
                "description": "Compute scenario simulation for dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "compute_drift_alerts",
                "description": "Compute drift alerts for dataset.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "generate_xlsx_report",
                "description": "Generate XLSX report and return the download URL.",
                "parameters": {"type": "object", "properties": {}}
            }
        ]

    def _execute_tool_call(self, name: str, args: Dict[str, Any], state: AnalyticsState) -> str:
        """
        Execute a tool call and optionally update deterministic state.
        """
        from app.agents.data_analytics.tools.deterministic_kpi import (
            compute_basic_kpis,
            compute_risk_alerts,
            compute_auto_stat_tests,
            compute_statistical_tests,
            compute_metadata,
            compute_segments,
            compute_governance_checks,
            compute_causal_proxy,
            compute_scenario_simulation,
            compute_drift_alerts,
        )
        try:
            if name == "execute_pandas_math":
                code = args.get("code", "")
                return self.pandas_sandbox.execute_deterministic_math(code)
            if name == "time_series_forecast_xgboost":
                var = args.get("variable_name", "")
                periods = int(args.get("periods") or 30)
                target_df = self.pandas_sandbox.repl.locals.get(var)
                from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_xgboost
                return time_series_forecast_xgboost(historical_data=target_df, periods=periods)
            if name == "time_series_forecast_prophet":
                var = args.get("variable_name", "")
                periods = int(args.get("periods") or 30)
                target_df = self.pandas_sandbox.repl.locals.get(var)
                from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_prophet
                return time_series_forecast_prophet(historical_data=target_df, periods=periods)
            if name == "linear_regression_projection":
                var = args.get("historical_variable_name", "")
                target_col = args.get("target_column", "")
                future_vals = args.get("future_feature_values", [])
                target_df = self.pandas_sandbox.repl.locals.get(var)
                from app.agents.data_analytics.tools.predictive_tools import linear_regression_projection
                return linear_regression_projection(features_data=target_df, target_column=target_col, future_feature_values=future_vals)
            if name == "compute_kpis":
                state["deterministic_kpis"] = compute_basic_kpis(self.primary_df)
                return "OK"
            if name == "compute_risk_alerts":
                state["deterministic_risks"] = compute_risk_alerts(self.primary_df)
                return "OK"
            if name == "compute_stat_tests":
                state["deterministic_tests"] = compute_auto_stat_tests(self.primary_df) or compute_statistical_tests(self.primary_df)
                return "OK"
            if name == "compute_metadata":
                state["deterministic_metadata"] = compute_metadata(self.primary_df, merge_note=self.merge_note)
                return "OK"
            if name == "compute_segments":
                state["deterministic_segments"] = compute_segments(self.primary_df)
                return "OK"
            if name == "compute_governance_checks":
                state["deterministic_governance"] = compute_governance_checks(self.primary_df)
                return "OK"
            if name == "compute_causal_proxy":
                state["deterministic_causal"] = compute_causal_proxy(self.primary_df)
                return "OK"
            if name == "compute_scenario_simulation":
                state["deterministic_scenario"] = compute_scenario_simulation(self.primary_df)
                return "OK"
            if name == "compute_drift_alerts":
                state["deterministic_drift"] = compute_drift_alerts(self.primary_df)
                return "OK"
            if name == "generate_xlsx_report":
                url = self.save_xlsx_report(state.get("session_id", ""), ExecutiveDashboard(
                    summary_paragraph=state.get("deterministic_summary", ""),
                    per_region_summaries=state.get("deterministic_per_region_summaries", []),
                    forecast_table=state.get("deterministic_forecast_table", []),
                    kpi_cards=[KPI(**k) for k in state.get("deterministic_kpis", [])],
                    risk_alerts=state.get("deterministic_risks", []),
                    inferred_csv_payload="",
                    csv_download_url="",
                    suggested_filename="analytics_export.csv",
                    statistical_tests=[StatisticalTest(**t) for t in state.get("deterministic_tests", [])],
                    forecasting=ForecastingSummary(**state.get("deterministic_forecast_meta", {})) if state.get("deterministic_forecast_meta") else ForecastingSummary(),
                    metadata=AnalyticsMetadata(**state.get("deterministic_metadata", {})),
                    segments=state.get("deterministic_segments", []),
                    time_windows=state.get("deterministic_time_windows", {}),
                    governance_checks=state.get("deterministic_governance", []),
                    causal_insights=state.get("deterministic_causal", {}),
                    scenario_simulation=state.get("deterministic_scenario", {}),
                    drift_alerts=state.get("deterministic_drift", []),
                ), self.samples)
                return url or ""
        except Exception as e:
            return f"Tool error: {e}"
        return "Unknown tool"

    def _toolcall_loop(self, provider: str, model: str, state: AnalyticsState) -> str:
        """
        Provider-agnostic tool-calling loop (JSON-based).
        """
        tool_specs = self._tool_specs()
        system = (
            "You are a Business Analytics tool orchestrator. "
            "Return ONLY valid JSON of the form: "
            "{\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}],\"final\":\"optional summary\"}. "
            "If no tools are needed, return {\"tool_calls\":[],\"final\":\"...\"}. "
            "Do not include any extra text."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"QUERY: {state.get('rewritten_query') or state.get('query')}"},
            {"role": "user", "content": f"SCHEMA: {state.get('df_schema')}"},
            {"role": "user", "content": f"TOOLS: {json.dumps(tool_specs)}"},
        ]

        from app.infra.llm_client import run_chat_completion
        transcript = []
        for _ in range(self.toolcall_max_steps):
            result = run_chat_completion(
                provider=provider,
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=900
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            payload = self._extract_json(content) or {}
            tool_calls = payload.get("tool_calls", []) if isinstance(payload, dict) else []
            final_text = payload.get("final") if isinstance(payload, dict) else ""

            if not tool_calls:
                if final_text:
                    return final_text
                return "Tool orchestration complete."

            for call in tool_calls:
                name = call.get("name")
                args = call.get("arguments", {}) if isinstance(call.get("arguments", {}), dict) else {}
                output = self._execute_tool_call(name, args, state)
                transcript.append({"tool": name, "output": output})
                messages.append({"role": "assistant", "content": json.dumps(call)})
                messages.append({"role": "tool", "content": json.dumps({"name": name, "output": output})})

        return f"Tool orchestration reached max steps. Transcript: {json.dumps(transcript)[:1500]}"

    def _toolcall_with_fallbacks(self, state: AnalyticsState) -> Optional[str]:
        """
        Try tool-calling across providers in priority order: modelslab -> groq -> gemini.
        """
        if not self.toolcall_enabled:
            return None

        attempts = [
            (self.toolcall_provider, self.toolcall_model),
            ("groq", os.getenv("BA_GROQ_TOOLCALL_MODEL", "llama-3.3-70b-versatile")),
            ("gemini", os.getenv("BA_GEMINI_TOOLCALL_MODEL", "gemini-2.5-flash")),
        ]
        for provider, model in attempts:
            provider = (provider or "").lower()
            if provider == "modelslab" and not os.getenv("MODELSLAB_API_KEY"):
                continue
            if provider == "groq" and not os.getenv("GROQ_API_KEY"):
                continue
            if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
                continue
            try:
                return self._toolcall_loop(provider, model, state)
            except Exception as e:
                msg = str(e).lower()
                if provider == "groq" and ("rate limit" in msg or "too many requests" in msg or "429" in msg):
                    state["groq_rate_limited"] = True
                if "out of credits" in msg or "rate limit" in msg:
                    continue
                # try next provider on any failure
                continue
        return None
    def _forecast_table_limits(self) -> tuple[int, int]:
        """
        Returns (max_rows, preview_rows). max_rows <= 0 disables truncation.
        """
        def _parse_int(value: str, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        max_rows = _parse_int(os.getenv("FORECAST_TABLE_MAX_ROWS", "200"), 200)
        preview_rows = _parse_int(os.getenv("FORECAST_TABLE_PREVIEW_ROWS", "12"), 12)
        if max_rows < 0:
            max_rows = 0
        if preview_rows < 0:
            preview_rows = 0
        return max_rows, preview_rows

    def _build_executive_summary(self, state: AnalyticsState) -> str:
        """
        Build a fuller 3-4 sentence deterministic executive summary.
        """
        parts: List[str] = []
        meta = state.get("deterministic_metadata") or {}
        per_region = state.get("deterministic_per_region_summaries") or []
        forecast_meta = state.get("deterministic_forecast_meta") or {}

        # Sentence 1: Data coverage
        rows = meta.get("rows_processed")
        date_range = meta.get("date_range")
        region_count = None
        product_count = None
        try:
            if not self.primary_df.empty:
                if "Region" in self.primary_df.columns:
                    region_count = int(self.primary_df["Region"].nunique())
                if "Product_Category" in self.primary_df.columns:
                    product_count = int(self.primary_df["Product_Category"].nunique())
        except Exception:
            region_count = None
            product_count = None

        coverage_bits = []
        if isinstance(rows, int) and rows > 0:
            coverage_bits.append(f"{rows} rows")
        if date_range:
            coverage_bits.append(f"date range {date_range}")
        if region_count:
            coverage_bits.append(f"{region_count} regions")
        if product_count:
            coverage_bits.append(f"{product_count} products")
        if coverage_bits:
            parts.append(f"Analyzed {', '.join(coverage_bits)}.")

        # Sentence 2: Forecast setup
        model_used = forecast_meta.get("model_used")
        horizon_days = forecast_meta.get("horizon_days")
        if model_used or horizon_days:
            if model_used and horizon_days:
                parts.append(f"Forecasting used {model_used} with a {horizon_days}-day horizon.")
            elif model_used:
                parts.append(f"Forecasting used {model_used} for near-term trend projection.")
            elif horizon_days:
                parts.append(f"Forecast horizon set to {horizon_days} days.")

        # Sentence 3: Overall trend distribution
        if per_region:
            total = len(per_region)
            down = sum(1 for r in per_region if r.get("overall_region_trend") == "down")
            up = sum(1 for r in per_region if r.get("overall_region_trend") == "up")
            flat = total - down - up
            parts.append(f"Overall regional trend: down in {down}/{total}, up in {up}/{total}, flat in {flat}/{total}.")

            # Sentence 4: Per-region top movers (compact)
            region_bits = []
            for r in per_region:
                region = r.get("region", "Unknown")
                top_products = r.get("top_products", [])[:2]
                if top_products:
                    movers = ", ".join([f"{t.get('product')} ({t.get('direction')}, {t.get('forecast_delta_percent', 0):.2f}%)" for t in top_products])
                    region_bits.append(f"{region} - {movers}")
            if region_bits:
                parts.append(f"Top movers by region: {'; '.join(region_bits)}.")

        # Fallback if we couldn't build anything meaningful
        if not parts:
            fallback = state.get("deterministic_summary") or "Summary generated, but no dominant patterns were detected."
            parts.append(fallback)

        return " ".join(parts[:4]).strip()

    def _should_auto_forecast(self, query: str) -> bool:
        if not query:
            return False
        q = query.lower()
        return any(k in q for k in ["forecast", "future", "trend", "projection", "predict"])

    def _run_deterministic_forecast(self, state: AnalyticsState) -> Dict[str, Any]:
        try:
            from app.agents.data_analytics.tools.predictive_tools import time_series_forecast_xgboost
            from app.agents.data_analytics.tools.deterministic_kpi import (
                _find_date_column,
                _guess_group_columns,
                _guess_metric_columns,
                load_forecast_dataframe,
                build_forecast_table,
                compute_per_region_summaries,
            )

            df = self.primary_df.copy()
            date_col = _find_date_column(df)
            if not date_col:
                return {}

            if "Region" in df.columns and "Product_Category" in df.columns:
                group_cols = ["Region", "Product_Category"]
            else:
                group_cols = _guess_group_columns(df)
            if len(group_cols) < 1:
                return {}

            metric_col = None
            intent_metrics = (state.get("intent_payload") or {}).get("metrics") or []
            if intent_metrics:
                for m in intent_metrics:
                    for col in df.columns:
                        if m.replace("_", "").lower() == col.replace("_", "").lower():
                            metric_col = col
                            break
                    if metric_col:
                        break
            if not metric_col:
                guesses = _guess_metric_columns(df)
                metric_col = guesses[0] if guesses else None
            if not metric_col:
                return {}

            df_project = df.groupby(group_cols + [date_col], as_index=False)[metric_col].sum()
            df_project = df_project.rename(columns={date_col: "ds", metric_col: "y"})
            if len(group_cols) >= 2:
                df_project["unique_id"] = df_project[group_cols[0]].astype(str) + "_" + df_project[group_cols[1]].astype(str)
            else:
                df_project["unique_id"] = df_project[group_cols[0]].astype(str)

            result = time_series_forecast_xgboost(df_project, periods=30)
            # Parse forecast meta
            meta: Dict[str, Any] = {}
            explain: Dict[str, Any] = {}
            if isinstance(result, str):
                if "xgboost" in result.lower():
                    meta["model_used"] = "xgboost"
                match = re.search(r"/api/v1/exports/[^\s']+", result)
                if match:
                    meta["forecast_csv_url"] = match.group(0)
                horizon_match = re.search(r"Projecting\s+(\d+)\s+days", result, re.IGNORECASE)
                if horizon_match:
                    meta["horizon_days"] = int(horizon_match.group(1))
                ci_match = re.search(r"CI=(?:\u00b1|\+/-)([0-9\.]+)", result)
                if ci_match:
                    meta["confidence_bounds"] = f"+/-{ci_match.group(1)}"
                mae_match = re.search(r"Backtest MAE=([0-9]+(?:\.[0-9]+)?)", result)
                if mae_match:
                    explain["mae"] = float(mae_match.group(1))
            forecast_df = load_forecast_dataframe(meta.get("forecast_csv_url", ""))
            metric_label = metric_col.lower()
            ci_width = None
            bounds = meta.get("confidence_bounds")
            if isinstance(bounds, str) and bounds.strip():
                try:
                    ci_width = float(bounds.replace("+/-", "").replace("±", "").strip())
                except Exception:
                    ci_width = None
            table = build_forecast_table(forecast_df, metric_label=metric_label, ci_width=ci_width)
            summaries = compute_per_region_summaries(table)
            return {
                "deterministic_forecast_meta": meta,
                "deterministic_forecast_explain": explain,
                "deterministic_forecast_table": table,
                "deterministic_per_region_summaries": summaries,
            }
        except Exception as e:
            logger.error(f"[ANALYTICS] Deterministic forecast failed: {e}")
            return {}

    def _is_smalltalk(self, text: str) -> bool:
        greetings = [
            r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bhow are you\b", r"\bgood morning\b",
            r"\bgood evening\b", r"\bthanks\b", r"\bthank you\b", r"\bwho are you\b"
        ]
        for pattern in greetings:
            if re.search(pattern, text.lower()):
                return True
        return False

    def _smalltalk_response(self, query: str) -> str:
        """
        Smalltalk response using fast Groq model when available.
        """
        if not os.getenv("GROQ_API_KEY"):
            return "Hello! I'm your Business Analytics Agent. Upload a CSV or Excel file and tell me the analysis you want."
        try:
            from app.infra.llm_client import run_chat_completion
            result = run_chat_completion(
                provider=self.smalltalk_provider,
                model=self.smalltalk_model,
                messages=[
                    {"role": "system", "content": "You are a concise, friendly business analytics assistant."},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=120
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content or "Hello! I'm your Business Analytics Agent. Upload a CSV or Excel file and tell me the analysis you want."
        except Exception:
            return "Hello! I'm your Business Analytics Agent. Upload a CSV or Excel file and tell me the analysis you want."

    def _guard_violations(self, text: str) -> Optional[str]:
        # Deterministic guardrails: block prompt injection / system override attempts
        disallowed = [
            "ignore previous instructions",
            "system prompt",
            "reveal your prompt",
            "bypass safety",
            "developer message",
            "api key",
            "password"
        ]
        lowered = text.lower()
        for term in disallowed:
            if term in lowered:
                return f"Unsafe request detected: {term}"
        return None

    def _rewrite_intent(self, text: str) -> Dict[str, Any]:
        # Deterministic intent parsing (no LLM)
        intent = {
            "intent": "analysis",
            "metrics": [],
            "filters": {},
            "time_horizon_days": None
        }

        for metric in ["revenue", "units_sold", "marketing_spend", "profit", "margin"]:
            if metric.replace("_", " ") in text.lower() or metric in text.lower():
                intent["metrics"].append(metric)

        # Simple horizon extraction (e.g. "next 30 days")
        match = re.search(r"next\s+(\d+)\s+days", text.lower())
        if match:
            intent["time_horizon_days"] = int(match.group(1))

        # Region / product keyword extraction
        if "region" in text.lower():
            intent["filters"]["group_by"] = intent["filters"].get("group_by", []) + ["Region"]
        if "product" in text.lower():
            intent["filters"]["group_by"] = intent["filters"].get("group_by", []) + ["Product_Category"]

        return intent

    def smalltalk_gate(self, state: AnalyticsState) -> Dict:
        query = state["query"]
        if self._is_smalltalk(query):
            return {
                "early_exit": True,
                "early_reason": "smalltalk",
                "rewritten_query": query,
                "intent_payload": {}
            }
        return {"early_exit": False, "rewritten_query": query, "intent_payload": {}}

    def prompt_guard(self, state: AnalyticsState) -> Dict:
        query = state.get("rewritten_query") or state.get("query")
        violation = self._guard_violations(query)
        if violation:
            return {
                "early_exit": True,
                "early_reason": violation
            }
        return {"early_exit": False}

    def prompt_rewriter(self, state: AnalyticsState) -> Dict:
        intent = self._rewrite_intent(state["query"])
        rewritten = state["query"]
        if intent["metrics"]:
            rewritten = f"{state['query']}\n\n[INTENT] metrics={intent['metrics']} filters={intent.get('filters', {})} horizon={intent.get('time_horizon_days')}"
        return {
            "query": rewritten,
            "rewritten_query": rewritten,
            "intent_payload": intent,
            "messages": [HumanMessage(content=rewritten)]
        }

    def detect_schema(self, state: AnalyticsState) -> Dict:
        """Runs the deterministic df.describe() sandbox to extract a lightweight token map."""
        if self.primary_df.empty:
            return {
                "df_schema": "NO_DATASET_PROVIDED",
                "deterministic_kpis": [],
                "deterministic_risks": [],
                "deterministic_tests": [],
                "deterministic_segments": [],
                "deterministic_time_windows": {},
                "deterministic_governance": [],
                "deterministic_causal": {},
                "deterministic_scenario": {},
                "deterministic_metadata": {
                    "rows_processed": 0,
                    "columns_used": [],
                    "date_range": "",
                    "missingness_percent": 0.0,
                    "merge_note": self.merge_note
                }
            }
            
        logger.info("[ANALYTICS] Auto-detecting Schema from dataset...")
        schema_map = self.pandas_sandbox.generate_schema_context()
        from app.agents.data_analytics.tools.deterministic_kpi import (
            compute_basic_kpis,
            compute_risk_alerts,
            compute_statistical_tests,
            compute_auto_stat_tests,
            compute_metadata,
            compute_segments,
            compute_time_windows,
            compute_governance_checks,
            compute_causal_proxy,
            compute_scenario_simulation,
            compute_deterministic_summary,
            compute_drift_alerts,
            compute_kpi_explanations,
        )
        kpis = compute_basic_kpis(self.primary_df)
        auto_forecast = {}
        if self._should_auto_forecast(state.get("query", "")):
            auto_forecast = self._run_deterministic_forecast(state) or {}
        return {
            "df_schema": schema_map,
            "deterministic_kpis": kpis,
            "deterministic_risks": compute_risk_alerts(self.primary_df),
            "deterministic_tests": compute_auto_stat_tests(self.primary_df) or compute_statistical_tests(self.primary_df),
            "deterministic_metadata": compute_metadata(self.primary_df, merge_note=self.merge_note),
            "deterministic_segments": compute_segments(self.primary_df),
            "deterministic_time_windows": compute_time_windows(self.primary_df),
            "deterministic_governance": compute_governance_checks(self.primary_df),
            "deterministic_causal": compute_causal_proxy(self.primary_df),
            "deterministic_scenario": compute_scenario_simulation(self.primary_df),
            "deterministic_summary": compute_deterministic_summary(self.primary_df),
            "deterministic_forecast_meta": auto_forecast.get("deterministic_forecast_meta", {}),
            "deterministic_forecast_table": auto_forecast.get("deterministic_forecast_table", []),
            "deterministic_per_region_summaries": auto_forecast.get("deterministic_per_region_summaries", []),
            "deterministic_drift": compute_drift_alerts(self.primary_df),
            "deterministic_kpi_explanations": compute_kpi_explanations(kpis),
            "deterministic_forecast_explain": auto_forecast.get("deterministic_forecast_explain", {})
        }

    def reasoning_agent(self, state: AnalyticsState) -> Dict:
        """The core ReAct loop that decides whether to run Python Math, call RAG, or Forecast."""
        system_prompt = f"""You are a senior Business Analyst computing calculations perfectly. 
TARGET GOAL: {state.get('rewritten_query') or state['query']}
PERSONA OVERLAY: {state['persona']}

CRITICAL RULES:
1. ALWAYS use 'execute_pandas_math' to perform math on the 'df' DataFrame. 
2. Write plain python syntax. NO GUESSING.
3. If you need to forecast, run 'execute_pandas_math' FIRST and assign the resulting dataframe to a variable like 'df_project'. Do NOT print massive dicts. 
4. STRICT CHAIN OF THOUGHT: NEVER nest a tool call. You MUST call 'execute_pandas_math' first to calculate your table, wait for the observation, and ONLY THEN pass the exact string variable name of your table into the forecasting tools.
5. IF using XGBoost for multivariate panel forecasting (e.g. multiple regions), you MUST construct a string column named `unique_id` in your dataframe first (e.g., `df_project['unique_id'] = df_project['Region'] + '_' + df_project['Product_Category']`).
6. Do not stop until all KPI numbers are 100% computed.
7. If the user explicitly asks to send email, call `send_email` once with user_id, recipients, subject, and body.
8. For CRM-related requests (hubspot/salesforce/zendesk), call `composio_search_tools` or `composio_manage_connections` to provide a connect link.

SCHEMA MAP REVEALED:
{state['df_schema']}
"""
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        
        logger.info("[ANALYTICS] Triggering Tool-Calling Analytical Reasoner.")
        # Prefer tool orchestration across providers (modelslab -> groq -> gemini)
        final_text = self._toolcall_with_fallbacks(state)
        if final_text is not None:
            return {"messages": [AIMessage(content=final_text)]}

        if os.getenv("BA_ALLOW_GROQ_TOOLNODE", "false").lower() == "true":
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        return {"messages": [AIMessage(content="Deterministic analytics completed.")] }

    def save_csv_and_generate_url(self, raw_csv_string: str, suggested_name: str, session_id: str) -> str:
        """Physically saves the inferred CSV to disk and returns a download route."""
        try:
            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # Use the LLM's suggested name, but prefix with a short session hash to prevent collisions
            safe_id = str(session_id)[:8]
            clean_name = suggested_name.replace(" ", "_").lower()
            if not clean_name.endswith('.csv'):
                clean_name += '.csv'
                
            file_name = f"{safe_id}_{clean_name}"
            file_path = os.path.join(export_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_csv_string)
                
            return f"/api/v1/exports/{file_name}"
        except Exception as e:
            logger.error(f"[CSV EXPORT] Failed to write file: {e}")
            return ""

    def save_xlsx_report(self, session_id: str, dashboard: ExecutiveDashboard, samples: List[pd.DataFrame], forecast_table_override: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate an XLSX report with metadata, samples, insights, and charts."""
        try:
            import xlsxwriter

            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            file_name = f"{str(session_id)[:8]}_analytics_report.xlsx"
            file_path = os.path.join(export_dir, file_name)

            workbook = xlsxwriter.Workbook(file_path)
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9E1F2"})

            # Sheet 1: Metadata
            ws_meta = workbook.add_worksheet("Metadata")
            ws_meta.write(0, 0, "Field", header_fmt)
            ws_meta.write(0, 1, "Value", header_fmt)
            meta = dashboard.metadata.model_dump() if isinstance(dashboard.metadata, AnalyticsMetadata) else {}
            row = 1
            for k, v in meta.items():
                ws_meta.write(row, 0, k)
                ws_meta.write(row, 1, str(v))
                row += 1

            # Sheet 2: Raw Samples
            from app.agents.data_analytics.tools.deterministic_kpi import redact_pii
            ws_samples = workbook.add_worksheet("Raw Samples")
            row = 0
            for i, df in enumerate(samples):
                df = redact_pii(df)
                ws_samples.write(row, 0, f"Dataset {i+1}", header_fmt)
                row += 1
                if not df.empty:
                    for col_idx, col in enumerate(df.columns):
                        ws_samples.write(row, col_idx, col, header_fmt)
                    row += 1
                    for _, r in df.head(50).iterrows():
                        for col_idx, col in enumerate(df.columns):
                            ws_samples.write(row, col_idx, str(r[col]))
                        row += 1
                row += 2

            # Sheet 3: Insights
            ws_insights = workbook.add_worksheet("Insights")
            ws_insights.write(0, 0, "Executive Summary", header_fmt)
            ws_insights.write(1, 0, dashboard.summary_paragraph or "")

            row_offset = 3
            ws_insights.write(row_offset, 0, "KPI", header_fmt)
            ws_insights.write(row_offset, 1, "Value", header_fmt)
            row_offset += 1
            for kpi in dashboard.kpi_cards:
                ws_insights.write(row_offset, 0, kpi.title)
                ws_insights.write(row_offset, 1, kpi.value)
                row_offset += 1

            row_offset += 2
            # Per-region trend summary
            ws_insights.write(row_offset, 0, "Per-Region Trend Summary", header_fmt)
            row_offset += 1
            ws_insights.write(row_offset, 0, "Region", header_fmt)
            ws_insights.write(row_offset, 1, "Overall Trend", header_fmt)
            ws_insights.write(row_offset, 2, "Top Products", header_fmt)
            row_offset += 1
            for summary in dashboard.per_region_summaries:
                top_products = summary.get("top_products", [])
                top_text = "; ".join([f"{t.get('product')} ({t.get('direction')}, {t.get('forecast_delta_percent', 0):.2f}%)" for t in top_products])
                ws_insights.write(row_offset, 0, summary.get("region", ""))
                ws_insights.write(row_offset, 1, summary.get("overall_region_trend", "flat"))
                ws_insights.write(row_offset, 2, top_text)
                row_offset += 1

            # Forecast table
            row_offset += 2
            ws_insights.write(row_offset, 0, "Forecast Table", header_fmt)
            row_offset += 1
            headers = ["region", "product", "date", "metric", "forecast_value", "lower_ci", "upper_ci"]
            for col_idx, h in enumerate(headers):
                ws_insights.write(row_offset, col_idx, h, header_fmt)
            row_offset += 1
            forecast_rows = forecast_table_override if forecast_table_override is not None else dashboard.forecast_table
            for row in forecast_rows:
                ws_insights.write(row_offset, 0, str(row.get("region", "")))
                ws_insights.write(row_offset, 1, str(row.get("product", "")))
                ws_insights.write(row_offset, 2, str(row.get("date", "")))
                ws_insights.write(row_offset, 3, str(row.get("metric", "")))
                ws_insights.write_number(row_offset, 4, float(row.get("forecast_value", 0.0)))
                lower = row.get("lower_ci")
                upper = row.get("upper_ci")
                if lower is None:
                    ws_insights.write(row_offset, 5, "")
                else:
                    ws_insights.write_number(row_offset, 5, float(lower))
                if upper is None:
                    ws_insights.write(row_offset, 6, "")
                else:
                    ws_insights.write_number(row_offset, 6, float(upper))
                row_offset += 1

            # Sheet 4: Charts
            ws_charts = workbook.add_worksheet("Charts")
            if dashboard.kpi_cards:
                ws_charts.write(0, 0, "KPI", header_fmt)
                ws_charts.write(0, 1, "Value", header_fmt)
                numeric_rows = 0
                for idx, kpi in enumerate(dashboard.kpi_cards):
                    ws_charts.write(idx + 1, 0, kpi.title)
                    try:
                        val = float(str(kpi.value).replace(",", ""))
                        ws_charts.write_number(idx + 1, 1, val)
                        numeric_rows += 1
                    except Exception:
                        ws_charts.write(idx + 1, 1, None)

                if numeric_rows > 0:
                    chart = workbook.add_chart({"type": "column"})
                    chart.add_series({
                        "categories": ["Charts", 1, 0, len(dashboard.kpi_cards), 0],
                        "values": ["Charts", 1, 1, len(dashboard.kpi_cards), 1],
                        "name": "KPI Overview"
                    })
                    chart.set_title({"name": "KPI Overview"})
                    ws_charts.insert_chart("D2", chart)

            # Forecast trend charts (first 3 regions)
            if dashboard.forecast_table:
                start_row = 1
                ws_charts.write(start_row, 0, "Region", header_fmt)
                ws_charts.write(start_row, 1, "Date", header_fmt)
                ws_charts.write(start_row, 2, "Forecast", header_fmt)
                start_row += 1
                for row in dashboard.forecast_table[:500]:
                    ws_charts.write(start_row, 0, str(row.get("region", "")))
                    ws_charts.write(start_row, 1, str(row.get("date", "")))
                    ws_charts.write_number(start_row, 2, float(row.get("forecast_value", 0.0)))
                    start_row += 1

                regions = list({r.get("region") for r in dashboard.forecast_table if r.get("region")})[:3]
                chart_row = 2
                for region in regions:
                    chart = workbook.add_chart({"type": "line"})
                    chart.add_series({
                        "name": f"Forecast - {region}",
                        "categories": ["Charts", 2, 1, start_row - 1, 1],
                        "values": ["Charts", 2, 2, start_row - 1, 2],
                    })
                    chart.set_title({"name": f"Forecast Trend: {region}"})
                    ws_charts.insert_chart(f"D{chart_row}", chart)
                    chart_row += 15

            workbook.close()
            return f"/api/v1/exports/{file_name}"
        except Exception:
            return ""

    def generate_dashboard(self, state: AnalyticsState) -> Dict:
        """Forces the raw text from the Agent into a strict Pydantic UI Dashboard mapping."""
        logger.info("[ANALYTICS] Synthesizing final structured dashboard payload.")

        # Build deterministic forecast artifacts (table + per-region summaries)
        artifacts = self._build_forecast_artifacts(state)
        if artifacts.get("deterministic_forecast_table") is not None:
            state["deterministic_forecast_table"] = artifacts["deterministic_forecast_table"]
        if artifacts.get("deterministic_per_region_summaries") is not None:
            state["deterministic_per_region_summaries"] = artifacts["deterministic_per_region_summaries"]

        # Build an executive summary with stronger coverage
        state["deterministic_summary"] = self._build_executive_summary(state)
        
        synthesis_prompt = f"""Based on the current chat execution and analysis, extract the exact KPIs and business summaries.
Do not hallucinate data. Map the results into the output structured format securely.

CRITICALLY: 
If the Analysis Results explicitly gave you a CSV URL (e.g. `/api/v1/exports/...`), YOU MUST place that exact URL into the `csv_download_url` field, and leave `inferred_csv_payload` BLANK. Do not invent data!
If the analysis did NOT provide a file URL, you must construct the aggregated data into `inferred_csv_payload` (max 20 rows) and provide a `suggested_filename`.

DO NOT HALLUCINATE DATES OR METRICS. Rely entirely on the Analysis Results context!

RAW DATA SCHEMA:
{state['df_schema']}

ANALYSIS RESULTS:
{state['messages'][-1].content}
"""
        try:
            dashboard = None
            if self.synthesis_provider == "modelslab" and os.getenv("MODELSLAB_API_KEY"):
                try:
                    from app.infra.llm_client import run_chat_completion
                    timeout_sec = int(os.getenv("MODELSLAB_TIMEOUT_SEC", "30"))
                    result = run_chat_completion(
                        provider="modelslab",
                        model=self.synthesis_model or self.modelslab_analytics_model,
                        messages=[
                            {"role": "system", "content": "You are a strict JSON mapping architect. Return valid json only."},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=2000,
                        timeout=timeout_sec
                    )
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    parsed = json.loads(content)
                    dashboard = ExecutiveDashboard(**parsed)
                except Exception:
                    dashboard = None

            if dashboard is None and os.getenv("GEMINI_API_KEY"):
                try:
                    from app.infra.llm_client import run_chat_completion
                    result = run_chat_completion(
                        provider="gemini",
                        model=os.getenv("BA_GEMINI_SYNTH_MODEL", "gemini-2.5-flash"),
                        messages=[
                            {"role": "system", "content": "You are a strict JSON mapping architect. Return valid json only."},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=2000,
                        timeout=30
                    )
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    parsed = json.loads(content)
                    dashboard = ExecutiveDashboard(**parsed)
                except Exception:
                    dashboard = None

            if dashboard is None:
                allow_groq = os.getenv("BA_ALLOW_GROQ_SYNTH", "true").lower() == "true"
                if state.get("groq_rate_limited"):
                    allow_groq = False
                if allow_groq:
                    dashboard = self.llm_structured.invoke([
                        SystemMessage(content="You are a strict JSON mapping architect."),
                        HumanMessage(content=synthesis_prompt)
                    ])
                else:
                    dashboard = ExecutiveDashboard(
                        summary_paragraph=state.get("deterministic_summary") or "Deterministic analytics completed.",
                        per_region_summaries=state.get("deterministic_per_region_summaries", []),
                        forecast_table=state.get("deterministic_forecast_table", []),
                        kpi_cards=[KPI(**k) for k in state.get("deterministic_kpis", [])][:4],
                        risk_alerts=state.get("deterministic_risks", []),
                        inferred_csv_payload="",
                        csv_download_url=state.get("deterministic_forecast_meta", {}).get("forecast_csv_url", ""),
                        suggested_filename="analytics_export.csv",
                        statistical_tests=[StatisticalTest(**t) for t in state.get("deterministic_tests", [])],
                        forecasting=ForecastingSummary(**state.get("deterministic_forecast_meta", {})) if state.get("deterministic_forecast_meta") else ForecastingSummary(),
                        metadata=AnalyticsMetadata(**state.get("deterministic_metadata", {})),
                        segments=state.get("deterministic_segments", []),
                        time_windows=state.get("deterministic_time_windows", {}),
                        governance_checks=state.get("deterministic_governance", []),
                        causal_insights=state.get("deterministic_causal", {}),
                        scenario_simulation=state.get("deterministic_scenario", {}),
                        drift_alerts=state.get("deterministic_drift", []),
                    )

            # Inject deterministic results (authoritative)
            if not dashboard.kpi_cards and state.get("deterministic_kpis"):
                dashboard.kpi_cards = [KPI(**k) for k in state["deterministic_kpis"]]

            if not dashboard.risk_alerts and state.get("deterministic_risks"):
                dashboard.risk_alerts = state["deterministic_risks"]

            if state.get("deterministic_tests"):
                dashboard.statistical_tests = [StatisticalTest(**t) for t in state["deterministic_tests"]]

            if state.get("deterministic_metadata"):
                dashboard.metadata = AnalyticsMetadata(**state["deterministic_metadata"])
                if state.get("deterministic_sources"):
                    dashboard.metadata.sources = state["deterministic_sources"]

            if state.get("deterministic_segments"):
                dashboard.segments = state["deterministic_segments"]

            if state.get("deterministic_time_windows"):
                dashboard.time_windows = state["deterministic_time_windows"]

            if state.get("deterministic_governance"):
                dashboard.governance_checks = state["deterministic_governance"]

            if state.get("deterministic_causal"):
                dashboard.causal_insights = state["deterministic_causal"]

            if state.get("deterministic_scenario"):
                dashboard.scenario_simulation = state["deterministic_scenario"]

            if state.get("deterministic_drift"):
                dashboard.drift_alerts = state["deterministic_drift"]

            if state.get("deterministic_kpi_explanations"):
                dashboard.kpi_explanations = state["deterministic_kpi_explanations"]

            if state.get("deterministic_forecast_explain"):
                dashboard.forecast_explainability = state["deterministic_forecast_explain"]

            if state.get("deterministic_summary"):
                dashboard.summary_paragraph = state["deterministic_summary"]

            if state.get("deterministic_per_region_summaries"):
                dashboard.per_region_summaries = state["deterministic_per_region_summaries"]

            full_forecast_table = state.get("deterministic_forecast_table", [])
            if not full_forecast_table and dashboard.forecast_table:
                full_forecast_table = dashboard.forecast_table
            if full_forecast_table is not None:
                max_rows, preview_rows = self._forecast_table_limits()
                preview = full_forecast_table[:preview_rows] if preview_rows > 0 else []
                truncated = False
                table = full_forecast_table
                if max_rows > 0 and len(full_forecast_table) > max_rows:
                    table = full_forecast_table[:max_rows]
                    truncated = True
                dashboard.forecast_table = table
                dashboard.forecast_table_preview = preview
                dashboard.forecast_table_total_rows = len(full_forecast_table)
                dashboard.forecast_table_preview_rows = len(preview)
                dashboard.forecast_table_truncated = truncated

            if state.get("deterministic_forecast_meta"):
                try:
                    for k, v in state["deterministic_forecast_meta"].items():
                        if hasattr(dashboard.forecasting, k):
                            setattr(dashboard.forecasting, k, v)
                except Exception:
                    pass

            if dashboard.csv_download_url and dashboard.csv_download_url.startswith("/api/v1/exports/"):
                # The LLM captured the physical URL from a forensic/predictive tool. Do not overwrite!
                pass
            elif dashboard.inferred_csv_payload:
                dashboard.csv_download_url = self.save_csv_and_generate_url(
                    raw_csv_string=dashboard.inferred_csv_payload,
                    suggested_name=dashboard.suggested_filename,
                    session_id=state["session_id"]
                )

            # XLSX export
            xlsx_url = self.save_xlsx_report(state["session_id"], dashboard, self.samples, forecast_table_override=state.get("deterministic_forecast_table"))
            if xlsx_url:
                dashboard.xlsx_download_url = xlsx_url
            
            return {"dashboard_json": dashboard.model_dump()}
        except Exception as e:
            logger.error(f"[ANALYTICS] Dashboard Pydantic Format Failure: {e}")
            # Deterministic fallback without LLM schema
            fallback = ExecutiveDashboard(
                summary_paragraph=state.get("deterministic_summary") or "Analytics fallback executed due to schema error.",
                per_region_summaries=state.get("deterministic_per_region_summaries", []),
                forecast_table=state.get("deterministic_forecast_table", []),
                kpi_cards=[KPI(**k) for k in state.get("deterministic_kpis", [])][:4],
                risk_alerts=state.get("deterministic_risks", []),
                inferred_csv_payload="",
                csv_download_url=state.get("deterministic_forecast_meta", {}).get("forecast_csv_url", ""),
                suggested_filename="analytics_export.csv",
                statistical_tests=[StatisticalTest(**t) for t in state.get("deterministic_tests", [])],
                forecasting=ForecastingSummary(**state.get("deterministic_forecast_meta", {})) if state.get("deterministic_forecast_meta") else ForecastingSummary(),
                metadata=AnalyticsMetadata(**state.get("deterministic_metadata", {})),
                segments=state.get("deterministic_segments", []),
                time_windows=state.get("deterministic_time_windows", {}),
                governance_checks=state.get("deterministic_governance", []),
                causal_insights=state.get("deterministic_causal", {}),
                scenario_simulation=state.get("deterministic_scenario", {}),
                drift_alerts=state.get("deterministic_drift", []),
            )
            if state.get("deterministic_sources"):
                fallback.metadata.sources = state["deterministic_sources"]
            if state.get("deterministic_kpi_explanations"):
                fallback.kpi_explanations = state["deterministic_kpi_explanations"]
            if state.get("deterministic_forecast_explain"):
                fallback.forecast_explainability = state["deterministic_forecast_explain"]
            # Apply forecast table preview/cap for fallback
            full_forecast_table = state.get("deterministic_forecast_table", []) or fallback.forecast_table
            max_rows, preview_rows = self._forecast_table_limits()
            preview = full_forecast_table[:preview_rows] if preview_rows > 0 else []
            truncated = False
            table = full_forecast_table
            if max_rows > 0 and len(full_forecast_table) > max_rows:
                table = full_forecast_table[:max_rows]
                truncated = True
            fallback.forecast_table = table
            fallback.forecast_table_preview = preview
            fallback.forecast_table_total_rows = len(full_forecast_table)
            fallback.forecast_table_preview_rows = len(preview)
            fallback.forecast_table_truncated = truncated
            xlsx_url = self.save_xlsx_report(state["session_id"], fallback, self.samples, forecast_table_override=state.get("deterministic_forecast_table"))
            if xlsx_url:
                fallback.xlsx_download_url = xlsx_url
            return {"dashboard_json": fallback.model_dump()}

    def generate_early_exit(self, state: AnalyticsState) -> Dict:
        """Creates a deterministic response for smalltalk or blocked requests."""
        reason = state.get("early_reason") or "smalltalk"
        if reason == "smalltalk":
            summary = self._smalltalk_response(state.get("query", "Hello"))
        else:
            summary = f"Request blocked by Prompt Guard: {reason}"

        fallback = ExecutiveDashboard(
            summary_paragraph=summary,
            kpi_cards=[],
            risk_alerts=[],
            inferred_csv_payload="",
            csv_download_url="",
            suggested_filename="analytics_export.csv",
            statistical_tests=[],
            forecasting=ForecastingSummary(),
            metadata=AnalyticsMetadata()
        )
        return {"dashboard_json": fallback.model_dump()}

    async def run(self, query: str, persona: str, session_id: str, tenant_id: str | None = None) -> Dict[str, Any]:
        """Entrypoint for the API Route execution."""
        set_agent_context("ba")
        from app.infra.database import init_analytics_memory_db, fetch_analytics_memory, insert_analytics_memory
        init_analytics_memory_db(tenant_id)
        # Load prior memory for multi-turn context
        memory_rows = fetch_analytics_memory(session_id=session_id, limit=6, tenant_id=tenant_id)
        memory_messages = []
        for role, content, kpi_json in memory_rows:
            if role == "user":
                memory_messages.append(HumanMessage(content=content))
            else:
                memory_messages.append(AIMessage(content=content))
        initial_state = {
            "session_id": session_id,
            "query": query,
            "df_schema": "",
            "persona": persona,
            "messages": memory_messages + [HumanMessage(content=query)],
            "dashboard_json": None,
            "deterministic_kpis": [],
            "deterministic_risks": [],
            "deterministic_tests": [],
            "deterministic_metadata": {},
            "deterministic_segments": [],
            "deterministic_time_windows": {},
            "deterministic_governance": [],
            "deterministic_causal": {},
            "deterministic_scenario": {},
            "deterministic_summary": "",
            "deterministic_forecast_meta": {},
            "deterministic_forecast_table": [],
            "deterministic_per_region_summaries": [],
            "deterministic_drift": [],
            "deterministic_sources": self.sources,
            "deterministic_kpi_explanations": {},
            "deterministic_forecast_explain": {},
            "rewritten_query": "",
            "intent_payload": {},
            "early_exit": False,
            "early_reason": "",
            "groq_rate_limited": False
        }
        
        config = {}
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler(
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://us.langfuse.com"),
                    tags=["BUSINESS_ANALYST"]
                )
                config["callbacks"] = [langfuse_handler]
                logger.info(f"[LANGFUSE] Tracing enabled for Business Analyst session: {session_id}")
            except Exception as e:
                logger.warning(f"[LANGFUSE] Failed to initialize Analytics callback: {e}")

        try:
            import asyncio
            # Hard 120-second timeout on the entire analytical reasoning pipeline
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=config), 
                timeout=120.0
            )
            # Persist memory
            insert_analytics_memory(session_id, "user", query, "", tenant_id=tenant_id)
            if final_state.get("dashboard_json"):
                insert_analytics_memory(
                    session_id,
                    "assistant",
                    json.dumps(final_state["dashboard_json"])[:5000],
                    json.dumps(final_state["dashboard_json"].get("kpi_cards", []))[:2000],
                    tenant_id=tenant_id
                )
            return final_state["dashboard_json"]
            
        except asyncio.TimeoutError:
            logger.error(f"[ANALYTICS] Supervisor timed out after 120 seconds. Session: {session_id}")
            fallback = ExecutiveDashboard(
                summary_paragraph="Analytics Agent Engine Timeout. The data logic or LLM inference loops exceeded the standard 2-minute API limit.",
                kpi_cards=[KPI(title="System Status", value="TIMEOUT", trend="Execution Halted")],
                risk_alerts=[{"type": "timeout", "message": "The dataset complexity caused a timeout during reasoning.", "severity": "high"}],
                inferred_csv_payload="Error\nSystem Timeout",
                csv_download_url="",
                suggested_filename="timeout_error.csv"
            )
            return fallback.model_dump()
            
        except Exception as e:
            logger.error(f"[ANALYTICS] Fatal crash during Graph execution. Session: {session_id} Error: {e}")
            fallback = ExecutiveDashboard(
                summary_paragraph="Analytics System Breakdown. A fatal exception crashed the reasoning supervisor.",
                kpi_cards=[KPI(title="System Status", value="ERROR", trend="Exception Caught")],
                risk_alerts=[{"type": "system_error", "message": f"Pipeline fault: {str(e)[:150]}", "severity": "high"}],
                inferred_csv_payload="Error\nSystem Exception",
                csv_download_url="",
                suggested_filename="crash_error.csv"
            )
            return fallback.model_dump()
