from dataclasses import dataclass, field
import streamlit as st
import requests
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from typing import Any
import traceback
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider 


SEGMA_ACCESS_URL = os.getenv("SEGMA_ACCESS_URL", "http://backend:3040").rstrip("/")
DEFAULT_ACTION_DATASET_ID = os.getenv("ACTION_DATASET_ID", "{ACTION_DATASET_ID}")


def config_value(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return ""
    return value

st.set_page_config(
    page_title="Chat with Data",
    layout="wide"
)

# Capture token once from URL (if provided)
if "token" not in st.session_state:
    token_from_url = st.query_params.get("token", None)
    if token_from_url:
        st.session_state.token = token_from_url

token = st.text_input("API bearer token", value=st.session_state.get("token", ""))
st.session_state.token = token

if not token:
    st.info("Please add your API bearer token to continue.")
    st.stop()


def extract_action_datasets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("action_datasets")
            or payload.get("data")
            or payload.get("items")
            or payload.get("results")
            or []
        )
    else:
        items = []

    return [item for item in items if isinstance(item, dict) and item.get("id") is not None]


def action_dataset_label(dataset: dict[str, Any]) -> str:
    name = (
        dataset.get("name")
        or "Untitled dataset"
    )
    return f"{name} (ID: {dataset['id']})"


@st.cache_data(show_spinner=True)
def fetch_action_datasets(api_token: str) -> list[dict[str, Any]]:
    headers = {"Authorization": f"bearer {api_token}"}
    url = f"{SEGMA_ACCESS_URL}/api/v1/action_datasets"
    response = requests.get(url, headers=headers, verify=False, timeout=30)
    response.raise_for_status()
    return extract_action_datasets(response.json())


try:
    action_datasets = fetch_action_datasets(token)
except requests.HTTPError as exc:
    st.error(f"Could not load action datasets: {exc.response.status_code} {exc.response.reason}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load action datasets: {exc}")
    st.stop()

if not action_datasets:
    st.info("No action datasets are available.")
    st.stop()

default_action_dataset_id = config_value(DEFAULT_ACTION_DATASET_ID)
selected_action_dataset_id = st.session_state.get("selected_action_dataset_id", default_action_dataset_id)
dataset_ids = [str(dataset["id"]) for dataset in action_datasets]
selected_index = dataset_ids.index(selected_action_dataset_id) if selected_action_dataset_id in dataset_ids else 0

selected_action_dataset = st.selectbox(
    "Action dataset",
    action_datasets,
    index=selected_index,
    format_func=action_dataset_label,
)
new_action_dataset_id = str(selected_action_dataset["id"])
if st.session_state.get("selected_action_dataset_id") != new_action_dataset_id:
    st.session_state.selected_action_dataset_id = new_action_dataset_id
    st.session_state.analyst_outputs = {}
    st.session_state.analyst_visualizations = []
    st.session_state.messages = []
    st.session_state.pa_history = []

action_dataset_id = st.session_state.selected_action_dataset_id

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()


@st.cache_data(show_spinner=True)
def fetch_data(api_token: str, selected_action_dataset_id: str):
    headers = {"Authorization": f"bearer {api_token}"}
#   params = {"limit": 100}
    url = f"{SEGMA_ACCESS_URL}/api/v1/action_datasets/{selected_action_dataset_id}/stream"
#    url = f"http://192.168.66.25/api/v1/action_datasets/7/stream"
#   response = requests.get(url, params=params, headers=headers, stream=True, verify=False, timeout=30)
    response = requests.get(url, headers=headers, stream=True, verify=False, timeout=30)
    response.raise_for_status()

    csv_bytes = bytearray()
    for chunk in response.iter_content(chunk_size=2048):
        if chunk:
            csv_bytes.extend(chunk)

    csv_buffer = BytesIO(csv_bytes)
    try:
        return pd.read_csv(csv_buffer, encoding="utf-8")
    except UnicodeDecodeError:
        csv_buffer.seek(0)
        return pd.read_csv(csv_buffer, encoding="utf-8-sig")

df = fetch_data(token, action_dataset_id)


@dataclass
class AnalystAgentDeps:
    # The single, fixed dataset the agent is allowed to use
    dataset_df: pd.DataFrame

    # Storage for query results (like notebook Out[1], Out[2], ...)
    output: dict[str, pd.DataFrame] = field(default_factory=dict)
    visualizations: list[dict[str, Any]] = field(default_factory=list)

    def store(self, value: pd.DataFrame) -> str:
        ref = f"Out[{len(self.output) + 1}]"
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(
                f"Error: {ref} is not a valid variable reference. Check the previous outputs and try again."
            )
        return self.output[ref]

    def store_visualization(self, title: str, image_bytes: bytes, summary: str = "") -> str:
        ref = f"Viz[{len(self.visualizations) + 1}]"
        self.visualizations.append(
            {
                "ref": ref,
                "title": title,
                "image_bytes": image_bytes,
                "summary": summary,
            }
        )
        return ref

model = OpenAIChatModel(
        "gpt-3.5-turbo",  # you can swap to another OpenAI chat model name
        provider=OpenAIProvider(api_key=openai_api_key),
    )
analyst_agent = Agent(
    model,
    deps_type=AnalystAgentDeps,
    instructions=(
        "You are a data analyst. You can ONLY answer questions about the fixed dataset. "
        "Use DuckDB SQL via the provided tools when SQL is the best fit. "
        "Use the Python analysis tool when you need richer pandas analysis, custom calculations, or charts. "
        "If the user asks for a chart, plot, histogram, distribution, or visualization, you must use `run_python_analysis` "
        "to generate the figure and call `save_chart(...)`. "
        "Do not say a visualization was created unless the tool returned a `Viz[n]` reference. "
        "In DuckDB SQL, the table name for the fixed dataset is `dataset`. "
        "In DuckDB SQL, you must enclose column names that contain spaces in double quotes instead of using underscores to replace spaces."
    ),
)


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], sql: str) -> str:
    """Run DuckDB SQL on the ONE fixed dataset.

    The fixed dataset is available as a virtual table named `dataset`.
    """
    df = ctx.deps.dataset_df
    result = duckdb.query_df(df=df, virtual_table_name="dataset", sql_query=sql)

    # Store result as Out[n] so the LLM can refer to it later
    ref = ctx.deps.store(result.df())
    return f"Executed SQL on the fixed dataset, result is `{ref}`"


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """Display at most 5 rows of a previously stored result dataframe (Out[n])."""
    df = ctx.deps.get(name)
    return df.head().to_string()  # pyright: ignore[reportUnknownMemberType]


@analyst_agent.tool
def dataset_preview(ctx: RunContext[AnalystAgentDeps]) -> str:
    """Show a quick preview of the fixed dataset (first 5 rows + columns)."""
    df = ctx.deps.dataset_df
    cols = ", ".join(map(str, df.columns))
    preview = df.head().to_string()
    return f"Fixed dataset columns: {cols}\n\nFirst 5 rows:\n{preview}"


@analyst_agent.tool
def run_python_analysis(ctx: RunContext[AnalystAgentDeps], code: str) -> str:
    """Run Python analysis against the fixed dataset and optionally save charts.

    Available objects inside the Python execution environment:
    - `dataset`: a copy of the fixed pandas DataFrame
    - `outputs`: dict of previously stored query results keyed by Out[n]
    - `pd`: pandas
    - `plt`: matplotlib.pyplot
    - `store_dataframe(value)`: store a DataFrame result and return an Out[n] reference
    - `save_chart(title, fig=None, summary="")`: save a matplotlib figure and return a Viz[n] reference

    For any visualization request, create a matplotlib figure and call `save_chart(...)`.
    If the column is categorical, prefer a bar chart of value counts over a histogram.
    Set a `result` variable if you want the tool to include a final scalar, dict, or DataFrame summary.
    """

    visualization_start = len(ctx.deps.visualizations)

    def store_dataframe(value: Any) -> str:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("store_dataframe expects a pandas DataFrame")
        return ctx.deps.store(value)

    def save_chart(title: str, fig: Any = None, summary: str = "") -> str:
        chart = fig if fig is not None else plt.gcf()
        if chart is None:
            raise ValueError("No matplotlib figure is available to save")

        buffer = BytesIO()
        chart.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        ref = ctx.deps.store_visualization(title, buffer.getvalue(), summary)
        plt.close(chart)
        return ref

    exec_globals = {
        "__builtins__": __builtins__,
        "dataset": ctx.deps.dataset_df.copy(),
        "outputs": {key: value.copy() for key, value in ctx.deps.output.items()},
        "pd": pd,
        "plt": plt,
        "store_dataframe": store_dataframe,
        "save_chart": save_chart,
    }
    exec_locals: dict[str, Any] = {}
    stdout_buffer = StringIO()

    try:
        with redirect_stdout(stdout_buffer):  # type: ignore[arg-type]
            exec(code, exec_globals, exec_locals)
    except Exception as exc:
        plt.close("all")
        raise ModelRetry(
            "Python analysis failed:\n"
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc(limit=2)}"
        ) from exc

    tool_messages: list[str] = []
    stdout_text = stdout_buffer.getvalue().strip()
    if stdout_text:
        tool_messages.append(f"stdout:\n{stdout_text}")

    result = exec_locals.get("result", exec_globals.get("result"))
    if isinstance(result, pd.DataFrame):
        ref = ctx.deps.store(result)
        tool_messages.append(f"Stored Python result as `{ref}`")
    elif result is not None:
        tool_messages.append(f"Python result: {result}")

    new_visualization_refs = [
        visualization["ref"] for visualization in ctx.deps.visualizations[visualization_start:]
    ]
    if new_visualization_refs:
        tool_messages.append(
            f"Saved visualizations: {', '.join(new_visualization_refs)}"
        )

    if not tool_messages:
        tool_messages.append("Python analysis completed with no explicit result.")

    return "\n\n".join(tool_messages)


def render_visualizations(visualizations: list[dict[str, Any]]) -> None:
    for visualization in visualizations:
        caption = visualization["title"]
        if visualization["summary"]:
            caption = f"{caption} - {visualization['summary']}"
        st.image(visualization["image_bytes"], caption=f"{visualization['ref']}: {caption}")


if "analyst_outputs" not in st.session_state:
    st.session_state.analyst_outputs = {}

if "analyst_visualizations" not in st.session_state:
    st.session_state.analyst_visualizations = []

deps = AnalystAgentDeps(
    dataset_df=df,
    output=st.session_state.analyst_outputs,
    visualizations=st.session_state.analyst_visualizations,
)



# UI chat messages (for display)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Pydantic AI message history (for model context across runs)
# We'll store the "ModelMessage" objects produced by the agent.
if "pa_history" not in st.session_state:
    st.session_state.pa_history = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_visualizations(message.get("visualizations", []))

if prompt := st.chat_input("What is up?"):
    # Store + show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream agent output into Streamlit
    visualization_start = len(deps.visualizations)

    def stream_agent_text():        
        # run_stream_sync streams text chunks. :contentReference[oaicite:2]{index=2}
        run = analyst_agent.run_stream_sync(
            user_prompt=prompt,
            deps=deps,
            message_history=st.session_state.pa_history
        )

        try:
            for chunk in run.stream_text(delta=True):
                yield chunk
        finally:
            # After streaming finishes, persist new messages for next turn. :contentReference[oaicite:3]{index=3}
            # (run_stream_sync returns a StreamedRunResultSync which is not a context manager.)
            st.session_state.pa_history.extend(run.new_messages())


    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_agent_text)
        new_visualizations = deps.visualizations[visualization_start:]
        render_visualizations(new_visualizations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "visualizations": new_visualizations,
        }
    )
