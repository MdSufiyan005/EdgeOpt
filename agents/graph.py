# graph.py
from typing import List, TypedDict, Annotated
import operator
from dotenv import load_dotenv
import requests

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from prompts import Planner_Prompt, Execution_Prompt, Summarizer_Prompt
from tools import convert_hf_to_gguf, quantize_gguf, upload_model_hf

load_dotenv()
BRIDGE_URL = "http://localhost:8080"
# State Definition

class AgentState(TypedDict):
    messages:       Annotated[List[BaseMessage], operator.add]  # full message history
    planner_output: str        # latest planner decision
    execution_output: str      # latest execution result
    summarizer_output: str     # latest summary
    iteration:      int        # current loop count
    best_quant:     str        # best strategy found so far
    final_result:   str        # populated at END

MAX_ITERATIONS = 4

# LLM Setup
llm = ChatGroq(model="llama-3.3-70b-versatile")   # stronger model for planning
llm_with_tools = llm.bind_tools(
    [convert_hf_to_gguf, quantize_gguf, upload_model_hf]
)

# Node Definitions
def planner_node(state: AgentState) -> AgentState:
    """
    Analyzes model + device specs and decides quantization strategy.
    On iteration > 0, it receives the summarizer's feedback and adapts.
    """
    messages = [SystemMessage(content=Planner_Prompt)] + state["messages"]

    # Inject summarizer feedback into context on subsequent iterations
    if state["iteration"] > 0 and state["summarizer_output"]:
        messages.append(
            HumanMessage(content=f"[Summarizer Feedback]\n{state['summarizer_output']}")
        )

    response = llm.invoke(messages)

    return {
        "messages":       [response],
        "planner_output": response.content,
        "iteration":      state["iteration"],   # unchanged here
    }


def _extract_quant_type(planner_output: str) -> str:
    for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
                  "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K"]:
        if quant in planner_output:
            return quant
    return "Q4_K_M"


def _extract_model_name(messages: list) -> str:
    """Pull model name from the initial HumanMessage."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content
            # Handles both "Model: meta-llama/Llama-3.2-1B" and bare names
            for line in content.splitlines():
                if line.lower().startswith("model:"):
                    return line.split(":", 1)[1].strip()
    return "unknown-model"


def execution_node(state: AgentState) -> AgentState:
    """
    Dynamic execution — inspects the model and decides steps at runtime:

    Case A: Model is already GGUF     → skip conversion, go straight to quantize
    Case B: Model is HF directory     → convert → quantize
    Case C: Model already quantized   → skip both, go straight to upload
    Case D: Model not found anywhere  → fail early with clear message
    """
    from langchain_core.messages import ToolMessage
    import json

    quant_type   = _extract_quant_type(state["planner_output"])
    model_input  = _extract_model_name(state["messages"])
    hf_repo_id   = os.environ.get("HF_REPO_ID", "your-username/autoedgequant-models")

    tool_log     = []
    messages     = [
        SystemMessage(content=Execution_Prompt),
        HumanMessage(content=f"[Planner Strategy]\n{state['planner_output']}")
    ]

    def fail(reason: str) -> AgentState:
        tool_log.append(f"[FAILED] {reason}")
        return {
            "messages":         messages + [AIMessage(content=reason)],
            "execution_output": f"FAILED: {reason}",
        }

    def run_tool(tool_fn, args: dict, step_name: str) -> tuple[dict, bool]:
        """Invoke a tool, append ToolMessage, return (result, success)."""
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # LLM might decide to call it — or we force it
        tool_call_id = None
        if response.tool_calls:
            tc = response.tool_calls[0]
            tool_call_id = tc["id"]
            args = {**tc["args"], **args}   # merge: LLM args + our forced args

        result = tool_fn.invoke(args)
        tool_log.append(f"[{step_name}]\n{json.dumps(result, indent=2)}")

        messages.append(ToolMessage(
            tool_call_id=tool_call_id or f"forced-{step_name}",
            content=json.dumps(result),
        ))
        return result, result.get("success", False)

    # ── Resolve what we're actually working with ──────────────────
    model_info = resolve_model_input(model_input)
    tool_log.append(
        f"[Model Resolution]\n"
        f"  Input:  {model_input}\n"
        f"  Found:  {model_info['path']}\n"
        f"  Format: {model_info['format']}"
    )

    if model_info["format"] == "not_found":
        return fail(
            f"Model '{model_input}' not found in models/hf/ or models/gguf/.\n"
            f"Please place your model in one of these directories:\n"
            f"  HF format  → models/hf/{model_info['name']}/\n"
            f"  GGUF file  → models/gguf/{model_info['name']}.gguf"
        )

    model_name  = model_info["name"]
    model_path  = model_info["path"]
    model_fmt   = model_info["format"]

    # Output paths — always in models/gguf/
    f16_path    = MODELS_GGUF_DIR / f"{model_name}-f16.gguf"
    quant_path  = MODELS_GGUF_DIR / f"{model_name}-{quant_type}.gguf"
    model_size_mb = 0.0

    # ── Case C: already a GGUF that matches the target quant ─────
    if model_fmt == "gguf" and quant_type.lower() in model_path.stem.lower():
        tool_log.append(
            f"[Skip] Model is already quantized as {quant_type}. "
            f"Skipping conversion and quantization."
        )
        quant_path    = model_path
        model_size_mb = round(model_path.stat().st_size / (1024**2), 2)

    else:
        #  Case A: GGUF but wrong/no quant → skip conversion only
        if model_fmt == "gguf":
            tool_log.append(
                f"[Skip Conversion] Input is already GGUF (f16 or different quant). "
                f"Using directly as quantization input."
            )
            f16_path = model_path   # use existing gguf as quantize input

        # Case B: HF directory → must convert first
        elif model_fmt == "hf_dir":
            tool_log.append(f"[Step 1] Converting HF model → GGUF F16 ...")
            messages.append(HumanMessage(
                content=f"Convert the HuggingFace model at {model_path} to GGUF F16 at {f16_path}."
            ))
            result, ok = run_tool(
                convert_hf_to_gguf,
                {"hf_model_path": str(model_path), "output_path": str(f16_path)},
                "convert_hf_to_gguf"
            )
            if not ok:
                return fail(f"Conversion failed: {result.get('error')}")
            f16_path = Path(result["output_model"])

        # Quantize (both Case A and B reach here))
        tool_log.append(f"[Step 2] Quantizing to {quant_type} ...")
        messages.append(HumanMessage(
            content=(
                f"Quantize {f16_path} to {quant_type}, "
                f"saving output to {quant_path}."
            )
        ))
        result, ok = run_tool(
            quantize_gguf,
            {
                "input_model":  str(f16_path),
                "output_model": str(quant_path),
                "quant_type":   quant_type,
            },
            "quantize_gguf"
        )
        if not ok:
            return fail(f"Quantization failed: {result.get('error')}")

        quant_path    = Path(result["output_model"])
        model_size_mb = result["model_size_mb"]

        # Clean up intermediate F16 if we created it (saves disk space on Pi)
        if model_fmt == "hf_dir" and f16_path.exists():
            f16_path.unlink()
            tool_log.append(f"[Cleanup] Removed intermediate F16 file: {f16_path.name}")

    #  Upload to HuggingFace (always) 
    tool_log.append(f"[Step 3] Uploading {quant_path.name} to HuggingFace ...")
    messages.append(HumanMessage(
        content=f"Upload {quant_path} to HuggingFace repo {hf_repo_id}."
    ))
    result, ok = run_tool(
        upload_model_hf,
        {
            "model_path":    str(quant_path),
            "repo_id":       hf_repo_id,
            "quant_type":    quant_type,
        },
        "upload_model_hf"
    )
    if not ok:
        return fail(f"Upload failed: {result.get('error')}")

    upload_result = result

    # Notify Pi 
    try:
        requests.post(f"{BRIDGE_URL}/notify-model", json={
            "hf_repo_id":    hf_repo_id,
            "quant_type":    quant_type,
            "filename":      upload_result["filename"],
            "model_size_mb": model_size_mb,
            "iteration":     state["iteration"],
        }, timeout=10)
        tool_log.append(f"[Bridge] Pi notified ✓ — {upload_result['hf_repo_url']}")
    except Exception as e:
        tool_log.append(f"[Bridge] WARNING: Pi notification failed: {e}")

    return {
        "messages":         messages,
        "execution_output": "\n\n".join(tool_log),
    }

def edge_processing_node(state: AgentState) -> AgentState:
    """
    Blocks until Pi submits inference results via the bridge API.
    This replaces dummy metrics — all data is real hardware numbers.
    """
    print(f"[Graph] Waiting for Pi results (timeout: 5 min)...")

    try:
        r = requests.get(f"{BRIDGE_URL}/wait-results", params={"timeout": 300})
        r.raise_for_status()
        pi_data = r.json()["results"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 408:
            pi_data = {"status": "timeout", "error": "Pi did not respond in 5 minutes"}
        else:
            raise

    edge_report = (
        f"[Edge Device Results - Iteration {state['iteration']}]\n"
        f"Status:          {pi_data['status']}\n"
        f"Quant Type:      {pi_data['quant_type']}\n"
        f"Avg Latency:     {pi_data['avg_latency_ms']} ms/token\n"
        f"Throughput:      {pi_data['tokens_per_sec']} tokens/sec\n"
        f"Peak RAM:        {pi_data['peak_ram_mb']} MB / {pi_data['total_ram_mb']} MB "
        f"({pi_data['ram_utilization']*100:.1f}%)\n"
        f"Sample Output:   {pi_data['sample_output']}\n"
        f"Error:           {pi_data.get('error', 'None')}"
    )

    return {
        "messages":         [AIMessage(content=edge_report)],
        "execution_output": edge_report,    # summarizer reads this field
    }


def summarizer_node(state: AgentState) -> AgentState:
    """
    Evaluates execution results, produces structured summary,
    and feeds a recommendation back to the planner.
    """
    messages = [
        SystemMessage(content=Summarizer_Prompt),
        HumanMessage(content=(
            f"[Planner Strategy - Iteration {state['iteration']}]\n"
            f"{state['planner_output']}\n\n"
            f"[Execution Results]\n"
            f"{state['execution_output']}"
        ))
    ]

    response = llm.invoke(messages)

    # Extract best quant from summary (agent should name it explicitly)
    best_quant = state.get("best_quant", "unknown")
    if "PASS" in response.content:
        # Try to pull the quant type from planner output as the confirmed best
        for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
            if quant in state["planner_output"]:
                best_quant = quant
                break

    return {
        "messages":          [response],
        "summarizer_output": response.content,
        "iteration":         state["iteration"] + 1,   # increment here, after full loop
        "best_quant":        best_quant,
    }


# Routing Logic
def should_continue(state: AgentState) -> str:
    """
    After summarizer runs:
    - PASS verdict   → END immediately (good result found)
    - ESCALATE       → END (device can't run any quant)
    - iter >= MAX    → END (budget exhausted)
    - RETRY          → back to planner
    """
    summary = state["summarizer_output"]
    iteration = state["iteration"]

    if "ESCALATE" in summary:
        print(f"[Router] ESCALATE — device cannot run any viable quantization.")
        return END

    if "PASS" in summary:
        print(f"[Router] PASS at iteration {iteration} — best quant: {state['best_quant']}")
        return END

    if iteration >= MAX_ITERATIONS:
        print(f"[Router] Max iterations ({MAX_ITERATIONS}) reached.")
        return END

    print(f"[Router] RETRY — starting iteration {iteration + 1}")
    return "planner"


# Graph Assembly
PLANNER    = "planner"
EXECUTION  = "execution"
EDGE = "edge_processing"
SUMMARIZER = "summarizer"

workflow = StateGraph(AgentState)

workflow.add_node(PLANNER,    planner_node)
workflow.add_node(EXECUTION,  execution_node)
workflow.add_node(EDGE,       edge_processing_node)   # ← new
workflow.add_node(SUMMARIZER, summarizer_node)

workflow.set_entry_point(PLANNER)
workflow.add_edge(PLANNER,   EXECUTION)
workflow.add_edge(EXECUTION, EDGE)         # ← wait for Pi
workflow.add_edge(EDGE,      SUMMARIZER)   # ← then summarize real metrics
workflow.add_conditional_edges(SUMMARIZER, should_continue, {"planner": PLANNER, END: END})


app = workflow.compile()

# Run
if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=(
                "Model: meta-llama/Llama-3.2-1B\n"
                "Edge Device: Raspberry Pi 4 (4GB RAM), ARM Cortex-A72, 4 cores, Ubuntu 22.04"
            ))
        ],
        "planner_output":   "",
        "execution_output": "",
        "summarizer_output": "",
        "iteration":        0,
        "best_quant":       "",
        "final_result":     "",
    }

    final_state = app.invoke(initial_state)

    print("\n" + "="*50)
    print(f"✅ Best quantization strategy: {final_state['best_quant']}")
    print(f"   Completed in {final_state['iteration']} iteration(s)")
    print("="*50)