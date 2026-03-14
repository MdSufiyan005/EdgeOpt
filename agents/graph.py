# graph.py
from typing import List, TypedDict, Annotated, Optional
from pathlib import Path
import operator
import json
import requests
import os
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END, StateGraph

from tools.execution import (
    convert_hf_to_gguf, 
    quantize_gguf, 
    upload_model_hf,
    download_hf_model
)
from chain import planner_chain, execution_chain, summarizer_chain
from logging_ import adding_logs

load_dotenv()
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://localhost:8080")
print(f"----- Using Broker url {BRIDGE_URL} -----")

MODELS_GGUF_DIR = Path("models/gguf")
MODELS_GGUF_DIR.mkdir(parents=True, exist_ok=True)

logger = adding_logs("graph")

# ====================== STATE ======================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    model_input: str
    planner_output: str
    execution_output: str
    summarizer_output: str
    iteration: int
    best_quant: str
    final_result: str

MAX_ITERATIONS = 4



# ====================== HELPERS ======================
def extract_upload_result(tool_log: List[str]) -> Optional[dict]:
    """Extract filename and size from the last successful upload_model_hf call."""
    for entry in reversed(tool_log):
        if "[upload_model_hf]" in entry:
            try:
                # tool_log contains json after the header
                data = json.loads(entry.split("]\n", 1)[-1])
                if data.get("success"):
                    return {
                        "filename": data.get("filename"),
                        "model_size_mb": data.get("model_size_mb", 0)
                    }
            except:
                continue
    return None


def _extract_quant_type(planner_output: str) -> str:
    for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                  "Q3_K_M", "Q3_K_S", "Q2_K"]:
        if quant in planner_output:
            return quant
    return "Q4_K_M"


# ====================== NODE: PLANNER ======================
def planner_node(state: AgentState) -> AgentState:
    logger.info(f"Starting planner node for iteration {state['iteration']}")
    messages = list(state["messages"])

    # Inject previous feedback on retries
    if state["iteration"] > 0 and state.get("summarizer_output"):
        messages.append(
            HumanMessage(content=f"[Summarizer Feedback - Iteration {state['iteration']-1}]\n{state['summarizer_output']}")
        )

    response = planner_chain.invoke({"messages": messages})
    logger.info(f"Planner output: {response.content[:100]}...")

    return {
        "messages": [response],                    # only add the planner's message
        "planner_output": response.content,
        "iteration": state["iteration"],           # unchanged
    }


# ====================== TOOL EXECUTION LOOP ======================
def run_agent_loop(messages: list, tool_log: list, max_steps: int = 12) -> list:
    logger.info("Starting tool execution loop")
    from langchain_core.messages import ToolMessage

    for step in range(max_steps):
        logger.debug(f"Step {step + 1}/{max_steps}")
        response = execution_chain.invoke({"messages": messages})
        messages.append(response)

        if not response.tool_calls:
            logger.info("No more tool calls, ending loop")
            break

        for tc in response.tool_calls:
            logger.info(f"Executing tool: {tc['name']}")
            tool_map = {
                "download_hf_model": download_hf_model,
                "convert_hf_to_gguf": convert_hf_to_gguf,
                "quantize_gguf": quantize_gguf,
                "upload_model_hf": upload_model_hf,
            }
            tool_fn = tool_map.get(tc["name"])
            if not tool_fn:
                logger.warning(f"Unknown tool: {tc['name']}")
                tool_log.append(f"[WARNING] Unknown tool: {tc['name']}")
                continue

            result = tool_fn.invoke(tc["args"])
            tool_log.append(f"[{tc['name']}]\n{json.dumps(result, indent=2)}")
            logger.info(f"Tool {tc['name']} result: success={result.get('success', False)}")

            messages.append(ToolMessage(
                tool_call_id=tc["id"],
                content=json.dumps(result, ensure_ascii=False)
            ))
    logger.info("Tool execution loop completed")
    return messages


# ====================== NODE: EXECUTION ======================
def execution_node(state: AgentState) -> AgentState:
    logger.info(f"Starting execution node for iteration {state['iteration']}")
    quant_type = _extract_quant_type(state["planner_output"])
    hf_repo_id = os.environ.get("HF_REPO_ID", "yourusername/quantized-edge-models")
    tool_log = []

    # Build task message (append to existing history)
    task_msg = HumanMessage(content=(
        f"[Task - Iteration {state['iteration']}]\n"
        f"Model: {state['model_input']}\n"
        f"Target quant: {quant_type}\n"
        f"Upload repo: {hf_repo_id}\n\n"
        f"[Planner Strategy]\n{state['planner_output']}\n\n"
        f"Execute full pipeline: download → convert → quantize → upload."
    ))

    messages = list(state["messages"]) + [task_msg]
    messages = run_agent_loop(messages, tool_log)

    # Notify Raspberry Pi (only if upload succeeded)
    upload_info = extract_upload_result(tool_log)
    if upload_info:
        logger.info("Upload successful, notifying Raspberry Pi")
        try:
            requests.post(
                f"{BRIDGE_URL}/notify-model",
                json={
                    "hf_repo_id": hf_repo_id,
                    "quant_type": quant_type,
                    "filename": upload_info["filename"],
                    "model_size_mb": upload_info["model_size_mb"],
                    "iteration": state["iteration"],
                },
                timeout=10
            )
            tool_log.append("[Bridge] Pi notified successfully ✓")
            logger.info("Raspberry Pi notified successfully")
        except Exception as e:
            logger.error(f"Failed to notify Raspberry Pi: {e}")
            tool_log.append(f"[Bridge] WARNING: {e}")
    else:
        logger.info("No successful upload, skipping Pi notification")

    return {
        "messages": messages,                      # full updated history
        "execution_output": "\n\n".join(tool_log),
        "iteration": state["iteration"],
    }


# ====================== NODE: EDGE (RaspConnector) ======================
def edge_processing_node(state: AgentState) -> AgentState:
    logger.info(f"Starting edge processing node for iteration {state['iteration']}")
    print(f"[Graph] Waiting for Pi results (timeout: 300s)...")

    try:
        r = requests.get(f"{BRIDGE_URL}/wait-results", params={"timeout": 300}, timeout=310)
        r.raise_for_status()
        pi_data = r.json().get("results", {})
        logger.info(f"Received Pi results: status={pi_data.get('status', 'unknown')}")
    except requests.exceptions.Timeout:
        pi_data = {"status": "timeout", "error": "Pi did not respond in time"}
        logger.error("Timeout waiting for Pi results")
    except Exception as e:
        pi_data = {"status": "error", "error": str(e)}
        logger.error(f"Error getting Pi results: {e}")

    edge_report = (
        f"[Edge Device Results - Iteration {state['iteration']}]\n"
        f"Status:          {pi_data.get('status', 'unknown')}\n"
        f"Quant Type:      {pi_data.get('quant_type', 'N/A')}\n"
        f"Avg Latency:     {pi_data.get('avg_latency_ms', 0)} ms/token\n"
        f"Throughput:      {pi_data.get('tokens_per_sec', 0)} tokens/sec\n"
        f"Peak RAM:        {pi_data.get('peak_ram_mb', 0)} MB / {pi_data.get('total_ram_mb', 4000)} MB "
        f"({pi_data.get('ram_utilization', 0)*100:.1f}%)\n"
        f"Sample Output:   {pi_data.get('sample_output', 'N/A')}\n"
        f"Error:           {pi_data.get('error', 'None')}"
    )
    logger.info(f"Edge report generated: {edge_report[:100]}...")

    return {
        "messages": [AIMessage(content=edge_report)],
        "execution_output": edge_report,
        "iteration": state["iteration"],
    }


# ====================== NODE: SUMMARIZER ======================
def summarizer_node(state: AgentState) -> AgentState:
    logger.info(f"Starting summarizer node for iteration {state['iteration']}")
    human_content = (
        f"[Planner Strategy - Iteration {state['iteration']}]\n"
        f"{state['planner_output']}\n\n"
        f"[Execution + Real Pi Results]\n"
        f"{state['execution_output']}"
    )
    

    response = summarizer_chain.invoke({"messages": human_content})
    logger.info(f"Summarizer output: {response.content[:100]}...")

    # Extract best quant on PASS
    best_quant = state.get("best_quant", "unknown")
    if "PASS" in response.content.upper():
        for quant in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
            if quant in state["planner_output"]:
                best_quant = quant
                logger.info(f"Best quant found: {best_quant}")
                break

    return {
        "messages": [response],
        "summarizer_output": response.content,
        "iteration": state["iteration"] + 1,
        "best_quant": best_quant,
    }


# ====================== ROUTER ======================
def should_continue(state: AgentState) -> str:
    summary = state.get("summarizer_output", "").upper()
    iteration = state["iteration"]

    if "ESCALATE" in summary:
        logger.info("Router decision: ESCALATE — device cannot run any viable quantization.")
        return END
    if "PASS" in summary:
        logger.info(f"Router decision: PASS at iteration {iteration} — best quant: {state['best_quant']}")
        return END
    if iteration >= MAX_ITERATIONS:
        logger.info(f"Router decision: Max iterations ({MAX_ITERATIONS}) reached.")
        return END

    logger.info(f"Router decision: RETRY — starting iteration {iteration + 1}")
    return "planner"


# ====================== GRAPH ASSEMBLY ======================
PLANNER = "planner"
EXECUTION = "execution"
EDGE = "edge_processing"
SUMMARIZER = "summarizer"

workflow = StateGraph(AgentState)

workflow.add_node(PLANNER,    planner_node)
workflow.add_node(EXECUTION,  execution_node)
workflow.add_node(EDGE,       edge_processing_node)
workflow.add_node(SUMMARIZER, summarizer_node)

workflow.set_entry_point(PLANNER)
workflow.add_edge(PLANNER,    EXECUTION)
workflow.add_edge(EXECUTION,  EDGE)
workflow.add_edge(EDGE,       SUMMARIZER)
workflow.add_conditional_edges(SUMMARIZER, should_continue, {"planner": PLANNER, END: END})

app = workflow.compile()

# ====================== RUN ======================
if __name__ == "__main__":
    print(app.get_graph().draw_mermaid())

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=(
                "Model: meta-llama/Llama-3.2-1B\n"
                "Edge Device: Raspberry Pi 4 (4GB RAM), ARM Cortex-A72, 4 cores, Ubuntu 22.04"
            ))
        ],
        "model_input": "meta-llama/Llama-3.2-1B",
        "planner_output": "",
        "execution_output": "",
        "summarizer_output": "",
        "iteration": 0,
        "best_quant": "",
        "final_result": "",
    }

    logger.info("Starting graph execution")
    final_state = app.invoke(initial_state)
    logger.info(f"Graph execution completed in {final_state['iteration']} iterations")

    print("\n" + "=" * 60)
    print(f"✅ BEST QUANTIZATION: {final_state['best_quant']}")
    print(f"   Completed in {final_state['iteration']} iteration(s)")
    print("=" * 60)