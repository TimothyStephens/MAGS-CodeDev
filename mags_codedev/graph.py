from langgraph.constants import END, START
from langgraph.graph import StateGraph
from mags_codedev.state import ModuleState

# Import placeholder agent and utility nodes. 
# (We will implement these in the agents/ and utils/ directories next)
from mags_codedev.agents.coder import coder_node
from mags_codedev.agents.tester import tester_node
from mags_codedev.utils.docker_ops import test_node, linter_node
from mags_codedev.agents.log_checker import log_checker_node
from mags_codedev.agents.reviewer import multi_llm_review_node

def build_function_graph():
    """
    Constructs the LangGraph state machine for processing a single module.
    This graph will be executed in parallel for multiple modules.
    """
    workflow = StateGraph(ModuleState)
    
    # ---------------------------------------------------------
    # 1. Define Nodes (Agents and Tools)
    # ---------------------------------------------------------
    # The Coder generates or fixes the function code
    workflow.add_node("coder", coder_node)
    
    # The Tester writes comprehensive unit tests for the code
    workflow.add_node("tester", tester_node)
    
    # Tool: Runs tests in the configured execution environment (docker, apptainer, local)
    workflow.add_node("run_tests", test_node)
    
    # Tool: Runs MyPy/Flake8/Ruff on the code
    workflow.add_node("run_linters", linter_node)
    
    # The Log Checker acts as a diagnostic agent, translating raw tracebacks
    workflow.add_node("log_checker", log_checker_node)
    
    # Multi-Agent Review runs concurrently to aggregate peer-review feedback
    workflow.add_node("multi_llm_review", multi_llm_review_node)
    
    # ---------------------------------------------------------
    # 2. Define Standard Edges (Linear Flow)
    # ---------------------------------------------------------
    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "tester")
    workflow.add_edge("tester", "run_tests")
    
    # ---------------------------------------------------------
    # 3. Define Conditional Edges (Decision Logic)
    # ---------------------------------------------------------
    
    # A. Evaluate Docker Test Results
    def evaluate_test_results(state: ModuleState) -> str:
        max_iters = state.get("max_iterations", 5)
        if max_iters > 0 and state["iteration_count"] >= max_iters:
            return "max_iterations_reached"
            
        # The test_node will populate state["test_results"]
        if "FAILED" in state["test_results"].upper() or "ERROR" in state["test_results"].upper():
            return "tests_failed"
        return "tests_passed"
        
    workflow.add_conditional_edges(
        "run_tests",
        evaluate_test_results,
        {
            "tests_passed": "run_linters",
            "tests_failed": "log_checker",
            "max_iterations_reached": END
        }
    )

    # Linters feed directly into the log checker to assess warnings/errors
    workflow.add_edge("run_linters", "log_checker")

    # B. Evaluate Logs (Diagnostic Phase)
    def evaluate_logs(state: ModuleState) -> str:
        max_iters = state.get("max_iterations", 5)
        if max_iters > 0 and state["iteration_count"] >= max_iters:
            return "max_iterations_reached"
            
        # If the log checker populates an error_summary, route to the correct fixer
        if state.get("error_summary"):
            if state.get("error_location") == "TEST_CODE":
                return "fix_tests"
            return "fix_source" # Default to fixing source
        return "clean"

    workflow.add_conditional_edges(
        "log_checker",
        evaluate_logs,
        {
            "clean": "multi_llm_review",
            "fix_source": "coder",
            "fix_tests": "tester",
            "max_iterations_reached": END
        }
    )

    # C. Multi-LLM Review Phase
    def evaluate_reviews(state: ModuleState) -> str:
        max_iters = state.get("max_iterations", 5)
        if max_iters > 0 and state["iteration_count"] >= max_iters:
            return "max_iterations_reached"
            
        # If reviewers aggregated actionable comments, send back to Coder to revise
        if state.get("review_comments") and len(state["review_comments"]) > 0:
            return "revise"
        
        # If the list is empty, all reviewers approved
        return "approved"

    workflow.add_conditional_edges(
        "multi_llm_review",
        evaluate_reviews,
        {
            "approved": END,
            "revise": "coder",
            "max_iterations_reached": END
        }
    )
    
    # Compile the graph into a runnable LangChain executable
    return workflow.compile()