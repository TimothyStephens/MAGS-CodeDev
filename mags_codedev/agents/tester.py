"""Tester agent: generates or fixes pytest unit tests via LLM."""

from pathlib import Path

from mags_codedev.state import ModuleState
from mags_codedev.utils.llm_call import build_context_blocks, call_llm


def tester_node(state: ModuleState) -> dict:
    """Write comprehensive unit tests for the current module code.

    The LLM call, retry, content cleanup, and conversation logging are handled
    by :func:`call_llm`; this node owns only the role-specific prompt assembly
    (initial-vs-fix context, import hint, review feedback about tests).
    """
    backend = state.get("backend")
    source_location = state["module_location"]

    if backend:
        import_instruction = backend.get_import_hint(Path(source_location))
    else:
        module_path = Path(source_location).with_suffix("").as_posix().replace("/", ".")
        import_instruction = (
            f"The module to test is '{source_location}'. "
            f"You can import from it using `from {module_path} import ...`."
        )

    system_prompt = (
        backend.tester_system_prompt(source_location, import_instruction) if backend
        else (
            "You are a strict QA Automation Engineer.\n"
            "Write robust unit tests for the provided module.\n"
            "The code is part of a larger project, so ensure imports are correct.\n"
            f"{import_instruction}\n"
            "Your tests should cover all functions and classes in the module.\n"
            "Include edge cases, type boundary checks, and failure scenarios.\n"
            f"The module to be tested is located at the path '{source_location}'.\n"
            "Return ONLY valid code for the test file. Do not include markdown formatting."
        )
    )

    proj_block, dep_block = build_context_blocks(state)

    is_fix = bool(state.get("test_error_summary")) and state.get("error_location") == "TEST_CODE"
    if is_fix:
        # Surface review comments that mention tests as extra fix guidance.
        review_feedback = ""
        test_reviews = [c for c in state.get("review_comments", []) if "test" in c.lower()]
        if test_reviews:
            review_feedback = (
                "REVIEW COMMENTS ABOUT TESTS:\n"
                + "\n".join(f"- {c}" for c in test_reviews) + "\n\n"
            )
        human_template = proj_block + dep_block + (
            backend.test_human_template_fix() if backend
            else (
                "The previous attempt to write tests failed. Please fix them.\n\n"
                "Module Specification:\n{spec}\n\n"
                "Generated Code to Test:\n{code}\n\n"
                "PREVIOUS (BROKEN) TESTS:\n{previous_tests}\n\n"
                "DIAGNOSIS OF FAILURE:\n{test_error_summary}\n\n"
                "{review_feedback}\n"
                "Your task is to provide a new, corrected version of the unit tests."
            )
        )
        params = {
            "spec": str(state["spec"]),
            "code": state["code"],
            "previous_tests": state["tests"],
            "test_error_summary": state["test_error_summary"],
            "review_feedback": review_feedback,
        }
        narrative = "Fix the failing tests based on the diagnosis."
    else:
        human_template = proj_block + dep_block + (
            backend.test_human_template_initial() if backend
            else "Module Specification:\n{spec}\n\nGenerated Code to Test:\n{code}"
        )
        params = {"spec": str(state["spec"]), "code": state["code"]}
        narrative = "Write unit tests for the module."

    response_content = call_llm(
        role="tester",
        system_prompt=system_prompt,
        human_template=human_template,
        params=params,
        state=state,
        narrative=narrative,
    )

    return {
        "tests": response_content + "\n",
        "iteration_count": state.get("iteration_count", 0),
    }
