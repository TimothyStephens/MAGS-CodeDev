"""Coder agent: generates or fixes module source code via LLM."""

from mags_codedev.state import ModuleState
from mags_codedev.utils.llm_call import build_context_blocks, call_llm


def coder_node(state: ModuleState) -> dict:
    """Generate or update the module code based on the spec and feedback.

    The LLM call, retry, content cleanup, and conversation logging are handled
    by :func:`call_llm`; this node owns only the role-specific prompt assembly
    (fix-vs-initial context, narrative, feedback packaging).
    """
    backend = state.get("backend")

    # Determine context based on whether this is a first run or a fix.
    is_fix = bool(state.get("review_comments")) or bool(state.get("test_error_summary"))
    feedback = ""
    prompt_narrative = "Write the initial implementation of this module."

    if is_fix:
        # Carry both diagnostics (log_checker errors + reviewer comments) and
        # the existing code/tests so the coder knows what to change.
        feedback_parts = []
        test_error = state.get("test_error_summary", "")
        if test_error and "no clear issues" not in test_error.lower():
            feedback_parts.append(f"TEST/LINT ERRORS:\n{test_error}")
        review_comments = state.get("review_comments", [])
        if review_comments:
            feedback_parts.append(
                "REVIEW COMMENTS:\n" + "\n".join(f"- {c}" for c in review_comments)
            )
        if state.get("code"):
            feedback_parts.append(f"EXISTING CODE:\n{state['code']}")
        if state.get("tests"):
            feedback_parts.append(f"CURRENT TESTS:\n{state['tests']}")
        feedback = "\n\n".join(feedback_parts) if feedback_parts else ""

        if review_comments and not test_error:
            prompt_narrative = "Revise the following code based on the review feedback."
        elif test_error:
            prompt_narrative = "Fix the following code based on the test/lint errors."
        else:
            prompt_narrative = "Update the following code based on the specification."
    elif state.get("code"):
        prompt_narrative = "Update the following module based on the specification."
        feedback = f"EXISTING CODE:\n{state.get('code', '')}"

    system_prompt = (
        backend.coder_system_prompt() if backend
        else (
            "You are an expert Software Engineer.\n"
            "Write a full, clean, production-ready module. "
            "Follow all instructions exactly.\n"
            "Return ONLY valid code for the entire file. "
            "Do not include markdown formatting."
        )
    )

    proj_block, dep_block = build_context_blocks(state)
    human_template = (
        "\nModule Location: {module_location}\n"
        "Specification: {spec}\n\n"
        "{project_instructions_block}"
        "{dependency_context}"
        "{prompt_narrative}\n\n"
        "{feedback}\n"
    )
    params = {
        "module_location": state["module_location"],
        "spec": str(state["spec"]),
        "project_instructions_block": proj_block,
        "dependency_context": dep_block,
        "prompt_narrative": prompt_narrative,
        "feedback": feedback,
    }

    response_content = call_llm(
        role="coder",
        system_prompt=system_prompt,
        human_template=human_template,
        params=params,
        state=state,
        narrative=prompt_narrative,
    )

    return {
        "code": response_content + "\n",
        "iteration_count": state.get("iteration_count", 0) + (0 if state.get("review_comments") else 1),
        "review_comments": [],
    }
