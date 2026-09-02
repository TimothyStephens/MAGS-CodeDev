import pytest
from mags_codedev.graph import evaluate_test_results, evaluate_logs, evaluate_reviews
from mags_codedev.state import ModuleState


class TestGraphRouting:
    def test_passed_tests_route_to_linters(self):
        state = ModuleState(
            test_results="===== 3 passed in 0.50s =====",
            iteration_count=1,
            max_test_fix_iterations=5,
        )
        assert evaluate_test_results(state) == "tests_passed"

    def test_failed_tests_route_to_log_checker(self):
        state = ModuleState(
            test_results="FAILED test_foo - AssertionError",
            iteration_count=1,
            max_test_fix_iterations=5,
        )
        assert evaluate_test_results(state) == "tests_failed"

    def test_max_test_iterations_ends(self):
        state = ModuleState(
            test_results="FAILED",
            iteration_count=5,
            max_test_fix_iterations=5,
        )
        assert evaluate_test_results(state) == "max_iterations_reached"

    def test_rc_zero_with_error_in_name_routes_to_linters(self):
        """Exit code 0 wins over failure keywords in test names (M4)."""
        state = ModuleState(
            test_results="test_error_handling PASSED\n===== 1 passed in 0.20s =====",
            test_returncode=0,
            iteration_count=1,
            max_test_fix_iterations=5,
        )
        assert evaluate_test_results(state) == "tests_passed"

    def test_rc_nonzero_without_keywords_routes_to_log_checker(self):
        """Non-zero exit code routes to log_checker even without failure keywords (M4)."""
        state = ModuleState(
            test_results="===== 1 passed in 0.50s =====",
            test_returncode=1,
            iteration_count=1,
            max_test_fix_iterations=5,
        )
        assert evaluate_test_results(state) == "tests_failed"

    def test_source_error_routes_to_coder(self):
        state = ModuleState(
            test_error_summary="TypeError in main function",
            error_location="SOURCE_CODE",
            iteration_count=2,
            max_test_fix_iterations=5,
        )
        assert evaluate_logs(state) == "fix_source"

    def test_test_error_routes_to_tester(self):
        state = ModuleState(
            test_error_summary="Test has wrong assertion",
            error_location="TEST_CODE",
            iteration_count=2,
            max_test_fix_iterations=5,
        )
        assert evaluate_logs(state) == "fix_tests"

    def test_clean_routes_to_review(self):
        state = ModuleState(
            test_error_summary="No clear issues detected",
            error_location=None,
            iteration_count=2,
            max_test_fix_iterations=5,
        )
        assert evaluate_logs(state) == "clean"

    def test_empty_error_summary_routes_to_review(self):
        state = ModuleState(
            test_error_summary="",
            iteration_count=2,
            max_test_fix_iterations=5,
        )
        assert evaluate_logs(state) == "clean"

    def test_review_with_comments_routes_to_coder(self):
        state = ModuleState(
            review_comments=["Add input validation"],
            review_round_count=1,
            max_review_rounds=3,
        )
        assert evaluate_reviews(state) == "revise"

    def test_review_approved_ends(self):
        state = ModuleState(
            review_comments=[],
            review_round_count=0,
            max_review_rounds=3,
        )
        assert evaluate_reviews(state) == "approved"

    def test_max_review_rounds_ends(self):
        state = ModuleState(
            review_comments=["Fix this"],
            review_round_count=3,
            max_review_rounds=3,
        )
        assert evaluate_reviews(state) == "max_review_rounds_reached"

    def test_linter_only_routes_to_clean(self):
        """Pure linter warnings (no test failures) should route to clean."""
        state = ModuleState(
            test_error_summary="linter warning: line too long",
            error_location=None,
            iteration_count=1,
            max_test_fix_iterations=5,
        )
        assert evaluate_logs(state) == "clean"
