"""Tests for spec-aware rebuild invalidation (P1-4)."""

from mags_codedev.utils.db import (
    hash_spec,
    hash_spec_content,
    init_db,
    is_function_built,
    mark_function_built,
)


def _spec(location="src/foo.py", description="does foo", dependencies=None):
    return {"location": location, "description": description,
            "dependencies": dependencies or []}


class TestSpecInvalidation:
    def test_location_hash_stable_across_spec_edits(self):
        """hash_spec is location-based (stable) so logs/artifacts stay continuous."""
        s = _spec(description="v1")
        assert hash_spec(s) == hash_spec(_spec(description="v2"))

    def test_content_hash_changes_with_description(self):
        s1 = _spec(description="v1")
        s2 = _spec(description="v2")
        assert hash_spec_content(s1) != hash_spec_content(s2)

    def test_content_hash_changes_with_dependencies(self):
        s1 = _spec(dependencies=["src/a.py", "src/b.py"])
        s2 = _spec(dependencies=["src/b.py", "src/a.py", "src/c.py"])
        assert hash_spec_content(s1) != hash_spec_content(s2)

    def test_dependency_order_does_not_change_hash(self):
        """Order-independent: deps in different order hash the same."""
        s1 = _spec(dependencies=["src/a.py", "src/b.py"])
        s2 = _spec(dependencies=["src/b.py", "src/a.py"])
        assert hash_spec_content(s1) == hash_spec_content(s2)

    def test_built_then_spec_edit_invalidates(self, temp_dir):
        import os
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        spec = _spec(description="original spec")
        mark_function_built("foo", spec, base_dir=base_dir)
        assert is_function_built(spec, base_dir=base_dir)

        # Edit the description -> the same location should now rebuild.
        edited = _spec(description="refined spec with more detail")
        assert not is_function_built(edited, base_dir=base_dir)

        # Rebuilding with the new spec re-marks it built.
        mark_function_built("foo", edited, base_dir=base_dir)
        assert is_function_built(edited, base_dir=base_dir)
