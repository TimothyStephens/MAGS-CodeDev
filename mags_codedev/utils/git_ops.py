"""Git operations: worktree management, branch creation, merge cleanup."""

import subprocess
import os
import shutil
import git
import logging

logger = logging.getLogger("mags_codedev")


def _get_base_branch(repo: git.Repo) -> str:
    """Return the primary branch name (main or master)."""
    if "main" in repo.heads:
        return "main"
    if "master" in repo.heads:
        return "master"
    return "main"  # fallback


def validate_git_repo():
    """Ensures the current directory is a valid git repo with a main/master branch."""
    try:
        repo = git.Repo(os.getcwd())
        base_branch = _get_base_branch(repo)
        if base_branch not in repo.heads:
            raise RuntimeError("Git repository must have a 'main' or 'master' branch.")
    except git.InvalidGitRepositoryError:
        raise RuntimeError("Current directory is not a git repository. Run `git init` first.")
    except Exception as e:
        raise RuntimeError(f"Git validation failed: {e}")


def create_parallel_worktree(branch_name: str, base_dir: str = ".mags-codedev", force_fresh: bool = False) -> str:
    """Creates a new git branch and checks it out in an isolated worktree directory.

    Worktrees are stored under <base_dir>/worktrees/feature-<safe_name>.
    """
    worktrees_dir = os.path.join(base_dir, "worktrees")
    os.makedirs(worktrees_dir, exist_ok=True)
    # Strip "feature/" prefix to avoid double prefix in directory name
    safe_branch = branch_name.removeprefix("feature/")
    safe_dir_name = safe_branch.replace("/", "_")
    worktree_path = os.path.abspath(os.path.join(worktrees_dir, f"feature-{safe_dir_name}"))

    repo = git.Repo(os.getcwd())

    if force_fresh:
        if os.path.exists(worktree_path):
            subprocess.run(
                ["git", "worktree", "remove", worktree_path, "--force"],
                check=False, capture_output=True,
            )
            shutil.rmtree(worktree_path)

    # 1. Reuse existing worktree if available (Iteration Mode)
    if os.path.exists(worktree_path):
        return worktree_path

    # 2. Reuse existing branch if available (but worktree dir is missing)
    if branch_name in repo.heads:
        subprocess.run(["git", "worktree", "prune"], check=False, capture_output=True)
        try:
            result = subprocess.run(
                ["git", "worktree", "add", worktree_path, branch_name],
                check=True, capture_output=True,
            )
            if result.returncode == 0:
                return worktree_path
        except subprocess.CalledProcessError:
            # Branch exists but worktree creation failed — delete branch
            repo.delete_head(branch_name, force=True)

    # 3. Create Fresh Worktree
    subprocess.run(["git", "worktree", "prune"], check=False, capture_output=True)

    base_branch = _get_base_branch(repo)
    try:
        subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, worktree_path, base_branch],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else "Unknown git error"
        raise RuntimeError(f"Failed to create worktree: {error_msg}") from e
    return worktree_path


def merge_and_cleanup_worktree(
    branch_name: str, worktree_path: str, success: bool,
    base_dir: str = ".mags-codedev",
) -> bool:
    """Merges the branch to main if successful. Preserves branch on merge conflict.

    On merge failure, backs up the worktree to <base_dir>/merges/ before cleanup.
    Returns True if merge succeeded, False otherwise.
    """
    repo = git.Repo(os.getcwd())
    merge_success = True

    try:
        base_branch = _get_base_branch(repo)

        if success:
            repo.git.checkout(base_branch)
            merge_result = subprocess.run(
                ["git", "merge", branch_name, "--no-edit"],
                capture_output=True, text=True
            )

            if merge_result.returncode != 0:
                # Merge conflict — abort and preserve branch
                repo.git.merge("--abort")
                merge_success = False
                # Backup worktree on merge failure before cleanup
                _backup_worktree(worktree_path, branch_name, base_dir)
            else:
                # Merge successful — delete branch and worktree
                subprocess.run(
                    ["git", "worktree", "remove", worktree_path, "--force"],
                    check=False, capture_output=True,
                )
                repo.delete_head(branch_name)
        else:
            # Build failed — remove branch and worktree
            subprocess.run(
                ["git", "worktree", "remove", worktree_path, "--force"],
                check=False, capture_output=True,
            )
            if branch_name in repo.heads:
                repo.delete_head(branch_name, force=True)
    except Exception:
        logger.exception("Error during merge and cleanup")
        merge_success = False
    return merge_success


def _backup_worktree(worktree_path: str, branch_name: str, base_dir: str) -> None:
    """Backup a worktree to <base_dir>/merges/ on merge failure."""
    import tarfile
    import logging
    from datetime import datetime

    logger = logging.getLogger("mags_codedev")
    merges_dir = os.path.join(base_dir, "merges")
    os.makedirs(merges_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = branch_name.replace("/", "_")
    backup_path = os.path.join(merges_dir, f"{timestamp}_{safe_name}")

    try:
        with tarfile.open(f"{backup_path}.tar.gz", "w:gz") as tar:
            tar.add(worktree_path, arcname=safe_name)
        logger.info(f"Worktree backed up to {backup_path}.tar.gz")
    except Exception as e1:
        try:
            shutil.copytree(worktree_path, f"{backup_path}_copy")
            logger.info(f"Worktree backed up (copy) to {backup_path}_copy")
        except Exception as e2:
            logger.error(
                f"Failed to backup worktree for branch '{branch_name}': "
                f"tar failed ({e1}), copy failed ({e2})"
            )

def ensure_git_repo() -> None:
    """Initialize a git repo if one doesn't exist. Creates 'main' branch."""
    if os.path.isdir(".git"):
        return

    repo = git.Repo.init()
    # Create an empty initial commit on 'main' branch
    repo.git.checkout("-b", "main")
    repo.git.commit("--allow-empty", "-m", "Initial commit")
