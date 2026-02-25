import subprocess
import os
import shutil
import git

def validate_git_repo():
    """Ensures the current directory is a valid git repo with a main/master branch."""
    try:
        repo = git.Repo(os.getcwd())
        if repo.bare:
            raise RuntimeError("Cannot run in a bare git repository.")
        # Check if 'main' exists (or master, though we default to main)
        if 'main' not in repo.heads and 'master' not in repo.heads:
            raise RuntimeError("Git repository must have a 'main' or 'master' branch.")
    except git.InvalidGitRepositoryError:
        raise RuntimeError("Current directory is not a git repository. Run `git init` first.")
    except Exception as e:
        raise RuntimeError(f"Git validation failed: {e}")

def create_parallel_worktree(branch_name: str) -> str:
    """Creates a new git branch and checks it out in an isolated worktree directory."""
    # Validation is now handled by the caller (cli.py) via validate_git_repo()

    # Sanitize branch name for directory usage to avoid nested paths (e.g. feature/foo -> feature_foo)
    safe_dir_name = branch_name.replace("/", "_")
    worktree_path = os.path.abspath(f".worktree_{safe_dir_name}")
    
    # Clean up any stale worktree directory from previous failed runs
    if os.path.exists(worktree_path):
        shutil.rmtree(worktree_path)
        
    # Prune git worktree metadata to ensure we can create a new one
    subprocess.run(["git", "worktree", "prune"], check=False, capture_output=True)
    
    # Ensure the branch doesn't exist from a failed previous run
    try:
        repo = git.Repo(os.getcwd())
        if branch_name in repo.heads:
            # Force delete the branch to start fresh
            repo.delete_head(branch_name, force=True)
    except (git.InvalidGitRepositoryError, OSError):
        pass 

    # Create branch and worktree
    try:
        # Determine base branch
        base_branch = "main"
        if "main" not in repo.heads and "master" in repo.heads:
            base_branch = "master"
        subprocess.run(["git", "worktree", "add", "-b", branch_name, worktree_path, base_branch], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='replace').strip() if e.stderr else "Unknown git error"
        raise RuntimeError(f"Failed to create worktree: {error_msg}") from e
    return worktree_path

def merge_and_cleanup_worktree(branch_name: str, worktree_path: str, success: bool) -> bool:
    """Merges the branch to main if successful. Preserves branch on merge conflict. Returns True if merge was successful."""
    merge_success = False
    
    if success:
        try:
            # Checkout main and merge
            # We use check=True to catch merge conflicts
            subprocess.run(["git", "checkout", "main"], check=True, capture_output=True)
            subprocess.run(["git", "merge", "--no-ff", "-m", f"feat: Merge function '{branch_name}'", branch_name], check=True, capture_output=True)
            merge_success = True
        except subprocess.CalledProcessError:
            print(f"\n[!] Merge conflict for {branch_name}. Branch preserved for manual resolution.")
            # We do NOT set merge_success to True, so the branch won't be deleted below

    # Cleanup logic:
    if merge_success:
        # 1. Remove the worktree directory
        subprocess.run(["git", "worktree", "remove", "--force", worktree_path], check=False)
        # 2. Delete the branch reference
        subprocess.run(["git", "branch", "-D", branch_name], check=False, capture_output=True)
        # 3. Failsafe cleanup if worktree remove didn't clear the directory
        if os.path.exists(worktree_path):
            shutil.rmtree(worktree_path)
    else:
        # If failed or conflict, we keep the worktree and branch for inspection.
        pass
        
    return merge_success