import subprocess
import os
import shutil
import git

def create_parallel_worktree(branch_name: str) -> str:
    """Creates a new git branch and checks it out in an isolated worktree directory."""
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
    subprocess.run(["git", "worktree", "add", "-b", branch_name, worktree_path, "main"], check=True, capture_output=True)
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

    # Always cleanup the worktree directory (the folder) to save space
    subprocess.run(["git", "worktree", "remove", "--force", worktree_path], check=False)
    
    # Delete the branch reference ONLY if we merged successfully OR if the build failed (abandoned)
    # If success=True but merge_success=False (conflict), we keep the branch.
    if merge_success or not success:
        subprocess.run(["git", "branch", "-D", branch_name], check=False, capture_output=True)
    
    # Failsafe cleanup if worktree remove didn't clear the directory
    if os.path.exists(worktree_path):
        shutil.rmtree(worktree_path)
        
    return merge_success