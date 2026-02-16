"""
Helper script to sync results from Colab to local repo.
Run this in Colab after training to prepare results for commit.
"""
import subprocess
import sys
from pathlib import Path

def main():
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("outputs").exists():
        print("Error: Run this from the verifoto-dl root directory")
        sys.exit(1)
    
    # Configure git
    email = input("Enter your git email: ").strip()
    name = input("Enter your git name: ").strip()
    
    if email and name:
        subprocess.run(["git", "config", "--global", "user.email", email])
        subprocess.run(["git", "config", "--global", "user.name", name])
        print("✓ Git configured")
    
    # Check for new results
    runs_dir = Path("outputs/runs")
    new_runs = []
    
    for run_path in runs_dir.iterdir():
        if run_path.is_dir() and run_path.name != ".gitkeep":
            # Check if already committed
            result = subprocess.run(
                ["git", "ls-files", str(run_path)],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                new_runs.append(run_path.name)
    
    if not new_runs:
        print("No new runs to commit")
        return
    
    print(f"\nFound {len(new_runs)} new run(s):")
    for run in new_runs:
        print(f"  - {run}")
    
    # Add and commit
    response = input("\nCommit these results? (y/n): ").strip().lower()
    if response == 'y':
        for run in new_runs:
            subprocess.run(["git", "add", f"outputs/runs/{run}"])
        
        commit_msg = f"Add results for {', '.join(new_runs)}"
        subprocess.run(["git", "commit", "-m", commit_msg])
        print("✓ Results committed")
        
        push = input("Push to GitHub? (y/n): ").strip().lower()
        if push == 'y':
            subprocess.run(["git", "push"])
            print("✓ Pushed to GitHub")
    else:
        print("Skipped commit")

if __name__ == "__main__":
    main()
