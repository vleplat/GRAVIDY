#!/usr/bin/env python3
"""
Repository Setup Script for GRAVIDY

This script helps you set up the GRAVIDY repository with proper Git configuration.
Run this after creating the GitHub repository.
"""

import os
import subprocess
import sys

def run_command(cmd, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_git():
    """Check if git is installed."""
    success, _, _ = run_command("git --version", check=False)
    return success

def setup_git_repo(repo_url=None):
    """Initialize and set up the git repository."""
    print("🔧 Setting up Git repository...")
    
    # Check if git is installed
    if not check_git():
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Initialize git if not already done
    if not os.path.exists(".git"):
        success, _, err = run_command("git init")
        if not success:
            print(f"❌ Failed to initialize git: {err}")
            return False
        print("✅ Initialized Git repository")
    
    # Add files
    success, _, err = run_command("git add .")
    if not success:
        print(f"❌ Failed to add files: {err}")
        return False
    
    # Initial commit
    success, _, err = run_command('git commit -m "Initial commit: GRAVIDY optimization framework"')
    if not success:
        print(f"⚠️  Commit failed (files may already be committed): {err}")
    else:
        print("✅ Created initial commit")
    
    # Add remote if provided
    if repo_url:
        success, _, err = run_command(f"git remote add origin {repo_url}")
        if not success:
            print(f"⚠️  Failed to add remote (may already exist): {err}")
        else:
            print(f"✅ Added remote origin: {repo_url}")
        
        # Push to remote
        success, _, err = run_command("git push -u origin main")
        if not success:
            # Try master branch
            success, _, err = run_command("git push -u origin master")
        
        if success:
            print("✅ Pushed to remote repository")
        else:
            print(f"⚠️  Failed to push to remote: {err}")
            print("   You may need to push manually:")
            print("   git branch -M main")
            print("   git push -u origin main")
    
    return True

def main():
    """Main setup function."""
    print("🚀 GRAVIDY Repository Setup")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("README.md"):
        print("❌ This doesn't appear to be the GRAVIDY directory.")
        print("   Please run this script from the GRAVIDY root directory.")
        return
    
    print("📂 Current directory looks good!")
    
    # Get repository URL
    repo_url = input("\n🔗 Enter your GitHub repository URL (or press Enter to skip): ").strip()
    if repo_url and not repo_url.startswith("https://github.com"):
        print("⚠️  URL should start with https://github.com")
        repo_url = input("🔗 Enter the correct URL (or press Enter to skip): ").strip()
    
    # Setup git
    if setup_git_repo(repo_url if repo_url else None):
        print("\n🎉 Repository setup completed!")
        print("\nNext steps:")
        print("1. Visit your GitHub repository")
        print("2. Update the repository description")
        print("3. Add topics: 'optimization', 'python', 'numerical-computing'")
        if not repo_url:
            print("4. Add remote and push:")
            print("   git remote add origin https://github.com/yourusername/GRAVIDY.git")
            print("   git branch -M main")
            print("   git push -u origin main")
    else:
        print("\n❌ Repository setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
