# Git Save Point Instructions

## Quick Save (Commit Current Changes)

### Step 1: Check what has changed
```bash
git status
```
This shows you which files have been modified, added, or deleted.

### Step 2: Add files to staging
```bash
# Add all changes
git add .

# OR add specific files only
git add src/config.py src/data_loader.py analysis/plots_variances.py
```

### Step 3: Commit with a descriptive message
```bash
git commit -m "Your descriptive message here"
```

**Example commit messages:**
- `"Add BST price to variance comparison plots"`
- `"Fix time series ordering to start from 2020"`
- `"Implement dual x-axis (date + step) in data exploration"`
- `"Add sort_index to ensure chronological data order"`

### Step 4: (Optional) Push to remote repository
```bash
git push origin main
```
*Note: Only if you have a remote repository set up*

---

## Complete Workflow Example

```bash
# 1. See what changed
git status

# 2. Add all changes
git add .

# 3. Commit with message
git commit -m "Integrate BST price data and fix time series ordering"

# 4. (Optional) Push to GitHub/remote
git push origin main
```

---

## View Your Save Points (Commit History)

```bash
# See recent commits
git log --oneline

# See last 5 commits with details
git log -5

# See commits with file changes
git log --stat
```

---

## Go Back to a Previous Save Point

### Option 1: Temporarily view old code (safe)
```bash
# Find the commit hash from git log
git log --oneline

# Check out that commit (e.g., abc1234)
git checkout abc1234

# Go back to latest
git checkout main
```

### Option 2: Undo last commit but keep changes
```bash
git reset --soft HEAD~1
```

### Option 3: Completely revert to previous commit (CAREFUL!)
```bash
# This discards all uncommitted changes
git reset --hard HEAD~1
```

---

## Create a Branch for Experiments

```bash
# Create and switch to new branch
git checkout -b experiment-agent-heterogeneity

# Work on your code...

# Commit changes
git add .
git commit -m "Test agent heterogeneity feature"

# Switch back to main
git checkout main

# If experiment worked, merge it
git merge experiment-agent-heterogeneity
```

---

## Quick Reference Card

| Command | What it does |
|---------|--------------|
| `git status` | Show what changed |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Save a checkpoint |
| `git log --oneline` | View save history |
| `git diff` | See exact changes before committing |
| `git checkout <file>` | Discard changes to a file |
| `git reset --soft HEAD~1` | Undo last commit, keep changes |

---

## Best Practices

1. **Commit often** - Every time you complete a logical unit of work
2. **Write clear messages** - Future you will thank you
3. **Check before committing** - Use `git status` and `git diff`
4. **Don't commit broken code** - Make sure it runs first
5. **One feature per commit** - Easier to track and revert

---

## Current Project Status

**Repository:** `llm-asset-market-herding`

**Recent major changes:**
- ✅ Integrated BST CEF NAV data as true value series
- ✅ Added BST market price to variance comparison plots
- ✅ Fixed time series ordering (2020→2025)
- ✅ Implemented dual x-axis (date + step) in exploration plots
- ✅ Created application draft document

**Recommended commit message for current state:**
```bash
git add .
git commit -m "Complete BST data integration: add price plots, fix chronological ordering, add dual x-axis"
```

---

*Last updated: December 2025*
