# ephys_fm: working agreement for agents

Several agents work on this repository at the same time, each in its own git worktree. Follow
this protocol before running any git command.

## Where to work

| Directory | Branch | Line of work |
|---|---|---|
| `/home/promit/Documents/ephys_fm` (main checkout) | `main`, always | Integration and packaging submissions only |
| `.claude/worktrees/track2` | currently `exp/dreyer-eog` | Track 2 (BCI decoding) competition work |
| `.claude/worktrees/data-atlas` | `research/data-atlas` | Data atlas research track ([vision](docs/atlas/vision.md)) |
| `.claude/worktrees/t2-fingerprint` | `exp/fingerprint-model` | Track 2 autoresearch loop A: a better per-window fingerprint (person ID) model |
| `.claude/worktrees/t2-expert-weights` | `exp/expert-weights` | Track 2 autoresearch loop B: learned weights for combining per-person EEGNet and Riemannian experts |
| `.claude/worktrees/t2-integration` | `track2/integrate-fp-c3` | Track 2: loop A's fingerprint and loop B's combiner in `submission.py`; contract check |

The two Track 2 research loops run in parallel with a shared heartbeat. Each writes run artifacts
to its own subfolder of the shared `outputs/` (`outputs/t2-fingerprint/`, `outputs/t2-expert-weights/`)
and only reads the shared `data/` caches.

`git worktree list` shows the current set.

## Rules

1. Work only inside the worktree for your line of work. Start sessions there
   (`cd .claude/worktrees/<name> && claude`) or enter it with `EnterWorktree(path=...)`.
2. Never switch the main checkout off `main`, and never develop in it.
3. Never edit, move, commit or stash another worktree's files. The git stash is shared by all
   worktrees: don't use bare `git stash` / `git stash pop`; set work aside with a WIP commit.
4. Before every commit, run `git branch --show-current` and check it is your worktree's branch.
5. Branch names: `exp/<topic>`, `track<N>/<topic>`, `research/<topic>`, `setup/<topic>`,
   `docs/<topic>`. Integrate into `main` through a GitHub pull request.
6. A new line of work gets its own worktree, created from the main checkout:
   `git worktree add .claude/worktrees/<name> -b <prefix>/<topic> main`. Then link the shared,
   git-ignored folders it needs to the main checkout's copies, one command per folder
   (`data`, `external`, `.venv`, `outputs`, `submissions`):
   `ln -s /home/promit/Documents/ephys_fm/<dir> .claude/worktrees/<name>/<dir>`.
   `.git/info/exclude` already lists `.claude/worktrees/` and those names. Never copy `data/` (30+ GB).
7. A new experiment within a line of work is a new branch inside that worktree
   (`git switch -c exp/<topic>`).
8. The competition venv (`.venv` in the main checkout) is shared through symlinks. Install
   research-only packages into a worktree's own venv (the atlas has one), never into the shared one.
9. After a branch is merged, remove its worktree: `git worktree remove .claude/worktrees/<name>`.
