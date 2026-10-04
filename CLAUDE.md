# ephys_fm: working agreement for agents

Several people and agents work on this repository at the same time, each line of work in its own
git worktree. Follow this protocol before running any git command. New collaborators: start with
[`docs/onboarding.md`](docs/onboarding.md).

## Where to work

Paths are relative to your main checkout (your clone of the repository; the first entry of
`git worktree list`).

| Directory | Branch | Line of work |
|---|---|---|
| main checkout | `main`, always | Integration and packaging submissions only |
| `.claude/worktrees/track2` | `track2/<topic>`, `exp/<topic>` | Track 2 (BCI decoding) competition work |
| `.claude/worktrees/data-atlas` | `research/data-atlas` | Data atlas research track (vision: `docs/atlas/vision.md` on `research/data-atlas`, not yet on `main`) |

Each line of work writes run artifacts to its own subfolder of the shared `outputs/` and only reads
the shared `data/` caches.

Concluded: the Track 2 research loops A (`exp/fingerprint-model`) and B (`exp/expert-weights`) and
their integration into `submission.py` (`track2/integrate-fp-c3`) merged in #7, #8 and #9. Their
brief is `docs/handoff/track2-research-loops.md`; their local artifacts stay in
`outputs/t2-fingerprint/`, `outputs/t2-expert-weights/` and `outputs/t2-integration/`.

`git worktree list` shows the current set on your machine.

## Rules

1. Work only inside the worktree for your line of work. Start sessions there
   (`cd .claude/worktrees/<name> && claude`) or enter it with `EnterWorktree(path=...)`.
2. Never switch the main checkout off `main`, and never develop in it.
3. Never edit, move, commit or stash another worktree's files. The git stash is shared by all
   worktrees: don't use bare `git stash` / `git stash pop`; set work aside with a WIP commit.
4. Before every commit, run `git branch --show-current` and check it is your worktree's branch.
5. Branch names: `exp/<topic>`, `track<N>/<topic>`, `research/<topic>`, `setup/<topic>`,
   `docs/<topic>`. Integrate into `main` through a GitHub pull request.
6. A new line of work gets its own worktree, created from the root of the main checkout:
   `git worktree add .claude/worktrees/<name> -b <prefix>/<topic> main`. Then link the shared,
   git-ignored folders it needs to the main checkout's copies:
   `for d in data external .venv outputs submissions; do ln -s "$PWD/$d" .claude/worktrees/<name>/$d; done`.
   `.gitignore` covers `.claude/worktrees/` and these links. Never copy `data/` (30+ GB).
7. A new experiment within a line of work is a new branch inside that worktree
   (`git switch -c exp/<topic>`).
8. The competition venv (`.venv` in the main checkout) is shared through symlinks. Install
   research-only packages into a worktree's own venv (the atlas has one), never into the shared one.
9. After a branch is merged, remove its worktree: `git worktree remove .claude/worktrees/<name>`.
