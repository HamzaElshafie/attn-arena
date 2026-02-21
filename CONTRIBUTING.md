# Contributing to `attn-arena`

This repository follows a professional, PR-first workflow even when its only me at the moment 😅.

## Workflow Principles

- Every meaningful change is made on a branch and merged via pull request.
- Each PR has one cohesive goal and clear acceptance criteria.
- CI must pass before merge.
- Prefer small, reviewable PRs over large mixed-scope PRs.
- Document design-impacting decisions in PR descriptions.

## Branch Strategy

Create branches from `main` using this pattern:

- `<area>-<goal>`

Examples:

- `contracts-attention-model`
- `registries-core`
- `config-schema-v1`
- `mha-llama-baseline`
- `fix-ci-mypy-config`

### When to open a new branch

Open a new branch when a change:

- touches multiple files with a single coherent objective,
- takes more than a short quick fix,
- changes behavior, interfaces, configuration, or tests.

Do not open a new branch for tiny typo-only changes unless you intentionally
bundle them with the active scoped branch.

## Commit Standards

- Use focused commits with imperative messages.
- Keep each commit logically atomic.
- Avoid mixing refactors and behavior changes in one commit when possible.

Recommended commit style:

- `feat(attention): add AttentionModule and KVCache protocols`
- `test(registry): cover duplicate registration errors`
- `chore(ci): add mypy check to pull request workflow`
- `docs(contributing): define branch and merge policy`

## Pull Request Policy

Every branch merges through a PR, including solo work.

A PR should include:

- clear problem statement,
- scope (in/out),
- implementation summary,
- test evidence,
- risks and follow-ups.

## Review Checklist (Self-Review Required)

Before requesting final merge:

- [ ] Scope is single-purpose and aligns with PR title.
- [ ] Public interfaces are typed and consistent.
- [ ] No unrelated file changes are included.
- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] `mypy attn_arena` passes.
- [ ] `pytest -m "not gpu and not correctness"` passes.
- [ ] New behavior is covered by tests or explicitly justified.
- [ ] PR description is complete and updated after final changes.

## Merge Policy

Merge method:

- Use **Squash and merge** for feature/fix branches.

Merge requirements:

- CI green on latest commit.
- Self-review checklist completed.
- No unresolved TODO comments that affect correctness.

Post-merge:

- Delete merged branch.
- Ensure `main` remains releasable.

## Labels and Milestones (Optional but Recommended)

Suggested labels:

- `area:attention`
- `area:models`
- `area:benchmarking`
- `area:distributed`
- `area:infra`
- `type:feature`
- `type:bug`
- `type:refactor`
- `type:docs`