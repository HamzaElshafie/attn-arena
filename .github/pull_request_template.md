## Summary

Describe what this PR changes and why.

## Problem

What problem does this PR solve?

## Implementation

List the main technical changes.

## Testing

Provide concrete evidence (commands and results).

```bash
ruff check .
ruff format --check .
mypy attn_arena
pytest -m "not gpu and not correctness"
```

## Risks

List behavioral, compatibility, or performance risks.

## Follow-ups

List intentional future work (if any).

## Self-Review Checklist

- [ ] Single-purpose PR with coherent scope.
- [ ] Branch name follows `<area>-<goal>`.
- [ ] Public interfaces/types are explicit and documented.
- [ ] No unrelated file changes included.
- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] `mypy attn_arena` passes.
- [ ] `pytest -m "not gpu and not correctness"` passes.
- [ ] If behavior changed, tests were added/updated.
- [ ] CI is green on latest commit.
