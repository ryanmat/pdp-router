# Contributing

This repository is a **one-way mirror**. Development happens in a private
upstream; a scanner-gated sync script copies the package here. Commits land
as `sync from upstream <sha>` and the mirror's history is write-only from
that pipeline.

What that means in practice:

- **Issues are welcome.** Bug reports and design questions get read.
- **PRs may be cherry-picked, not merged.** If a change is accepted, it is
  applied upstream and arrives here in a later sync commit with credit in
  the commit message. Your PR will be closed, not merged -- nothing personal,
  it is how the mirror stays consistent.
- **No support contract.** This is a reference implementation; see the
  README's "What this is" section.

## Dev setup

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
```

Tests are hermetic: no network, no live databases, no credentials needed.
