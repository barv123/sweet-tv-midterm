# SWEET.TV Midterm

Workspace for the SWEET.TV midterm data-analysis project.

## Project structure

```text
data/
  raw/       Original datasets (local only; ignored by Git)
  src/       Analysis scripts
  reports/   Written analysis and reports
  results/   Results intended for sharing
  sessions/  OpenCode session exports (local only; ignored by Git)
```

The existing directory layout is preserved. Placeholder files keep empty source,
report, and result directories in Git.

## Getting started

```sh
git clone https://github.com/barv123/sweet-tv-midterm.git
cd sweet-tv-midterm
```

Create `data/raw/` locally and place the original datasets there. Create
`data/sessions/` if needed for local session exports. Neither directory is
included in the repository.

No analysis code or dependency configuration has been added yet. Add scripts
under `data/src/` and document their dependencies and usage as they are developed.
Only publish results and reports suitable for sharing; keep credentials in local,
ignored environment files.
