# Repository layout

The root contains the project README and Git configuration. Code, media, presentations, and historical notes live in directories.

| Directory | Contents |
|---|---|
| `assets/brand/` | Original cover and generated pixel SVG headline/buttons |
| `docs/presentation/` | Downloadable project presentation |
| `docs/translations/` | Maintained Chinese README source |
| `docs/history/` | Historical designs, review, and previous workspace README |
| `docs/archive/website/` | Preserved original website and its logo |
| `scripts/` | Earlier command-line tools and README asset builders |
| `scripts/slurm/` | Earlier cluster launch scripts |
| `requirements/` | Earlier core Python dependency list |
| `iota/` | Earlier core mapping and evidence modules |
| `geometry_mvp/` | Measured offline geometry explorer |
| `steering_lab/` | Search workspace and activation-steering engine |
| `network_pilot/` | Navigator experiments, scoring, and independent audits |
| `tests/` | Core module checks |

Run the moved command-line tools from the repository root, for example:

```bash
python scripts/discover.py --help
python scripts/analyze_campaign.py --help
sbatch scripts/slurm/run_discovery.sbatch
```

Experimental run directories and their frozen source snapshots are retained locally and excluded from Git. The Search/Navigator computational source paths remain in place.

## README presentation

GitHub strips custom README CSS and scripts. The headline and navigation buttons therefore use repository-native SVG pixel lettering, with no web-font dependencies:

```bash
python scripts/build_readme_assets.py
```

The English repository home is `main`. The Chinese button opens the `zh-CN` repository home, where only the root README differs. It opens the same repository layout and localized introduction instead of a single-file preview. Both languages link the presentation to GitHub's raw-download endpoint.

Maintain the translation in `docs/translations/README.zh-CN.md`. After committing shared changes to `main`, update the language branch in a separate worktree based on the latest `main`:

```bash
git worktree add -b zh-home-update /tmp/iota-zh-home main
python scripts/localize_readme.py --output /tmp/iota-zh-home/README.md
git -C /tmp/iota-zh-home add README.md
git -C /tmp/iota-zh-home commit -m "Update Chinese repository home"
```

Publish that commit to `zh-CN` while preserving branch history. If `zh-CN` already exists, base the update on its current tip and merge the latest `main` before regenerating the root README. Avoid force-pushing shared branch history.
