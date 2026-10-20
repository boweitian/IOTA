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

Both languages live on `main`, in the same repository homepage. The Chinese and English pixel buttons jump to the corresponding section of the README. There is no language branch and no separate file-preview navigation.

Maintain the two text sources in `docs/translations/README.en.md` and `docs/translations/README.zh-CN.md`, then regenerate the homepage:

```bash
python scripts/build_readme.py
```

Commit the sources, generated README, and any artwork changes together on `main`. The two languages share the same artwork and presentation download URL. The `moonlight-*.svg` filenames identify the blue artwork and avoid reusing the previous image-cache URLs.

## Animated cover

Both language sections use `assets/brand/moonlit-video-cover.webp`, a looping
animation generated from the full `assets/brand/icon.mp4` clip. The source video
is preserved, and clicking the cover opens its GitHub file page.

The WebP retains the video's 864×480 resolution and full duration, samples 12
frames per second, and loops continuously. Regenerate it with an installed FFmpeg
or the optional `imageio-ffmpeg` package:

```bash
python scripts/build_cover.py
python scripts/build_readme.py
```

Pass `--ffmpeg /path/to/ffmpeg` when the executable is outside `PATH`.
