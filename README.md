<p align="center">
  <a href="https://github.com/boweitian/IOTA/blob/main/assets/brand/icon.mp4"><img src="assets/brand/moonlit-video-cover.webp" alt="IOTA animated cover — original MP4 video" width="100%"></a>
</p>

<p align="center">
  <img src="assets/brand/moonlight-headline.svg?v=moonlight-soft-1" alt="An inner world for intelligence that people can understand and shape." width="680">
</p>

<table align="center">
  <tr>
    <td align="center" valign="top"><a href="https://bowei.netlify.app/iota/"><img src="assets/brand/moonlight-demo.svg" alt="Demo" height="36"></a></td>
    <td align="center" valign="top"><a href="https://github.com/boweitian/IOTA/raw/refs/heads/main/docs/presentation/IOTA-general.pptx"><img src="assets/brand/moonlight-presentation.svg" alt="Presentation" height="36"></a></td>
    <td align="center" valign="top"><a href="https://bowei.netlify.app/blog/"><img src="assets/brand/moonlight-blog.svg" alt="Blog" height="36"></a></td>
    <td align="center" valign="top">
      <details>
        <summary><picture><img src="assets/brand/moonlight-language.svg" alt="Language" height="36" align="absmiddle"></picture></summary>
        <p>
          <a href="https://github.com/boweitian/IOTA"><img src="assets/brand/moonlight-english.svg" alt="English" height="36"></a><br>
          <a href="https://github.com/boweitian/IOTA/blob/main/docs/translations/README.zh-CN.md"><img src="assets/brand/moonlight-chinese.svg" alt="简体中文" height="36"></a>
        </p>
      </details>
    </td>
  </tr>
</table>

IOTA connects human intentions to testable changes inside a model. We study how to understand behavior, shape internal concept geometry, and grow alongside a persistent digital individual.

One foundation, **Concept Geometry**, supports two product paths:

| | IOTA Navigation · Navigator | IOTA Search |
|---|---|---|
| Central question | Which internal settings help with this input? | How can people create and shape one persistent digital individual? |
| Workflow | Select steering within an existing geometry | Connect internal editing with ongoing interaction |
| Evidence | Task benefits, costs, and side effects on unseen inputs | Inspectable behavior changes, continuity, and recoverable configurations |

## IOTA Navigation

**IOTA Navigator selects useful steering for each input within a fixed Concept Geometry.**

The workflow is **task input → direction selection → intervention and generation → evaluation**. Geometry stays fixed at this stage. The Navigator chooses a direction, a strength, or a combination; **no intervention is a valid choice**.

An intervention can repair one answer while damaging another. The goal is to organize the available geometry around the current task and input, and measure whether this selection improves performance.

Validation compares the base model and optimized prompts, static steering, random selection, and dynamic navigation. Useful gains must persist on unseen inputs, alongside measurements of added latency, compute cost, and degradation of other capabilities.

The [`network_pilot/`](network_pilot) code contains direction sweeps, routing and ensemble experiments, per-item feasible configurations, answer parsing, diagnostics, and independent audits. See the [ANLI R3 experiment note](docs/ANLI_R3.md) for the latest exploratory scan. Label-aware coverage is an upper bound for the tested candidates, not a deployed Navigator score.

## IOTA Search

**IOTA Search explores the creation of a digital individual: an agent with a name, a persistent identity, and an inner structure that people can shape.**

Two interfaces stay connected to the same individual:

- **Inner World** — edit directions and strengths, create and test custom steering, and save, compare, or restore configurations.
- **Outer World** — converse with the agent, test scenarios, observe the effects of internal edits, and build experiences across interactions.

The development loop is **create → interact → validate → grow**. Internal changes should be inspectable, attributable to a version, and recoverable. A new direction becomes useful through repeatable behavioral validation, including its effects on other abilities.

The [`steering_lab/`](steering_lab) workspace implements the editing and conversation interfaces, activation interventions, direction construction, scenario comparisons, and configuration workflows. Long-term identity consistency, accumulated experience, and reliable individual growth remain research goals requiring further evidence.

## Shared foundation: Concept Geometry

Concept Geometry organizes actionable directions such as caution, courage, empathy, and curiosity. Each direction is tied to a model, layer, intervention strength, construction procedure, scope, evaluation, and version.

The current implementation primarily uses **activation steering**: it adds a direction vector to a selected model layer during inference while keeping model weights unchanged. A direction name describes its construction intent; its actual effect depends on the model, layer, input, and strength. Visual projections help explore the space, while interventions use the original high-dimensional vectors.

Navigator contributes input-to-direction utility data and measurements of benefits and costs. Search contributes individual contexts, expansion needs, new directions, and behavioral feedback. Their shared assets are evaluated geometry and versioned individual configurations.

## Getting started

### Hosted demo

**[https://bowei.netlify.app/iota/](https://bowei.netlify.app/iota/)**

Use the hosted workspace to explore the product interfaces. Live generation requires a connected model service; static hosting itself does not provide GPU inference.

### Lightweight offline geometry explorer

The repository includes an earlier measured geometry prototype with small sample artifacts. It can be built and explored without downloading model weights:

```bash
git clone https://github.com/boweitian/iota.git
cd iota
python -m venv .venv
source .venv/bin/activate
pip install numpy
python geometry_mvp/build.py
python geometry_mvp/serve.py --port 8765
```

Open **http://127.0.0.1:8765**. This explorer replays measured responses and exports direction settings. Its measurements belong to the earlier Qwen2.5 prototype; the current Search/Navigator experiments use their own model-bound artifacts.

### Full research workspace

Install the pinned steering environment in a separate virtual environment:

```bash
python -m venv steering_lab/.venv
source steering_lab/.venv/bin/activate
pip install -r steering_lab/requirements.txt
```

Current 9B experiments use Qwen3.5-9B. Live inference and experiment reproduction require the matching model snapshot, prepared inputs, and geometry/run artifacts. Experimental scripts record model revisions, source hashes, and protocols; some launch scripts also encode the original Slurm account and filesystem paths and need adaptation to your environment.

Useful entry points:

| Path | Purpose |
|---|---|
| [`steering_lab/engine.py`](steering_lab/engine.py) | Hidden-state measurement and activation intervention |
| [`steering_lab/studio_geometry.py`](steering_lab/studio_geometry.py) | Construct and validate the studio geometry |
| [`steering_lab/interactive.py`](steering_lab/interactive.py) | Interactive model portal and GPU-session workflow |
| [`steering_lab/agent.js`](steering_lab/agent.js) | Digital-individual editing and conversation UI |
| [`network_pilot/fullspace_detailed.py`](network_pilot/fullspace_detailed.py) | Frozen detailed-answer direction scan |
| [`network_pilot/fullspace_analysis.py`](network_pilot/fullspace_analysis.py) | Coverage, fixed-direction comparisons, and diagnostic annotation |
| [`network_pilot/audit_fullspace_detailed.py`](network_pilot/audit_fullspace_detailed.py) | Independent scan verification |
| [`geometry_mvp/`](geometry_mvp) | Lightweight measured geometry explorer |
| [`iota/`](iota) | Earlier mapping, exploration, and evidence prototypes |
| [`scripts/`](scripts) | Command-line tools, documentation assets, and Slurm launchers |
| [`requirements/`](requirements) | Core prototype dependencies |

Large generated runs, downloaded benchmarks, model checkpoints, local credentials, live conversations, and GPU-session state are excluded from Git. The small `geometry_mvp/data*` sample artifacts are included for the offline explorer. Historical workspace notes are preserved in [`docs/history/README_LEGACY.md`](docs/history/README_LEGACY.md); their local result links refer to separately generated experiment artifacts.

## Development direction

The first stage validates Navigator comparisons and the Search model workflow. Later stages investigate identity across sessions, version comparison and rollback, long-term memory, and expansion of an individual's geometry.

Navigator validates the value of internal control. Search explores how people can create and grow with a digital individual. Progress is measured through behavioral evidence, task outcomes, and persistent effects of validated changes.
