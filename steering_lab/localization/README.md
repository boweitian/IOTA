# English frontend for the public IOTA workspace

This package adds English presentation to the existing public application at
`https://bowei.netlify.app/iota/`. It covers the workspace, Search, Navigator,
standalone viewers, historical pages, dynamic labels, accessibility attributes,
placeholders, status messages, errors, and canvas labels.

The original application JavaScript, embedded JSON, experiment files, model
prompts, requests, configuration values, storage keys, and conversation content
are retained. Original evidence and editable user content remain in their native
language. Built-in display names can be transliterated without renaming the saved
agent. Implicit option values are pinned before translating their labels.

`en.json` is the shared display catalog. `frontend-en.js` applies it to UI text
and observes newly rendered nodes. Native evidence and conversation elements are
explicitly excluded. `frontend-en.css` allows longer English labels to fit the
existing responsive layout. The content hash versions both frontend assets.

To update a **copy of a currently published Netlify snapshot**, containing the
public `deploy-manifest.json`:

```sh
python steering_lab/localization/build.py /path/to/copied/public/iota
python steering_lab/localization/verify.py /path/to/original/iota /path/to/copied/public/iota
```

The builder updates all HTML entry points and their public manifest hashes.
It preserves the backend in the manifest. The `/iota/` asset prefix targets this
public deployment; this is a deployment packaging step, not a model build step.
The personal site's existing `scripts/restore-iota.mjs` preserves the published
English files, and verifies their hashes, on subsequent Hugo builds.

Before publication, compare requests and storage with localization enabled and
disabled, exercise configuration export/import and the asynchronous chat flow,
and check standalone pages, lazy workspace modules, both themes, filters,
dynamic intervention labels, and desktop/mobile layouts. Evidence, archived
prompts, user messages, and raw model responses must remain byte-identical.
