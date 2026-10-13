# Paper website

The paper homepage is https://ymsunny.github.io/REAL-MT-Vuln/.
Its complete source is in `docs/` on the `main` branch.

## Editing and publishing

GitHub Pages publishes the `main` branch's `/docs` directory. Once configured,
commits to that branch automatically deploy the updated website.

The website is static and requires no package installation or build step.
Open `docs/index.html` directly for a local preview.

- `docs/index.html`: content, author information, initial charts, tables and citation.
- `docs/styles.css`: colors, layout and responsive styling.
- `docs/data.js`: the paper's context conditions, Table 3 Fidelity scores and Table 4 mitigation results.
- `docs/script.js`: context switching, charts, figure viewer and citation copying.
- `docs/assets/`: conference PDF, original figures, local font, BibTeX and social preview.

Keep `data.js` and the initial HTML results consistent when changing numbers.
Keep the on-page citation and downloadable BibTeX consistent too.
All numeric results on the homepage follow the AAAI conference paper;
Table 3's C column is COMET-22, not Context Adoption Rate.

You can edit the files directly on GitHub or in a local clone. GitHub edits
do not automatically update a local clone: run `git pull --ff-only` before
editing locally. Commit and push local changes to publish them.
