# ModularEIT wiki site

The Obsidian vault in `../markdown` is rendered as a static site with
[MathStruct's Quartz](https://github.com/MathStruct/quartz): Quartz v4 plus the plugins the
MathStruct sites share, among them

- ```` ```tikz ```` blocks (Obsidian *inline-tikz* format) rendered to SVG at build time, with
  a light and a dark variant. Results are cached in `.tikz-cache/` (committed, so CI only
  renders new diagrams);
- ```` ````tabs ```` blocks of the Obsidian *Markdown Tabs* plugin;
- math with KaTeX; a page renders with Typst instead with `math: typst` in its frontmatter,
  and a ```` ```typst ```` block always does.

This directory only holds what is particular to this site: `quartz.config.ts`,
`quartz.layout.ts` and the TikZ cache.

## Build

```bash
site/build.sh            # -> site/public/
site/build.sh --serve    # preview at http://localhost:8080
```

Needs git and Node ≥ 22. The shared Quartz is cloned into `site/.quartz/` (gitignored) on the
first run. `docs/make.jl` builds the wiki into `docs/build/wiki/` after the Documenter pages,
so both are deployed together (`BUILD_WIKI=false julia --project=docs docs/make.jl` skips it).

## Editing in Obsidian

Open `markdown/` as a vault. It enables the community plugins *Wypst* (Typst math),
*Inline TikZ* and *Markdown Tabs*; their bundles are not committed, so install them
once via *Settings → Community plugins → Browse*.
