# ModularEIT wiki site

[Quartz v4](https://quartz.jzhao.xyz) setup that renders the Obsidian vault in
`../markdown` as a static site. It is taken from the Category Theory ML Wiki and keeps
its custom transformers:

- `quartz/plugins/transformers/tikz.ts` renders ```` ```tikz ```` blocks (Obsidian
  *inline-tikz* format) to SVG at build time with `node-tikzjax`, with a light and a
  dark variant. Results are cached in `.tikz-cache/` (committed, so CI only renders new
  diagrams).
- `quartz/plugins/transformers/tabs.ts` renders ```` ````tabs ```` blocks of the
  Obsidian *Markdown Tabs* plugin.
- Math is rendered with KaTeX; `Plugin.Latex({ renderEngine: "typst" })` in
  `quartz.config.ts` switches to Typst.

## Build

```bash
cd site
npm ci
npx quartz build -d ../markdown --serve   # preview at http://localhost:8080
```

`docs/make.jl` builds the wiki into `docs/build/wiki/` after the Documenter pages, so
both are deployed together (`BUILD_WIKI=false julia --project=docs docs/make.jl` skips it).

## Editing in Obsidian

Open `markdown/` as a vault. It enables the community plugins *Wypst* (Typst math),
*Inline TikZ* and *Markdown Tabs*; their bundles are not committed, so install them
once via *Settings → Community plugins → Browse*.
