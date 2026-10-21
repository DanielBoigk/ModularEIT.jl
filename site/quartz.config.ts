import { siteConfig } from "./mathstruct/config"

// The plugins, theme and fixes are MathStruct's shared Quartz preset
// (https://github.com/MathStruct/quartz, mathstruct/config.ts); this file only says what is
// particular to the ModularEIT wiki.
export default siteConfig({
  pageTitle: "ModularEIT Wiki",
  baseUrl: "danielboigk.github.io/ModularEIT.jl/dev/wiki",
  ignorePatterns: ["templates"],
})
