import { siteLayout } from "./mathstruct/layout"

export const { sharedPageComponents, defaultContentPageLayout, defaultListPageLayout } =
  siteLayout({
    footerLinks: {
      GitHub: "https://github.com/DanielBoigk/ModularEIT.jl",
      "API docs": "https://danielboigk.github.io/ModularEIT.jl/dev/",
    },
  })
