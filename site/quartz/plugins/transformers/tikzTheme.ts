/**
 * Derives a dark-mode variant of a (light-mode) TikZ SVG by remapping its colours:
 * greys are mapped onto the dark theme's foreground→background ramp (black becomes the
 * theme's text colour, white becomes the page background), coloured values keep their
 * hue/saturation but have their lightness inverted.
 */
export interface DarkPalette {
  foreground: string
  background: string
}

type RGB = [number, number, number]

const NAMED: Record<string, RGB> = {
  black: [0, 0, 0],
  white: [255, 255, 255],
}

function parseColor(value: string): RGB | null {
  const v = value.trim().toLowerCase()
  if (v in NAMED) return NAMED[v]
  let m = v.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/)
  if (m) {
    const hex = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1]
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as RGB
  }
  m = v.match(/^rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/)
  if (m) return [Number(m[1]), Number(m[2]), Number(m[3])]
  return null
}

function toHex([r, g, b]: RGB): string {
  return "#" + [r, g, b].map((c) => Math.round(Math.min(255, Math.max(0, c))).toString(16).padStart(2, "0")).join("")
}

function rgbToHsl([r, g, b]: RGB): [number, number, number] {
  r /= 255
  g /= 255
  b /= 255
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  if (max === min) return [0, 0, l]
  const d = max - min
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  const h =
    max === r ? ((g - b) / d + (g < b ? 6 : 0)) / 6 : max === g ? ((b - r) / d + 2) / 6 : ((r - g) / d + 4) / 6
  return [h, s, l]
}

function hslToRgb([h, s, l]: [number, number, number]): RGB {
  if (s === 0) return [l * 255, l * 255, l * 255]
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const hue = (t: number) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  return [hue(h + 1 / 3) * 255, hue(h) * 255, hue(h - 1 / 3) * 255]
}

function mapColor(value: string, fg: RGB, bg: RGB): string {
  const rgb = parseColor(value)
  if (!rgb) return value // none, currentColor, url(#…), unknown names: leave alone
  const [h, s, l] = rgbToHsl(rgb)
  if (s < 0.05) {
    // grey: black → foreground, white → background, linear in between
    return toHex([0, 1, 2].map((i) => fg[i] + (bg[i] - fg[i]) * l) as RGB)
  }
  return toHex(hslToRgb([h, s, 1 - l]))
}

const COLOR_PROPS = "fill|stroke|color|stop-color|flood-color|lighting-color"

export function darkVariant(svg: string, palette: DarkPalette): string {
  const fg = parseColor(palette.foreground) ?? [235, 235, 236]
  const bg = parseColor(palette.background) ?? [22, 22, 24]
  const attrRe = new RegExp(`(\\s(?:${COLOR_PROPS}))="([^"]*)"`, "g")
  const styleRe = new RegExp(`((?:^|;)\\s*(?:${COLOR_PROPS})\\s*:\\s*)([^;]+)`, "g")

  let out = svg
    .replace(attrRe, (_, attr, val) => `${attr}="${mapColor(val, fg, bg)}"`)
    .replace(/(\sstyle)="([^"]*)"/g, (_, attr, style: string) =>
      `${attr}="${style.replace(styleRe, (_m, prefix, val) => prefix + mapColor(val, fg, bg))}"`,
    )

  // Elements without an explicit fill (e.g. text) inherit SVG's default black: give the
  // root the foreground colour instead.
  const rootTag = out.match(/<svg\b[^>]*>/)?.[0] ?? ""
  if (!/\sfill=/.test(rootTag)) out = out.replace(/<svg\b/, `<svg fill="${toHex(fg)}"`)

  // Both variants live in the same document, so keep ids unique.
  const ids = [...out.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])
  for (const id of ids) {
    const esc = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    out = out
      .replace(new RegExp(`(\\sid=")${esc}"`, "g"), `$1${id}-dark"`)
      .replace(new RegExp(`url\\(#${esc}\\)`, "g"), `url(#${id}-dark)`)
      .replace(new RegExp(`(href=")#${esc}"`, "g"), `$1#${id}-dark"`)
  }
  return out
}
