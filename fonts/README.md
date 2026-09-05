# Chinese name lettering

## Current: Long Cang

`long-cang-name.ttf` is the Google Fonts subset of Long Cang for **闫泽宇**
(U+95EB, U+6CFD, U+5B87). The CSS family alias is `Zeyu Signature`. The header
and footer use this handwriting with a slightly narrower proportion and more
space between the characters. A small optical kerning adjustment brings 宇
closer to 泽 so the handwritten shapes appear evenly spaced.

- Source: https://github.com/google/fonts/tree/main/ofl/longcang
- Subset: https://fonts.googleapis.com/css2?family=Long+Cang&text=%E9%97%AB%E6%B3%BD%E5%AE%87&display=swap
- License: SIL Open Font License 1.1; see `OFL-Long-Cang.txt`.

The name remains selectable, accessible text. The font is served locally and
restricted to these three Unicode characters by the CSS `unicode-range`.

## Unused earlier design: slender lettering

`zeyu-slender.woff2` is the earlier custom three-glyph design. It is retained
along with its source, `build-name-font.py`, but is no longer loaded by the site.
To rebuild it, install
`fonttools[woff]`, then run `python3 fonts/build-name-font.py` from the repo root.
This optional script does not run as part of the website.
