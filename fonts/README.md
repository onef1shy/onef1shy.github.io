# Typography

## Body type

Body text, publication titles, author lists, and controls use the
locally hosted Nunito rounded font. Body text uses weight 500; paper titles and
emphasized text use 700. Both upright and italic variable fonts are included, so
the page does not depend on which fonts the visitor has installed.
Publication authors use the italic face at 14.5px; Zeyu Yan is emphasized at
weight 750 in dark gray.

- Files: `nunito-latin.ttf` and `nunito-latin-italic.ttf`.
- Source: https://github.com/google/fonts/tree/main/ofl/nunito
- License: SIL Open Font License 1.1; see `OFL-Nunito.txt`.
- Subset: Latin U+0020–00FF, punctuation U+2000–206F, U+20AC, and U+2122,
  created with `fonttools` while retaining the variable weight axis.
- CSS stack: `Nunito, Verdana, sans-serif`.

## Navigation: Kalam

The navigation brand **Zeyu Yan / Homepage** and section links use Kalam Regular
at 20px and weight 400, with the CSS alias `Zeyu Navigation`.

- File: `kalam-navigation.ttf`, subset with `fonttools` to Basic Latin
  U+0020–007E, covering the brand and all navigation labels.
- Source: https://github.com/google/fonts/tree/main/ofl/kalam
- License: SIL Open Font License 1.1; see `OFL-Kalam.txt`.

## Current: Caveat and Ma Shan Zheng

- `caveat-name.ttf`: Caveat Medium (weight 500), subset to **Zeyu Yan**.
  The CSS alias is `Zeyu Handwritten`.
  Source: https://github.com/google/fonts/tree/main/ofl/caveat
  License: SIL Open Font License 1.1; see `OFL-Caveat.txt`.
- `ma-shan-zheng-name.ttf`: Ma Shan Zheng Regular, subset to **闫泽宇**.
  The CSS alias is `Zeyu Signature`.
  Source: https://github.com/google/fonts/tree/main/ofl/mashanzheng
  License: SIL Open Font License 1.1; see `OFL-Ma-Shan-Zheng.txt`.

Both names remain selectable text. The fonts are served locally and used only
for the header and footer names. The English font is a static instance of the
upstream variable font at weight 500. Both files are subset with `fonttools`
to the characters listed above.

## Retained previous design: Long Cang

`long-cang-name.ttf` is the Google Fonts subset of Long Cang for **闫泽宇**
(U+95EB, U+6CFD, U+5B87). The header
and footer previously used this handwriting with a slightly narrower proportion and more
space between the characters. A small optical kerning adjustment brings 宇
closer to 泽 so the handwritten shapes appear evenly spaced.

- Source: https://github.com/google/fonts/tree/main/ofl/longcang
- Subset: https://fonts.googleapis.com/css2?family=Long+Cang&text=%E9%97%AB%E6%B3%BD%E5%AE%87&display=swap
- License: SIL Open Font License 1.1; see `OFL-Long-Cang.txt`.

This earlier font is retained but is no longer loaded by the page.

## Unused earlier design: slender lettering

`zeyu-slender.woff2` is the earlier custom three-glyph design. It is retained
along with its source, `build-name-font.py`, but is no longer loaded by the site.
To rebuild it, install
`fonttools[woff]`, then run `python3 fonts/build-name-font.py` from the repo root.
This optional script does not run as part of the website.
