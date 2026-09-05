"""Build the original three-character, slender-gold-inspired name lettering.

Optional rebuild: python3 -m pip install 'fonttools[woff]'
                 python3 fonts/build-name-font.py
Coordinates are in a 1000-unit drawing space, with y pointing down.
"""

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.svgLib.path import parse_path


def horizontal(x, y, end_x, end_y):
    """A fine rising horizontal with a small angled entry and pointed finish."""
    return (
        f"M {x - 12} {y - 16} L {x + 19} {y - 5} "
        f"L {end_x - 29} {end_y - 5} L {end_x - 11} {end_y - 24} "
        f"L {end_x + 24} {end_y + 7} L {end_x - 3} {end_y + 17} "
        f"L {x + 16} {y + 13} Z"
    )


# These outlines are drawn for this site, not extracted from a third-party font.
OUTLINES = {
    "闫": [
        # 门: pointed dot, left stem, top/right stem and hook.
        "M 215 95 Q 262 119 284 150 Q 296 169 285 185 "
        "Q 261 181 251 157 Q 234 116 215 95 Z",
        "M 178 212 L 218 230 L 207 254 L 209 792 "
        "Q 205 850 188 904 L 180 857 L 184 254 Z",
        "M 343 168 L 374 180 L 777 160 L 797 137 L 838 169 "
        "L 822 193 L 823 841 Q 824 886 791 919 "
        "Q 778 883 721 848 L 722 838 L 785 855 "
        "Q 798 852 797 830 L 794 184 L 372 197 Z",
        # 三: generous white space and distinct short/long horizontals.
        horizontal(331, 378, 676, 360),
        horizontal(371, 545, 639, 531),
        horizontal(305, 735, 703, 713),
    ],
    "泽": [
        # 氵: two tapered dots and a rising final stroke.
        "M 100 134 Q 148 148 177 183 Q 190 207 177 222 "
        "Q 149 212 141 189 Q 123 153 100 134 Z",
        "M 61 362 Q 111 373 143 407 Q 157 430 144 445 "
        "Q 115 433 106 412 Q 84 379 61 362 Z",
        "M 61 753 L 87 763 Q 113 760 126 732 L 219 492 "
        "L 229 497 Q 175 669 145 790 L 129 873 "
        "Q 119 879 108 862 L 90 791 Z",
        # 又: crossing sweeps, with an extended pointed right foot.
        "M 325 168 L 355 182 L 756 161 L 777 138 L 812 169 "
        "L 789 189 Q 694 363 545 449 Q 423 520 293 548 "
        "L 289 539 Q 439 487 546 414 Q 683 319 756 184 "
        "L 354 199 Z",
        "M 402 225 L 430 237 Q 506 360 652 422 "
        "Q 763 467 908 484 L 909 492 L 847 505 "
        "Q 820 516 792 505 Q 575 446 474 321 "
        "Q 430 268 402 225 Z",
        # The lower three strokes.
        horizontal(380, 624, 797, 602),
        horizontal(301, 790, 863, 763),
        "M 574 498 L 612 517 L 603 542 L 607 860 "
        "Q 603 919 591 949 L 579 912 L 581 550 Z",
    ],
    "宇": [
        # 宀: fine central dot, left hanging dot, and hooked roof.
        "M 458 68 Q 507 88 534 119 Q 549 141 535 158 "
        "Q 509 151 500 124 Q 479 87 458 68 Z",
        "M 235 187 L 247 190 Q 246 267 227 311 "
        "Q 210 340 193 330 Q 184 310 201 285 "
        "Q 226 244 235 187 Z",
        "M 245 224 L 276 233 L 804 209 L 827 184 L 874 223 "
        "Q 837 244 800 299 L 790 294 L 815 233 L 274 250 Z",
        # 于: two lifted horizontals and the central vertical hook.
        horizontal(335, 413, 725, 393),
        horizontal(180, 600, 853, 568),
        "M 523 400 L 551 400 L 549 801 Q 553 852 533 883 "
        "L 512 905 Q 496 872 439 841 L 442 831 L 508 850 "
        "Q 526 846 525 818 Z",
    ],
}


def make_font():
    builder = FontBuilder(1000, isTTF=True)
    glyph_names = {char: f"uni{ord(char):04X}" for char in OUTLINES}
    builder.setupGlyphOrder([".notdef", *glyph_names.values()])
    builder.setupCharacterMap({ord(char): name for char, name in glyph_names.items()})
    glyphs = {".notdef": TTGlyphPen(None).glyph()}
    for char, paths in OUTLINES.items():
        pen = TTGlyphPen(None)
        quadratic = Cu2QuPen(pen, max_err=0.5, reverse_direction=True)
        drawing = TransformPen(quadratic, (1, 0, 0, -1, 0, 1000))
        for path in paths:
            parse_path(path, drawing)
        glyphs[glyph_names[char]] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (1000, glyph.xMin if hasattr(glyph, "xMin") else 0)
                                    for name, glyph in glyphs.items()})
    builder.setupHorizontalHeader(ascent=1000, descent=-120)
    builder.setupNameTable({
        "familyName": "Zeyu Slender",
        "styleName": "Regular",
        "uniqueFontIdentifier": "ZeyuSlender-Regular-1.0",
        "fullName": "Zeyu Slender Regular",
        "psName": "ZeyuSlender-Regular",
        "version": "Version 1.0",
        "description": "Original slender-gold-inspired lettering for 闫泽宇; three glyphs only.",
    })
    builder.setupOS2(sTypoAscender=1000, sTypoDescender=-120, sTypoLineGap=0,
                    usWinAscent=1000, usWinDescent=120, usWeightClass=400, fsType=0)
    builder.setupPost()
    builder.font.flavor = "woff2"
    builder.save(Path(__file__).with_name("zeyu-slender.woff2"))


if __name__ == "__main__":
    make_font()
