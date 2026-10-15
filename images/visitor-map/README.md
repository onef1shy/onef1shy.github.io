# 访客地图底图

`world.svg` 基于 Natural Earth 公有领域数据，来源为
[world-atlas@2.0.2](https://github.com/topojson/world-atlas)，授权见 `LICENSE-world-atlas.txt`。

源数据保存在 `scripts/data/countries-110m.json`，运行 `npm run map:build` 重新生成。
底图和圆点使用相同的 Miller 投影，不含南极洲；国家边界为浅蓝细线。

海洋 `#2d78ad`，陆地白色，历史访问红色 `#ff1717`，在线访客橙色 `#f79a00`。
