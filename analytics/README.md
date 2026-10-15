# 访客地图后端

接口：https://map.onef1shy.top

使用 Cloudflare Worker、D1、Durable Object 和 IPinfo Legacy。
每次打开或刷新页面记录一次浏览，重试去重；在线城市为橙色，历史访问为红色。
不保存原始 IP。

## 配置与维护

- 后端配置：`wrangler.jsonc`。
- 本地密钥：`.dev.vars`，参考 `.dev.vars.example`，已排除 Git 提交。
- 前端接口配置：`../assets/visitor-map/config.js`。

在项目根目录运行：

```sh
npm ci
npm test
npm run analytics:build
```

`analytics:build` 仅打包检查。需要更新后端时自行运行：

```sh
npx wrangler deploy --config analytics/wrangler.jsonc
```

接口：`POST /track` 记录访问，`GET /stats` 读取统计，`GET /health` 健康检查，
`/live` 通过 WebSocket 推送更新。
