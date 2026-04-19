
# onef1shy.github.io

我的个人学术主页源码仓库：<https://onef1shy.github.io/>

## 主要文件

- `_pages/about.md`：主页正文内容
- `_config.yml`：站点配置、Google Scholar 数据源配置
- `_includes/fetch_google_scholar_stats.html`：论文引用数渲染逻辑
- `.github/workflows/google_scholar_crawler.yaml`：定时抓取 Google Scholar 数据
- `google_scholar_crawler/`：爬虫脚本，结果会推送到 `google-scholar-stats` 分支

## Google Scholar 引用显示

页面通过下面这类标签渲染单篇论文引用数：

```html
<span class='show_paper_citations' data='SCHOLAR_PAPER_ID'></span>
```

`data` 的值需要和 `gs_data.json` 里的论文键一致，例如：

```text
fjLiFtgAAAAJ:d1gkVwhDpl0C
```

获取方式：

1. 打开自己的 Google Scholar 主页
2. 点击具体论文
3. 在地址栏中找到 `citation_for_view=...`
4. 使用后面的完整值作为 `data`

注意：Google Scholar 可能在条目合并、重建索引或正式出版后更换 paper ID。

## Google Scholar 数据源

当前站点建议直接读取 GitHub Raw：

```text
https://raw.githubusercontent.com/onef1shy/onef1shy.github.io/google-scholar-stats/gs_data.json
```

对应配置在 `_config.yml`：

```yml
google_scholar_stats_use_cdn : false
```

这样更新更及时，也更容易排查问题。

## jsDelivr CDN 刷新

如果以后重新启用 CDN：

```yml
google_scholar_stats_use_cdn : true
```

当 `gs_data.json` 更新后，可在 jsDelivr 官方工具里手动清缓存：

1. 打开 <https://www.jsdelivr.com/tools/purge>
2. 输入下面这个 URL 并 purge：

```text
https://cdn.jsdelivr.net/gh/onef1shy/onef1shy.github.io@google-scholar-stats/gs_data.json
```

## 本地调试

1. 安装 Jekyll 依赖环境：<https://jekyllrb.com/docs/installation/>
2. 运行：

```bash
bash run_server.sh
```

3. 打开 <http://127.0.0.1:4000>
4. 如果修改了 `_config.yml`，需要重启本地服务
