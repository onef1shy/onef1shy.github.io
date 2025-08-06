# 博客转换脚本

这个目录包含了用于将Markdown文件转换为JSON格式的脚本，以便在GitHub Pages上正确显示博客文章。

## 文件结构

```
posts/
├── md/          # 存放原始的Markdown文件
└── json/        # 存放转换后的JSON文件
```

## 使用方法

### 1. 添加新的博客文章

1. 在 `posts/md/` 目录下创建新的 `.md` 文件
2. 确保文件包含正确的YAML front matter格式：

```yaml
---
title: 文章标题
date: 2024-01-01 10:00:00
categories:
  - 分类1
  - 分类2
tags:
  - 标签1
  - 标签2
---

文章内容...
```

### 2. 转换文件

运行转换脚本：

```bash
# 转换所有markdown文件
node scripts/convert_md_to_json.js

# 转换指定的文件
node scripts/convert_md_to_json.js CNN
node scripts/convert_md_to_json.js CNN ResNet

# 显示帮助信息
node scripts/convert_md_to_json.js --help
```

这个脚本会：
- 自动创建 `posts/md/` 和 `posts/json/` 目录（如果不存在）
- 将 `posts/` 目录下的 `.md` 文件移动到 `posts/md/` 目录
- 智能转换：只转换需要更新的文件，跳过已经是最新的文件
- 支持指定文件名进行转换，避免重复处理

### 3. 更新博客列表

转换完成后，需要更新 `js/blog.js` 文件中的文章列表，添加新文章的JSON文件路径。

## 使用示例

### 添加新文章
1. 在 `posts/md/` 目录下创建 `new-article.md`
2. 运行转换：`node scripts/convert_md_to_json.js new-article`
3. 更新 `js/blog.js` 中的文章列表

### 更新现有文章
1. 修改 `posts/md/CNN.md` 文件
2. 运行转换：`node scripts/convert_md_to_json.js CNN`
3. 脚本会自动检测文件变化并只转换更新的文件

### 批量转换
```bash
# 转换所有文件
node scripts/convert_md_to_json.js

# 转换多个指定文件
node scripts/convert_md_to_json.js CNN ResNet MNIST

# 显示帮助
node scripts/convert_md_to_json.js --help
```

## 注意事项

- 确保每个Markdown文件都有正确的YAML front matter
- 转换后的JSON文件会自动包含文章的标题、日期、分类、标签、摘要和正文内容
- 脚本会自动处理文件移动和目录创建
- 脚本会智能跳过已经是最新的文件，提高转换效率
- 支持指定文件名进行转换，避免重复处理
- 转换完成后，博客页面会自动从JSON文件加载文章内容