// Blog JavaScript
class BlogManager {
    constructor() {
        this.posts = [];
        this.currentPost = null;
        this.cache = new Map(); // 添加缓存
        this.init();
    }

    async init() {
        await this.loadPosts();
        this.setupEventListeners();
        this.displayPosts();
    }

    async loadPosts() {
        try {
            // 文章文件列表 - 使用JSON格式，兼容 GitHub Pages
            const postFiles = [
                'posts/json/MNIST.json',
                'posts/json/CNN.json', 
                'posts/json/ResNet.json',
                'posts/json/face_recognition.json',
                'posts/json/Google_Scholar_Crawler.json',
                'posts/json/Deep_Gaussian_Process_Crop_Yield_Prediction.json',
                'posts/json/ResNet-Code.json'
            ];

            this.posts = [];
            
            for (const filename of postFiles) {
                try {
                    console.log(`Attempting to load: ${filename}`); // 调试信息
                    const response = await fetch(filename);
                    if (!response.ok) {
                        console.warn(`Failed to load ${filename}: ${response.status} ${response.statusText}`);
                        continue;
                    }
                    
                    const postData = await response.json();
                    const post = {
                        id: postData.id,
                        title: postData.title,
                        date: postData.date,
                        categories: postData.categories || [],
                        tags: postData.tags || [],
                        excerpt: postData.excerpt,
                        filename: filename,
                        body: postData.body
                    };
                    
                    this.posts.push(post);
                    console.log(`Successfully loaded: ${filename}`); // 调试信息
                } catch (error) {
                    console.warn(`Error loading ${filename}:`, error);
                }
            }

            // 按日期排序（最新的在前）
            this.posts.sort((a, b) => new Date(b.date) - new Date(a.date));
            
        } catch (error) {
            console.error('Failed to load posts:', error);
        }
    }

    parseMarkdownFrontMatter(content, filename) {
        // 解析YAML front matter
        const frontMatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
        if (!frontMatterMatch) {
            console.warn(`No front matter found in ${filename}`);
            return null;
        }

        const frontMatter = frontMatterMatch[1];
        const body = content.replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '');

        // 解析YAML
        const titleMatch = frontMatter.match(/title:\s*(.+)/);
        const dateMatch = frontMatter.match(/date:\s*(.+)/);
        const categoriesMatch = frontMatter.match(/categories:\s*\n((?:\s*-\s*.+\n?)*)/);
        const tagsMatch = frontMatter.match(/tags:\s*\n((?:\s*-\s*.+\n?)*)/);

        if (!titleMatch) {
            console.warn(`No title found in ${filename}`);
            return null;
        }

        const title = titleMatch[1].trim();
        const date = dateMatch ? dateMatch[1].trim() : 'Unknown';
        
        // 解析分类
        let categories = [];
        if (categoriesMatch) {
            const categoriesText = categoriesMatch[1];
            categories = categoriesText.match(/-\s*(.+)/g)?.map(cat => cat.replace('-', '').trim()) || [];
        }

        // 解析标签
        let tags = [];
        if (tagsMatch) {
            const tagsText = tagsMatch[1];
            tags = tagsText.match(/-\s*(.+)/g)?.map(tag => tag.replace('-', '').trim()) || [];
        }

        // 生成摘要（取前200个字符）
        const excerpt = body.replace(/[#*`]/g, '').substring(0, 200).trim() + '...';

        // 生成ID（从文件名）
        const id = filename.replace('posts/', '').replace('.md', '');

        return {
            id,
            title,
            date,
            categories,
            tags,
            excerpt,
            filename,
            body
        };
    }

    setupEventListeners() {
        // 返回列表按钮
        document.getElementById('back-to-list').addEventListener('click', () => {
            this.showPostList();
        });

        // 导航菜单
        document.querySelectorAll('.page-link').forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.getAttribute('href') === '#blog-list') {
                    e.preventDefault();
                    this.showPostList();
                }
            });
        });
    }

    displayPosts() {
        const container = document.getElementById('posts-container');
        container.innerHTML = '';

        this.posts.forEach(post => {
            const postCard = this.createPostCard(post);
            container.appendChild(postCard);
        });
    }

    createPostCard(post) {
        const card = document.createElement('div');
        card.className = 'post-card';
        card.dataset.postId = post.id;

        const categoriesHtml = post.categories.map(cat => 
            `<a href="#" class="category-link">${cat}</a>`
        ).join('');

        const tagsHtml = post.tags.map(tag => 
            `<a href="#" class="tag-link">${tag}</a>`
        ).join('');

        card.innerHTML = `
            <h3>${post.title}</h3>
            <div class="post-excerpt">${post.excerpt}</div>
            <div class="post-meta">
                <span class="post-date">
                    <i class="fa fa-calendar"></i>
                    ${post.date}
                </span>
                <span class="post-categories">
                    <i class="fa fa-folder"></i>
                    ${categoriesHtml}
                </span>
                <span class="post-tags">
                    <i class="fa fa-tags"></i>
                    ${tagsHtml}
                </span>
            </div>
        `;

        card.addEventListener('click', () => {
            this.showLoading();
            this.loadPost(post);
        });

        return card;
    }

    showLoading() {
        document.getElementById('post-content').style.display = 'block';
        document.getElementById('blog-list').style.display = 'none';
        document.getElementById('post-title').textContent = 'Loading...';
        document.getElementById('post-body').innerHTML = '<div class="loading">Loading article...</div>';
    }

    async loadPost(post) {
        try {
            // 检查缓存
            if (this.cache.has(post.id)) {
                const cachedHtml = this.cache.get(post.id);
                this.displayPost(post, cachedHtml);
                return;
            }

            // 如果post对象已经有body内容，直接使用
            if (post.body) {
                const html = this.convertMarkdownToHtml(post.body);
                this.cache.set(post.id, html); // 缓存结果
                this.displayPost(post, html);
                return;
            }

            // 否则重新加载文件 - 使用相对路径
            const filename = post.filename.startsWith('/') ? post.filename.substring(1) : post.filename;
            console.log(`Loading post file: ${filename}`); // 调试信息
            const response = await fetch(filename);
            console.log(`Response status: ${response.status} ${response.statusText}`); // 调试信息
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const markdown = await response.text();
            const html = this.convertMarkdownToHtml(markdown);
            
            this.cache.set(post.id, html); // 缓存结果
            this.displayPost(post, html);
        } catch (error) {
            console.error('Failed to load post:', error);
            this.displayPost(post, `<p>文章加载失败: ${error.message}</p>`);
        }
    }

    convertMarkdownToHtml(markdown) {
        let html = markdown;
        console.log('Original markdown:', markdown.substring(0, 500)); // 调试信息

        // 处理数学公式（行内和块级）
        html = html.replace(/\$\$([\s\S]*?)\$\$/g, '\\[$1\\]');
        html = html.replace(/\$([^\$\n]+?)\$/g, '\\($1\\)');

        // 处理代码块
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // 处理标题 - 使用更精确的正则表达式，跳过代码块内的内容
        const lines = html.split('\n');
        let inCodeBlock = false;
        let processedLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // 检查是否进入或离开代码块
            if (line.includes('<pre><code')) {
                inCodeBlock = true;
            } else if (line.includes('</code></pre>')) {
                inCodeBlock = false;
            }
            
            // 如果在代码块内，直接保留原行
            if (inCodeBlock) {
                processedLines.push(line);
                continue;
            }
            
            // 处理标题，但只在非代码块内
            const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
            if (headingMatch) {
                const hashes = headingMatch[1];
                const content = headingMatch[2];
                const level = hashes.length;
                // 清理标题内容，移除可能的编号
                const cleanContent = content.replace(/^\d+\.\s*/, '').replace(/^\d+\.\d+\s*/, '');
                console.log(`Processing heading: ${hashes} ${cleanContent}`); // 调试信息
                processedLines.push(`<h${level}>${cleanContent}</h${level}>`);
            } else {
                processedLines.push(line);
            }
        }

        html = processedLines.join('\n');
        console.log('After heading processing:', html.substring(0, 500)); // 调试信息

        // 处理粗体和斜体
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // 处理链接
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // 处理图片 - 修复路径问题，兼容 GitHub Pages
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
            console.log('Processing image:', src); // 调试信息
            
            // 如果路径以 /images 开头，转换为相对路径
            if (src.startsWith('/images')) {
                const newSrc = src.substring(1); // 移除开头的 /
                return `<img src="${newSrc}" alt="${alt}">`;
            }
            // 如果路径以 ../images 开头，保持原样
            if (src.startsWith('../images')) {
                return `<img src="${src}" alt="${alt}">`;
            }
            // 如果路径以 images 开头，保持原样
            if (src.startsWith('images')) {
                return `<img src="${src}" alt="${alt}">`;
            }
            // 如果是完整的URL，保持原样
            if (src.startsWith('http')) {
                return `<img src="${src}" alt="${alt}">`;
            }
            // 其他情况，尝试添加 images 前缀（相对路径）
            return `<img src="images/${src}" alt="${alt}">`;
        });

        // 处理表格
        html = this.processTables(html);

        // 处理引用块
        html = html.replace(/^>\s+(.+)$/gm, '<blockquote>$1</blockquote>');

        // 处理列表 - 改进版本，避免重复
        const listLines = html.split('\n');
        let inList = false;
        let listProcessedLines = [];

        for (let i = 0; i < listLines.length; i++) {
            const line = listLines[i];
            // 跳过已经处理过的列表项，但保留标题
            if (line.includes('<li>') || line.includes('</ul>')) {
                listProcessedLines.push(line);
                continue;
            }
            
            // 如果是标题，直接保留
            if (line.match(/^<h[1-6]>/) || line.match(/^<\/h[1-6]>/)) {
                if (inList) {
                    listProcessedLines.push('</ul>');
                    inList = false;
                }
                listProcessedLines.push(line);
                continue;
            }
            
            if (line.match(/^[\*\-]\s+/)) {
                if (!inList) {
                    listProcessedLines.push('<ul>');
                    inList = true;
                }
                listProcessedLines.push('<li>' + line.replace(/^[\*\-]\s+/, '') + '</li>');
            } else {
                if (inList) {
                    listProcessedLines.push('</ul>');
                    inList = false;
                }
                listProcessedLines.push(line);
            }
        }
        if (inList) {
            listProcessedLines.push('</ul>');
        }

        html = listProcessedLines.join('\n');

        // 处理段落 - 避免将标题包装在p标签中
        const paragraphLines = html.split('\n');
        let paragraphProcessedLines = [];
        let inParagraph = false;
        
        for (let i = 0; i < paragraphLines.length; i++) {
            const line = paragraphLines[i];
            
            // 如果是标题、列表、代码块等，结束当前段落
            if (line.match(/^<h[1-6]>/) || line.match(/^<ul>/) || line.match(/^<pre>/) || 
                line.match(/^<blockquote>/) || line.match(/^<\/ul>/) || line.match(/^<\/pre>/) || 
                line.match(/^<\/blockquote>/) || line.trim() === '') {
                if (inParagraph) {
                    paragraphProcessedLines.push('</p>');
                    inParagraph = false;
                }
                paragraphProcessedLines.push(line);
            } else {
                // 普通文本行
                if (!inParagraph) {
                    paragraphProcessedLines.push('<p>');
                    inParagraph = true;
                }
                paragraphProcessedLines.push(line);
            }
        }
        
        if (inParagraph) {
            paragraphProcessedLines.push('</p>');
        }
        
        html = paragraphProcessedLines.join('\n');

        // 清理多余的标签
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p><h([1-6])>/g, '<h$1>');
        html = html.replace(/<\/h([1-6])><\/p>/g, '</h$1>');
        html = html.replace(/<p><ul>/g, '<ul>');
        html = html.replace(/<\/ul><\/p>/g, '</ul>');
        html = html.replace(/<p><pre>/g, '<pre>');
        html = html.replace(/<\/pre><\/p>/g, '</pre>');
        html = html.replace(/<p><blockquote>/g, '<blockquote>');
        html = html.replace(/<\/blockquote><\/p>/g, '</blockquote>');

        console.log('Final HTML result:', html.substring(0, 500)); // 调试信息
        return html;
    }

    processTables(markdown) {
        // 匹配表格的正则表达式
        const tableRegex = /^\|(.+)\|$/gm;
        const lines = markdown.split('\n');
        let inTable = false;
        let tableLines = [];
        let processedLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            if (line.match(tableRegex)) {
                if (!inTable) {
                    inTable = true;
                    tableLines = [];
                }
                tableLines.push(line);
            } else {
                if (inTable) {
                    // 处理表格
                    processedLines.push(this.convertTableToHtml(tableLines));
                    inTable = false;
                }
                processedLines.push(line);
            }
        }

        if (inTable) {
            processedLines.push(this.convertTableToHtml(tableLines));
        }

        return processedLines.join('\n');
    }

    convertTableToHtml(tableLines) {
        if (tableLines.length < 2) return tableLines.join('\n');

        let html = '<table>\n<thead>\n<tr>\n';
        
        // 处理表头
        const headerLine = tableLines[0];
        const headers = headerLine.split('|').slice(1, -1).map(h => h.trim());
        
        headers.forEach(header => {
            html += `<th>${header}</th>\n`;
        });
        
        html += '</tr>\n</thead>\n<tbody>\n';

        // 跳过表头和分隔行（第二行通常是分隔符）
        for (let i = 2; i < tableLines.length; i++) {
            const rowLine = tableLines[i];
            // 跳过分隔行（只包含 | 和 - 的行）
            if (rowLine.match(/^\|[\s\-:|]+\|$/)) {
                continue;
            }
            
            const cells = rowLine.split('|').slice(1, -1).map(c => c.trim());
            
            html += '<tr>\n';
            cells.forEach(cell => {
                html += `<td>${cell}</td>\n`;
            });
            html += '</tr>\n';
        }

        html += '</tbody>\n</table>';
        return html;
    }

    displayPost(post, html) {
        document.getElementById('blog-list').style.display = 'none';
        document.getElementById('post-content').style.display = 'block';

        document.getElementById('post-title').textContent = post.title;
        
        const meta = document.querySelector('.post-meta');
        meta.innerHTML = `
            <span class="post-date">
                <i class="fa fa-calendar"></i>
                ${post.date}
            </span>
            <span class="post-categories">
                <i class="fa fa-folder"></i>
                ${post.categories.map(cat => `<a href="#">${cat}</a>`).join('')}
            </span>
            <span class="post-tags">
                <i class="fa fa-tags"></i>
                ${post.tags.map(tag => `<a href="#">${tag}</a>`).join('')}
            </span>
        `;

        document.getElementById('post-body').innerHTML = html;

        // 延迟渲染MathJax，避免重复渲染
        setTimeout(() => {
            if (window.MathJax) {
                MathJax.typesetPromise().then(() => {
                    console.log('MathJax rendering completed');
                }).catch((err) => {
                    console.log('MathJax error:', err);
                });
            }
        }, 100);

        // 滚动到顶部
        window.scrollTo(0, 0);
    }



    showPostList() {
        document.getElementById('post-content').style.display = 'none';
        document.getElementById('blog-list').style.display = 'block';
    }
}

// 初始化博客管理器
document.addEventListener('DOMContentLoaded', () => {
    new BlogManager();
}); 