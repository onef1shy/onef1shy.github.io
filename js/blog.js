// Blog JavaScript
class BlogManager {
    constructor() {
        this.posts = [];
        this.currentPost = null;
        this.cache = new Map(); // 添加缓存
        this.init();
    }

    async init() {
        this.showLoading();
        await this.loadPosts();
        this.setupEventListeners();
        this.displayPosts();
        this.hideLoading();
    }

    showLoading() {
        const loadingContainer = document.getElementById('loading-container');
        const postsContainer = document.getElementById('posts-container');
        
        if (loadingContainer) {
            loadingContainer.style.display = 'flex';
            loadingContainer.classList.remove('fade-out');
        }
        
        if (postsContainer) {
            postsContainer.classList.remove('show');
        }
    }

    hideLoading() {
        const loadingContainer = document.getElementById('loading-container');
        const postsContainer = document.getElementById('posts-container');
        
        // 同时开始两个动画，让它们重叠
        if (loadingContainer) {
            loadingContainer.classList.add('fade-out');
            // 延长时间，确保所有CSS动画（包括高度变化）完成后再移除
            setTimeout(() => {
                loadingContainer.style.display = 'none';
            }, 700);
        }
        
        if (postsContainer) {
            // 在loading开始收缩后立即开始显示内容
            setTimeout(() => {
                postsContainer.classList.add('show');
            }, 50);
        }
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

        // 监听浏览器返回/前进按钮
        window.addEventListener('popstate', (event) => {
            this.handlePopState(event);
        });

        // 检查URL中是否有博客文章ID
        this.checkInitialURL();
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
            `<span class="category-link">${cat}</span>`
        ).join('');

        const tagsHtml = post.tags.map(tag => 
            `<span class="tag-link">${tag}</span>`
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
            this.loadPost(post);
        });

        return card;
    }

    async loadPost(post) {
        try {
            // 更新URL以包含文章ID
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('post', post.id);
            window.history.pushState({ type: 'post', postId: post.id }, '', newUrl);

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
        let headings = []; // 收集标题信息用于生成目录

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
                // 保留原始内容用于显示，但生成ID时包含编号以确保唯一性
                const cleanContent = content.replace(/^\d+\.\s*/, '').replace(/^\d+\.\d+\s*/, '');
                console.log(`Processing heading: ${hashes} ${cleanContent}`); // 调试信息
                
                // 生成唯一的ID，使用原始内容以确保唯一性
                const headingId = this.generateHeadingId(content);
                processedLines.push(`<h${level} id="${headingId}">${cleanContent}</h${level}>`);
                
                // 调试信息
                console.log(`Heading ID generated: "${headingId}" for "${content}" -> "${cleanContent}"`);
                
                // 收集标题信息
                headings.push({
                    level: level,
                    text: cleanContent, // 显示清理后的内容
                    id: headingId,
                    originalText: content // 保存原始内容用于调试
                });
            } else {
                processedLines.push(line);
            }
        }

        html = processedLines.join('\n');
        console.log('After heading processing:', html.substring(0, 500)); // 调试信息
        
        // 生成目录
        this.generateTableOfContents(headings);

        // 处理图片 - 必须在链接处理之前，避免被链接正则表达式误匹配
        html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
            console.log('Processing image:', src); // 调试信息
            
            let imgSrc = src;
            // 如果路径以 /images 开头，转换为相对路径
            if (src.startsWith('/images')) {
                imgSrc = src.substring(1); // 移除开头的 /
            }
            // 如果路径以 ../images 开头，保持原样
            else if (src.startsWith('../images')) {
                imgSrc = src;
            }
            // 如果路径以 images 开头，保持原样
            else if (src.startsWith('images')) {
                imgSrc = src;
            }
            // 如果是完整的URL，保持原样
            else if (src.startsWith('http')) {
                imgSrc = src;
            }
            // 其他情况，尝试添加 images 前缀（相对路径）
            else {
                imgSrc = `images/${src}`;
            }
            
            // 如果有标题，创建带标题的图片容器
            if (alt && alt.trim()) {
                return `<figure class="blog-image-container">
                    <img src="${imgSrc}" alt="${alt}" class="blog-image">
                    <figcaption class="blog-image-caption">${alt}</figcaption>
                </figure>`;
            } else {
                return `<img src="${imgSrc}" alt="${alt}" class="blog-image">`;
            }
        });

        // 处理粗体和斜体
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // 处理链接 - 在图片处理之后，避免误匹配图片语法
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

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

    generateHeadingId(text) {
        // 生成唯一的标题ID，改进版本
        let id = text
            .toLowerCase()
            .replace(/[^\w\s-]/g, '') // 移除特殊字符
            .replace(/\s+/g, '-') // 空格替换为连字符
            .replace(/-+/g, '-') // 多个连字符替换为单个
            .trim();
        
        // 确保ID不为空
        if (!id) {
            id = 'heading-' + Math.random().toString(36).substr(2, 9);
        }
        
        // 确保ID唯一性
        let counter = 1;
        let originalId = id;
        while (document.getElementById(id)) {
            id = originalId + '-' + counter;
            counter++;
        }
        
        return id;
    }

    generateTableOfContents(headings) {
        const tocNav = document.getElementById('toc-nav');
        if (!tocNav) {
            return;
        }

        if (headings.length === 0) {
            tocNav.innerHTML = '<p style="text-align: center; color: #586069; font-style: italic; padding: 1rem;">暂无目录</p>';
            return;
        }

        let tocHtml = '<ul>';
        
        headings.forEach(heading => {
            const indentClass = `toc-h${heading.level}`;
            tocHtml += `
                <li>
                    <a href="#${heading.id}" class="${indentClass}" data-level="${heading.level}">
                        ${heading.text}
                    </a>
                </li>
            `;
        });
        
        tocHtml += '</ul>';
        tocNav.innerHTML = tocHtml;

        // 添加点击事件
        tocNav.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    // 计算偏移量，考虑固定头部
                    const headerHeight = document.querySelector('.site-header')?.offsetHeight || 0;
                    const offset = targetElement.offsetTop - headerHeight - 20;
                    
                    // 平滑滚动到目标位置
                    window.scrollTo({
                        top: offset,
                        behavior: 'smooth'
                    });
                    
                    // 更新活动状态
                    this.updateActiveTocItem(targetId);
                    
                    // 更新URL，但不触发页面跳转
                    const currentUrl = new URL(window.location);
                    currentUrl.hash = targetId;
                    window.history.replaceState({}, '', currentUrl);
                    
                    // 调试信息
                    console.log(`Successfully scrolled to: ${targetId}`);
                } else {
                    console.warn('Target element not found:', targetId);
                    console.log('Available headings:', Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({id: h.id, text: h.textContent})));
                }
            });
        });

        // 设置滚动监听
        this.setupScrollSpy();
    }

    setupScrollSpy() {
        // 获取所有标题元素
        const headings = document.querySelectorAll('.post-body h1, .post-body h2, .post-body h3, .post-body h4, .post-body h5, .post-body h6');
        
        if (headings.length === 0) return;

        // 创建Intersection Observer来监听标题的可见性
        const observer = new IntersectionObserver((entries) => {
            let activeHeading = null;
            let maxRatio = 0;
            
            entries.forEach(entry => {
                if (entry.isIntersecting && entry.intersectionRatio > maxRatio) {
                    maxRatio = entry.intersectionRatio;
                    activeHeading = entry.target;
                }
            });
            
            if (activeHeading) {
                this.updateActiveTocItem(activeHeading.id);
            }
        }, {
            rootMargin: '-20% 0px -60% 0px', // 调整触发区域，让标题在视口中更早被激活
            threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] // 更多阈值，提高精度
        });

        // 观察所有标题
        headings.forEach(heading => {
            observer.observe(heading);
        });
        
        // 添加滚动事件监听作为备用，提高响应性
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                this.updateActiveTocItemOnScroll(headings);
            }, 50); // 减少延迟，提高响应速度
        });
    }

    updateActiveTocItem(activeId) {
        // 移除所有活动状态
        document.querySelectorAll('#toc-nav a').forEach(link => {
            link.classList.remove('active');
        });

        // 添加活动状态到当前项
        const activeLink = document.querySelector(`#toc-nav a[href="#${activeId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
            
            // 确保活动项在视图中可见
            const tocNav = document.getElementById('toc-nav');
            if (tocNav) {
                const linkRect = activeLink.getBoundingClientRect();
                const navRect = tocNav.getBoundingClientRect();
                
                if (linkRect.top < navRect.top || linkRect.bottom > navRect.bottom) {
                    activeLink.scrollIntoView({
                        behavior: 'smooth',
                        block: 'nearest'
                    });
                }
            }
        }
    }

    updateActiveTocItemOnScroll(headings) {
        const scrollTop = window.pageYOffset;
        const headerHeight = document.querySelector('.site-header')?.offsetHeight || 0;
        const offset = headerHeight + 150; // 增加偏移量，让标题更早被激活
        
        let activeHeading = null;
        let minDistance = Infinity;
        
        // 找到当前最接近视口顶部的标题
        headings.forEach(heading => {
            const headingTop = heading.offsetTop - offset;
            const distance = Math.abs(scrollTop - headingTop);
            
            // 如果标题在视口上方或接近视口顶部，选择距离最小的
            if (scrollTop >= headingTop - 50 && distance < minDistance) {
                minDistance = distance;
                activeHeading = heading;
            }
        });
        
        // 如果没有找到合适的标题，选择最后一个在视口上方的标题
        if (!activeHeading) {
            for (let i = headings.length - 1; i >= 0; i--) {
                const heading = headings[i];
                const headingTop = heading.offsetTop - offset;
                if (scrollTop >= headingTop) {
                    activeHeading = heading;
                    break;
                }
            }
        }
        
        if (activeHeading) {
            this.updateActiveTocItem(activeHeading.id);
        }
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
                ${post.categories.map(cat => `<span>${cat}</span>`).join('')}
            </span>
            <span class="post-tags">
                <i class="fa fa-tags"></i>
                ${post.tags.map(tag => `<span>${tag}</span>`).join('')}
            </span>
        `;

        document.getElementById('post-body').innerHTML = html;

        // 延迟渲染MathJax，避免重复渲染
        setTimeout(() => {
            if (window.MathJax) {
                MathJax.typesetPromise().then(() => {
                    console.log('MathJax rendering completed');
                    
                    // MathJax渲染完成后，处理锚点链接
                    this.handleAnchorLink();
                }).catch((err) => {
                    console.log('MathJax error:', err);
                    this.handleAnchorLink();
                });
            } else {
                this.handleAnchorLink();
            }
        }, 100);

        // 滚动到顶部（如果没有锚点）
        if (!window.location.hash) {
            window.scrollTo(0, 0);
        }

        // 更新页面标题
        document.title = `${post.title} - Blog - Zeyu Yan`;
    }

    handleAnchorLink() {
        // 检查URL中是否有锚点
        if (window.location.hash) {
            const targetId = window.location.hash.substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                // 延迟滚动，确保内容已完全加载
                setTimeout(() => {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // 更新导引栏活动状态
                    this.updateActiveTocItem(targetId);
                }, 500);
            }
        }
    }



    showPostList() {
        document.getElementById('post-content').style.display = 'none';
        document.getElementById('blog-list').style.display = 'block';
        
        // 如果文章列表为空，显示loading并重新加载
        if (this.posts.length === 0) {
            this.showLoading();
            this.loadPosts().then(() => {
                this.displayPosts();
                this.hideLoading();
            });
        }
        
        // 更新URL到博客列表页面
        const newUrl = new URL(window.location);
        newUrl.pathname = '/blog.html';
        newUrl.search = '';
        newUrl.hash = '';
        window.history.pushState({ type: 'blog-list' }, '', newUrl);
        
        // 更新页面标题
        document.title = 'Blog - Zeyu Yan';
        
        // 滚动到顶部
        window.scrollTo(0, 0);
    }

    handlePopState(event) {
        // 处理浏览器返回/前进按钮
        const url = new URL(window.location);
        const postId = url.searchParams.get('post');
        
        // 检查当前页面是否在博客页面
        const isOnBlogPage = window.location.pathname.includes('blog.html') || 
                            window.location.pathname.endsWith('/') && window.location.search.includes('post');
        
        // 如果当前在博客页面
        if (isOnBlogPage) {
            if (postId) {
                // 如果有文章ID，显示对应的文章
                const post = this.posts.find(p => p.id === postId);
                if (post) {
                    this.loadPost(post);
                } else {
                    // 如果找不到文章，显示博客列表
                    this.showPostList();
                }
            } else {
                // 如果没有文章ID，显示博客列表
                this.showPostList();
            }
        } else {
            // 如果不在博客页面，不进行任何操作（让浏览器正常导航）
            return;
        }
    }

    checkInitialURL() {
        // 检查初始URL是否包含博客文章ID
        const url = new URL(window.location);
        const postId = url.searchParams.get('post');
        
        if (postId) {
            // 如果有文章ID，延迟加载文章（等待文章列表加载完成）
            setTimeout(() => {
                const post = this.posts.find(p => p.id === postId);
                if (post) {
                    this.loadPost(post);
                }
            }, 100);
        }
    }
}

// 初始化博客管理器
document.addEventListener('DOMContentLoaded', () => {
    new BlogManager();
}); 