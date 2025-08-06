const fs = require('fs');
const path = require('path');

// 解析YAML front matter的函数
function parseMarkdownFrontMatter(content, filename) {
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
    const id = filename.replace('.md', '');

    return {
        id,
        title,
        date,
        categories,
        tags,
        excerpt,
        body
    };
}

// 转换函数
function convertMdToJson(specificFiles = null) {
    const mdDir = path.join(__dirname, '..', 'posts', 'md');
    const jsonDir = path.join(__dirname, '..', 'posts', 'json');
    
    // 确保目录存在
    if (!fs.existsSync(mdDir)) {
        console.log(`Creating directory: ${mdDir}`);
        fs.mkdirSync(mdDir, { recursive: true });
    }
    
    if (!fs.existsSync(jsonDir)) {
        console.log(`Creating directory: ${jsonDir}`);
        fs.mkdirSync(jsonDir, { recursive: true });
    }
    
    // 检查md目录是否存在文件
    if (!fs.existsSync(mdDir)) {
        console.error(`MD directory does not exist: ${mdDir}`);
        return;
    }
    
    let filesToProcess = [];
    
    if (specificFiles && specificFiles.length > 0) {
        // 处理指定的文件
        specificFiles.forEach(filename => {
            const mdFile = filename.endsWith('.md') ? filename : `${filename}.md`;
            const mdPath = path.join(mdDir, mdFile);
            
            if (fs.existsSync(mdPath)) {
                filesToProcess.push(mdFile);
            } else {
                console.warn(`⚠ File not found: ${mdFile}`);
            }
        });
    } else {
        // 处理所有md文件
        const allFiles = fs.readdirSync(mdDir);
        filesToProcess = allFiles.filter(file => file.endsWith('.md'));
    }
    
    let convertedCount = 0;
    let skippedCount = 0;
    
    filesToProcess.forEach(file => {
        const mdPath = path.join(mdDir, file);
        const jsonPath = path.join(jsonDir, file.replace('.md', '.json'));
        
        // 检查是否需要转换（如果JSON文件不存在或MD文件更新了）
        let needConversion = true;
        if (fs.existsSync(jsonPath)) {
            const mdStats = fs.statSync(mdPath);
            const jsonStats = fs.statSync(jsonPath);
            
            if (mdStats.mtime <= jsonStats.mtime) {
                console.log(`⏭ Skipped ${file} (already up to date)`);
                skippedCount++;
                needConversion = false;
            }
        }
        
        if (needConversion) {
            try {
                const content = fs.readFileSync(mdPath, 'utf8');
                const postData = parseMarkdownFrontMatter(content, file);
                
                if (postData) {
                    fs.writeFileSync(jsonPath, JSON.stringify(postData, null, 2));
                    console.log(`✓ Converted ${file} to ${file.replace('.md', '.json')}`);
                    convertedCount++;
                } else {
                    console.warn(`⚠ Failed to parse ${file}`);
                }
            } catch (error) {
                console.error(`✗ Error processing ${file}:`, error.message);
            }
        }
    });
    
    console.log(`\n转换完成！`);
    console.log(`✓ 转换了 ${convertedCount} 个文件`);
    console.log(`⏭ 跳过了 ${skippedCount} 个文件（已是最新）`);
    console.log(`JSON文件保存在: ${jsonDir}`);
}

// 移动现有markdown文件到md目录的函数
function moveExistingMdFiles() {
    const postsDir = path.join(__dirname, '..', 'posts');
    const mdDir = path.join(__dirname, '..', 'posts', 'md');
    
    // 确保md目录存在
    if (!fs.existsSync(mdDir)) {
        fs.mkdirSync(mdDir, { recursive: true });
    }
    
    const files = fs.readdirSync(postsDir);
    let movedCount = 0;
    
    files.forEach(file => {
        if (file.endsWith('.md')) {
            const sourcePath = path.join(postsDir, file);
            const targetPath = path.join(mdDir, file);
            
            try {
                fs.renameSync(sourcePath, targetPath);
                console.log(`✓ Moved ${file} to posts/md/`);
                movedCount++;
            } catch (error) {
                console.error(`✗ Error moving ${file}:`, error.message);
            }
        }
    });
    
    if (movedCount > 0) {
        console.log(`\n移动完成！共移动了 ${movedCount} 个markdown文件到 posts/md/ 目录。`);
    }
}

// 主函数
function main() {
    // 获取命令行参数
    const args = process.argv.slice(2);
    let specificFiles = null;
    
    if (args.length > 0) {
        if (args[0] === '--help' || args[0] === '-h') {
            console.log(`
使用方法: node convert_md_to_json.js [文件名1] [文件名2] ...

选项:
  --help, -h    显示帮助信息
  --all, -a     转换所有markdown文件（默认行为）

示例:
  node convert_md_to_json.js                    # 转换所有文件
  node convert_md_to_json.js CNN                # 只转换CNN.md
  node convert_md_to_json.js CNN ResNet         # 转换CNN.md和ResNet.md
  node convert_md_to_json.js --all              # 转换所有文件
            `);
            return;
        } else if (args[0] === '--all' || args[0] === '-a') {
            specificFiles = null; // 转换所有文件
        } else {
            specificFiles = args; // 转换指定的文件
        }
    }
    
    console.log('开始转换markdown文件到JSON格式...\n');
    
    // 首先移动现有的markdown文件
    moveExistingMdFiles();
    
    // 然后进行转换
    convertMdToJson(specificFiles);
}

// 运行主函数
main(); 