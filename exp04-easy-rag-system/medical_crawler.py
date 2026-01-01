import os
import time
import random
import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class MedicalSpider:
    def __init__(self, base_url, output_dir="./data", delay=2):
        """
        初始化爬虫
        :param base_url: 起始网址（例如：'https://www.baikemy.com/' 或医学百科站）
        :param output_dir: 保存HTML的目录
        :param delay: 请求间隔（秒），避免被封
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': UserAgent().random})
        self.visited_urls = set() # 记录已访问URL，避免重复
        os.makedirs(output_dir, exist_ok=True)

    def fetch_page(self, url):
        """获取页面内容，包含错误处理和随机延迟"""
        try:
            time.sleep(self.delay + random.uniform(0, 1)) # 随机延迟更安全
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status() # 检查HTTP错误
            # 注意编码：很多中文网站是gbk或gb2312
            resp.encoding = resp.apparent_encoding or 'utf-8'
            return resp.text
        except Exception as e:
            logging.error(f"抓取 {url} 失败: {e}")
            return None

    def extract_links(self, html, css_selector='a'):
        """从页面中提取链接（可根据目标网站调整选择器）"""
        soup = BeautifulSoup(html, 'lxml')
        links = set()
        for a in soup.select(css_selector):
            href = a.get('href', '')
            full_url = urljoin(self.base_url, href)
            # 可根据需要添加过滤条件，例如只保留包含“disease”的链接
            if full_url.startswith(self.base_url) and full_url not in self.visited_urls:
                links.add(full_url)
        return list(links)

    def save_html(self, html, filename):
        """将HTML内容保存到文件（Windows路径处理）"""
        # 确保文件名在Windows下合法
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        path = os.path.join(self.output_dir, f"{safe_filename}.html")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            logging.info(f"已保存: {path}")
            return True
        except Exception as e:
            logging.error(f"保存文件失败 {filename}: {e}")
            return False

    def crawl(self, start_path='/', max_pages=20, link_selector='a'):
        """
        主爬取流程
        :param start_path: 起始路径
        :param max_pages: 最大抓取页面数
        :param link_selector: 链接CSS选择器
        """
        start_url = urljoin(self.base_url, start_path)
        to_visit = [start_url]

        while to_visit and len(self.visited_urls) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in self.visited_urls:
                continue

            logging.info(f"正在抓取 ({len(self.visited_urls)}/{max_pages}): {current_url}")
            html = self.fetch_page(current_url)
            if not html:
                continue

            # 提取标题作为文件名
            soup = BeautifulSoup(html, 'lxml')
            title = soup.title.string.strip() if soup.title else f"page_{len(self.visited_urls)}"
            self.save_html(html, title)

            self.visited_urls.add(current_url)

            # 提取新链接（控制深度，避免抓取过多）
            if len(self.visited_urls) < max_pages:
                new_links = self.extract_links(html, link_selector)
                # 可以在这里添加逻辑：只添加看起来是疾病详情页的链接
                to_visit.extend([link for link in new_links if link not in self.visited_urls])

        logging.info(f"爬取结束，共抓取 {len(self.visited_urls)} 个页面。")

# 使用示例
if __name__ == "__main__":
    print("=== 爬虫脚本开始执行 (测试模式) ===")

    # 1. 初始化爬虫，目标是医学百科
    spider = MedicalSpider(
        base_url='https://www.yixue.com',
        output_dir='./data',
        delay=5  # 延迟设置长一些，表示友好
    )

    # 2. 手动指定一个你想要抓取的、具体的疾病页面URL
    # 你需要先去医学百科网站，搜索一个疾病（比如“感冒”），把它的完整URL复制到这里
    test_disease_url = "https://www.yixue.com/感冒"  # ！！！请替换成真实有效的URL！！！
    
    print(f"尝试抓取单页: {test_disease_url}")

    # 3. 仅抓取这一个页面
    html_content = spider.fetch_page(test_disease_url)
    
    if html_content:
        # 尝试从页面中提取标题作为文件名
        soup = BeautifulSoup(html_content, 'lxml')
        title_tag = soup.find('title')
        filename = title_tag.text.strip() if title_tag else "test_page"
        print(f"页面抓取成功，标题为: {filename}")
        
        # 保存文件
        spider.save_html(html_content, filename)
        print("单页抓取测试完成，请检查 ./data 文件夹。")
    else:
        print("页面抓取失败，请检查：")
        print("  1. 上述URL是否能在浏览器中正常打开？")
        print("  2. 网站是否有反爬机制（如需要登录）？")
        print("  3. 网络连接是否正常？")