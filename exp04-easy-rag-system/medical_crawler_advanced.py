import os
import time
import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class MedicalSpider:
    def __init__(self, base_url, output_dir="./data", delay=5):
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        os.makedirs(output_dir, exist_ok=True)

    def fetch_page(self, url):
        try:
            time.sleep(self.delay)
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            # 医学百科多为utf-8编码，但用apparent_encoding更安全
            resp.encoding = resp.apparent_encoding or 'utf-8'
            return resp.text
        except Exception as e:
            logging.error(f"抓取失败 {url}: {e}")
            return None

    def save_html(self, html, filename):
        # 清理文件名，去除Windows不合法的字符，并截取前50个字符防止过长
        safe_filename = re.sub(r'[<>:"/\\|?*]', '', filename)[:50]
        path = os.path.join(self.output_dir, f"{safe_filename}.html")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            logging.info(f"已保存: {path}")
            return True
        except Exception as e:
            logging.error(f"保存失败 {filename}: {e}")
            return False

def mode_direct_crawl(spider):
    """
    模式1：直接爬取指定的疾病页面列表（最稳定、推荐）
    """
    # ===== 请在这里编辑你想要爬取的疾病列表 =====
    disease_list = [
        "感冒",
        "高血压",
        "糖尿病",
        "冠心病",
        "胃炎"
    ]
    # ===========================================

    logging.info(f"开始【直接爬取】模式，计划抓取 {len(disease_list)} 个页面。")

    for disease_name in disease_list:
        # 构建URL，例如：https://www.yixue.com/感冒
        url = urljoin(spider.base_url, disease_name)
        logging.info(f"正在抓取: {url}")

        html = spider.fetch_page(url)
        if html:
            # 尝试从页面<title>标签提取更精确的标题，若失败则用疾病名
            soup = BeautifulSoup(html, 'lxml')
            page_title = soup.title.string.strip() if soup.title else disease_name
            spider.save_html(html, page_title)
        else:
            logging.warning(f"跳过: {disease_name}")

def mode_auto_discover(spider, start_path):
    """
    模式2：从起始页开始，自动发现并爬取疾病链接（需要适配网站结构）
    """
    logging.info(f"开始【自动发现】模式，起始页: {start_path}")

    start_url = urljoin(spider.base_url, start_path)
    first_page_html = spider.fetch_page(start_url)

    if not first_page_html:
        logging.error("起始页抓取失败，无法继续。")
        return

    # 1. 先保存起始页本身
    soup = BeautifulSoup(first_page_html, 'lxml')
    start_page_title = soup.title.string.strip() if soup.title else "index"
    spider.save_html(first_page_html, start_page_title)

    # 2. 尝试发现疾病链接 (这是关键，需要根据实际页面调整)
    # 假设疾病链接都是 <a href="/疾病名"> 的形式
    disease_links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        # 过滤逻辑示例：寻找直接以疾病名（中文字符）构成的简单路径
        # 你可以根据实际看到的HTML结构修改这里
        if (href.startswith('/') and 
            not href.startswith(('http', '#', '/wiki', '/w/')) and 
            len(href) > 1 and 
            '/' not in href[1:]): # 确保是单级路径，如“/感冒”
            full_url = urljoin(spider.base_url, href)
            disease_links.append(full_url)

    # 去重并限制数量
    unique_links = list(set(disease_links))[:5] # 只取前5个作为测试
    logging.info(f"从起始页发现了 {len(unique_links)} 个疑似疾病链接。")

    # 3. 爬取发现的链接
    for i, url in enumerate(unique_links):
        logging.info(f"抓取发现的链接 ({i+1}/{len(unique_links)}): {url}")
        html = spider.fetch_page(url)
        if html:
            soup = BeautifulSoup(html, 'lxml')
            page_title = soup.title.string.strip() if soup.title else f"discovered_{i}"
            spider.save_html(html, page_title)

if __name__ == "__main__":
    import re # 用于清理文件名

    BASE_URL = 'https://www.yixue.com'
    OUTPUT_DIR = './data'
    DELAY = 7  # 抓取间隔，为了友好访问，请勿设置过小

    spider = MedicalSpider(base_url=BASE_URL, output_dir=OUTPUT_DIR, delay=DELAY)

    print("请选择爬虫模式：")
    print("  1 - 直接爬取指定疾病列表（稳定，推荐）")
    print("  2 - 从疾病列表页自动发现链接（需探索网站结构）")
    choice = input("请输入你的选择 (1 或 2): ").strip()

    if choice == '1':
        mode_direct_crawl(spider)
        logging.info("【直接爬取】模式执行完毕。")
    elif choice == '2':
        # !!! 注意：使用此模式，你需要先找到一个真实的疾病列表页路径 !!!
        # 例如，你可以尝试 '/jibing/' 或 '/category/disease' 等（这需要你手动探索网站）
        LIST_PAGE_PATH = '/疾病/'  # <--- 这个路径是猜测的，可能需要修改！
        mode_auto_discover(spider, LIST_PAGE_PATH)
        logging.info("【自动发现】模式执行完毕。")
    else:
        print("输入无效，程序退出。")