import requests
from bs4 import BeautifulSoup
import time

# 模拟浏览器请求头，避免被拦截
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 配置参数
LIST_URL = "https://edu.gmw.cn/node_9729.htm"  # 列表页地址
ARTICLE_COUNT = 20                              # 抓取文章数量
OUTPUT_FILE = "光明网教育专栏前20篇文本.txt"    # 输出文件名


def get_article_links(list_url, count):
    """从列表页提取前N篇文章的标题和详情页链接"""
    resp = requests.get(list_url, headers=headers, timeout=10)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []
    # 定位新闻列表条目：光明网频道页通常为 ul>li 结构
    for item in soup.select("ul li a")[:count * 3]:  # 多取一些用于过滤无效链接
        title = item.get_text(strip=True)
        href = item.get("href", "")
        # 过滤空标题、非文章链接
        if not title or len(title) < 5 or "content" not in href:
            continue
        # 拼接完整URL（处理相对路径）
        full_url = href if href.startswith("http") else "https://edu.gmw.cn/" + href.lstrip("/")
        articles.append({"title": title, "url": full_url})
        if len(articles) >= count:
            break
    return articles


def get_article_content(article_url):
    """从详情页提取正文纯文本"""
    try:
        resp = requests.get(article_url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 定位正文容器：光明网正文常见 class/id
        content_div = (
            soup.select_one(".article-content")
            or soup.select_one("#articleContent")
            or soup.select_one(".u-mainText")
            or soup.select_one("#content")
        )
        if not content_div:
            return "【正文提取失败，可手动检查页面结构】"

        # 提取所有段落，过滤空行和无关文本
        paragraphs = [
            p.get_text(strip=True)
            for p in content_div.find_all("p")
            if p.get_text(strip=True) and "来源：" not in p.get_text() and "记者" not in p.get_text()[:5]
        ]
        return "\n".join(paragraphs)

    except Exception as e:
        return f"【页面访问失败：{str(e)}】"


def main():
    print("正在获取文章列表...")
    articles = get_article_links(LIST_URL, ARTICLE_COUNT)
    print(f"共获取 {len(articles)} 篇有效文章，开始抓取正文...\n")

    output_text = (
        f"光明网教育频道（node_9729）前{len(articles)}篇文章汇总\n"
        f"列表来源：{LIST_URL}\n"
        f"整理时间：{time.strftime('%Y-%m-%d')}\n"
        f"{'='*60}\n\n"
    )

    for idx, item in enumerate(articles, 1):
        print(f"[{idx}/{len(articles)}] 正在抓取：{item['title']}")
        content = get_article_content(item["url"])
        output_text += (
            f"第{idx}篇\n"
            f"标题：{item['title']}\n"
            f"链接：{item['url']}\n"
            f"正文：\n{content}\n"
            f"\n{'='*60}\n\n"
        )
        # 控制请求频率，避免给服务器造成压力
        time.sleep(1)

    # 保存为TXT文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"\n抓取完成！文件已保存为：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()