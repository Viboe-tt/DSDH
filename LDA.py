import jieba
import os
import re
from gensim import corpora, models
import numpy as np

# ==================================================
# 【实验参数配置区】所有设置请在此处修改，并记录到你的实验报告中
# ==================================================
# 1. 数据路径
# 方式A：文件夹模式（每篇文章单独一个 .txt 文件，放在 articles 文件夹下）
DATA_FOLDER = "./articles"
# 方式B：汇总文件模式（爬虫输出的单一大TXT，用分隔线拆分文章）
SUMMARY_FILE = "./光明网教育专栏前20篇文本.txt"
USE_SUMMARY_FILE = True  # True=用汇总文件，False=用文件夹

# 2. 停用词设置
# 基础停用词（内置常用中文停用词，也可替换为哈工大/川大停用词表文件）
BASE_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "这个", "那个", "他", "她", "它", "们", "而", "与", "或", "但",
    "却", "并", "且", "等", "等等", "及", "其", "之", "以", "为", "因", "由于", "因此", "所以", "如果", "虽然", "但是", "然而",
    "可以", "能够", "可能", "应该", "必须", "需要", "进行", "通过", "开展", "推动", "加强", "提升", "打造", "建设", "实现",
    "记者", "来源", "光明网", "本报讯", "近日", "日前", "今年", "以来", "目前", "截至", "据悉", "据介绍", "相关", "进一步"
}
# 自定义停用词（根据你的数据集补充，比如领域高频无意义词）
CUSTOM_STOPWORDS = {"教育", "学校", "学生", "老师", "教师"}  # 可根据结果迭代调整
STOPWORDS = BASE_STOPWORDS.union(CUSTOM_STOPWORDS)

# 3. LDA 模型参数
TOPIC_NUM_LIST = [3, 5, 7]  # 多组主题数，用于对比结果稳定性
TOP_WORDS_NUM = 10          # 每个主题输出的高权重词数量
RANDOM_STATE = 42           # 随机种子，保证结果可复现
DICT_NO_BELOW = 2           # 过滤出现次数少于n次的词
DICT_NO_ABOVE = 0.9         # 过滤出现在超过90%文档中的词

# 输出文件
RESULT_FILE = "./LDA主题建模结果.txt"

# ==================================================
# 模块1：文本数据读取
# ==================================================
def load_articles_from_folder(folder_path):
    """从文件夹读取所有txt文件，返回 标题列表、原文列表 """
    titles, contents = [], []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            titles.append(fname.replace(".txt", ""))
            contents.append(text)
    return titles, contents

def load_articles_from_summary(file_path):
    """从汇总TXT文件拆分文章，适配之前爬虫的输出格式"""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    # 按分隔线拆分文章，适配 "======" 格式的分隔
    blocks = re.split(r"={20,}", full_text)
    titles, contents = [], []
    for block in blocks:
        block = block.strip()
        if not block or "文章汇总" in block or "列表来源" in block:
            continue
        # 提取标题和正文
        title_match = re.search(r"标题：(.+?)\n", block)
        content_match = re.search(r"正文：\n([\s\S]+)", block)
        if title_match and content_match:
            titles.append(title_match.group(1).strip())
            contents.append(content_match.group(1).strip())
    return titles, contents

# ==================================================
# 模块2：分词 + 停用词处理
# ==================================================
def text_process(text):
    """单篇文本预处理：去符号 → 分词 → 去停用词 → 去单字/数字"""
    # 去除标点、数字、特殊符号
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # jieba分词
    words = jieba.lcut(text)
    # 过滤：停用词、单字词、纯空格
    filtered = [
        w for w in words
        if len(w) > 1 and w not in STOPWORDS and not w.isspace()
    ]
    return filtered

def build_corpus(all_words_list):
    """构建gensim词典和BOW语料，同时返回TF-IDF语料可选"""
    dictionary = corpora.Dictionary(all_words_list)
    # 过滤极端词
    dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE)
    # 词袋模型 BOW
    corpus_bow = [dictionary.doc2bow(words) for words in all_words_list]
    # TF-IDF 转换（可选，也可直接用BOW训练LDA）
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    return dictionary, corpus_bow, corpus_tfidf

# ==================================================
# 模块3：LDA 主题建模 + 结果输出
# ==================================================
def train_lda(corpus, dictionary, num_topics):
    """训练LDA模型"""
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=RANDOM_STATE,
        passes=20,
        alpha="auto",
        per_word_topics=True
    )
    return lda_model

def print_topics(lda_model, num_topics, topn, save_to_file=True):
    """输出每个主题的高权重词"""
    result_lines = [f"\n{'='*50}", f"【主题数 K = {num_topics}】", "="*50]
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=topn):
        # 格式化输出：只保留词语，去掉权重数值
        words = [w.split("*")[1].strip('"') for w in topic.split(" + ")]
        line = f"主题{idx+1}：{'、'.join(words)}"
        result_lines.append(line)
        print(line)
    
    if save_to_file:
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(result_lines) + "\n")
    return result_lines

# ==================================================
# 模块4：原文例句抽查（验证主题解释力）
# ==================================================
def check_examples(topic_keywords, titles, contents, top_n=3):
    """
    输入某个主题的关键词列表，在原文中查找包含多个关键词的例句
    用于验证主题是否有原文证据支撑
    """
    print(f"\n【抽查例句】关键词：{'、'.join(topic_keywords)}")
    examples = []
    for title, content in zip(titles, contents):
        sentences = re.split(r"[。！？；\n]", content)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            # 统计句子中包含的关键词数量
            match_count = sum(1 for kw in topic_keywords if kw in sent)
            if match_count >= 2:  # 至少匹配2个关键词
                examples.append((title, sent, match_count))
                break  # 每篇只取最匹配的一句
    
    # 按匹配关键词数量排序，输出前n条
    examples.sort(key=lambda x: x[2], reverse=True)
    for i, (title, sent, cnt) in enumerate(examples[:top_n], 1):
        print(f"{i}. 《{title}》：{sent}")
    if not examples:
        print("未找到匹配例句")
    return examples

# ==================================================
# 主程序执行
# ==================================================
if __name__ == "__main__":
    # 1. 读取文本
    print("正在读取文本数据...")
    if USE_SUMMARY_FILE:
        titles, raw_contents = load_articles_from_summary(SUMMARY_FILE)
    else:
        titles, raw_contents = load_articles_from_folder(DATA_FOLDER)
    print(f"共读取 {len(titles)} 篇文章")

    # 2. 分词与预处理
    print("正在进行分词与停用词处理...")
    word_lists = [text_process(text) for text in raw_contents]

    # 3. 构建词典与语料
    dictionary, corpus_bow, corpus_tfidf = build_corpus(word_lists)
    print(f"词典大小：{len(dictionary)} 个词")

    # 初始化结果文件
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("LDA 主题建模实验记录\n")
        f.write(f"数据集规模：{len(titles)} 篇文章\n")
        f.write(f"停用词：基础停用词 + {CUSTOM_STOPWORDS}\n")
        f.write(f"随机种子：{RANDOM_STATE}\n")
        f.write(f"词典过滤：no_below={DICT_NO_BELOW}, no_above={DICT_NO_ABOVE}\n")

    # 4. 多组主题数训练，对比稳定性
    for k in TOPIC_NUM_LIST:
        print(f"\n正在训练 K={k} 的LDA模型...")
        lda = train_lda(corpus_bow, dictionary, num_topics=k)
        print_topics(lda, k, TOP_WORDS_NUM)

    # 5. 原文抽查示例（以K=5的第1个主题前5个词为例，可自行修改关键词）
    # 先重新获取K=5的模型
    lda_k5 = train_lda(corpus_bow, dictionary, num_topics=5)
    # 提取第1个主题的前5个关键词
    topic1_words = [w.split("*")[1].strip('"') for w in lda_k5.print_topic(0, 5).split(" + ")]
    check_examples(topic1_words, titles, raw_contents)

    print(f"\n全部完成！结果已保存到 {RESULT_FILE}")