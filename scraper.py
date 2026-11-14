import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import urllib3
import re
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

# ==========================================
# 爬蟲設定
# ==========================================
NEWS_BASE = "https://news.stust.edu.tw/User/RwdNewsList.aspx"

# 新聞網分類與代碼對應表
NEWS_CATEGORIES = {
    "所有訊息": "",
    "學術活動": "0015",
    "校園活動": "0003",
    "工讀機會": "0014",
    "徵才訊息": "0016"
}

# 有獨立公告頁的系網 (網址中的 {} 會代入頁碼)
DEPT_SITES = [
    ("https://csie.stust.edu.tw/?npr=1-{}", "資工系", ["系網公告", "資工", "系所活動"]),
    ("https://vc.stust.edu.tw/?npr=209-{}", "視傳系", ["系網公告", "視傳", "設計", "展覽"]),
]

MAX_AGE_DAYS = 365      # 超過一年的公告不收 (各系置頂的舊公告會一直留在第一頁)
REQUEST_DELAY = 0.3     # 同一個任務連續翻頁之間的延遲，保護學校伺服器
MAX_WORKERS = 4         # 同時進行的抓取任務數
TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

EXCLUDE_KEYWORDS = [
    "首頁", "網站導覽", "english", "回上一頁", "更多", "登入", "行事曆", "南臺科技大學",
    "隱私權政策", "privacy policy", "免試入學", "相關連結", "版權所有", "資訊安全政策",
    "個人資料保護", "校園地圖", "交通資訊", "聯絡我們", "常見問題", "faq", "系統操作手冊",
    "法規辦法", "檔案下載"
]

DATE_RE = re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})')
# 發布單位需以「系 / 研究所 / 學程」結尾才視為系所，避免「圖資處網路系統組」這類行政單位被誤判
DEPT_RE = re.compile(r'(系|研究所|學程)$')
DEPT_NEWS_LINK_RE = re.compile(r'/news/\d+-\d+')


# ==========================================
# 共用工具
# ==========================================
def is_valid_activity(title, href):
    """過濾導覽、頁腳與無效公告連結"""
    if not title or len(title.strip()) < 5:
        return False
    lower_t = title.lower()
    if any(kw in lower_t for kw in EXCLUDE_KEYWORDS):
        return False
    if href.startswith("#") or "javascript" in href or "mailto:" in href or "Login" in href:
        return False
    return True

def clean_activity_title(title):
    """去除標題末尾的系所麵包屑字串 (例如：資工系 單位公告 151 2026/11/12)"""
    title = " ".join(str(title).split())
    cleaned = re.sub(r'\s*(\S+系)?\s*(單位公告|系上公告)\s*\d*\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})?\s*$', '', title)
    return cleaned.strip()

def normalize_date(text):
    """擷取第一個日期並統一為 YYYY/MM/DD (補零，字串排序才正確)；找不到回傳 None"""
    m = DATE_RE.search(str(text or ''))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}/{int(mo):02d}/{int(d):02d}"

def _merge_duplicates(items, key_fn):
    """依 key 合併重複公告：聯集標籤，並優先保留有明確系所的 dept_target"""
    merged = {}
    for v in items:
        key = key_fn(v)
        if key not in merged:
            merged[key] = dict(v)
            continue
        cur = merged[key]
        tags = [t for t in cur['tags'].split() + v['tags'].split() if t]
        tags = list(dict.fromkeys(tags))
        if len(tags) > 1 and '一般' in tags:
            tags.remove('一般')
        cur['tags'] = ' '.join(tags)
        if cur.get('dept_target') == '全校' and v.get('dept_target') != '全校':
            cur['dept_target'] = v['dept_target']
        if cur.get('date') == '近期' and v.get('date') != '近期':
            cur['date'] = v['date']
    return list(merged.values())

def make_session():
    """建立帶自動重試的連線：連線失敗或 429/5xx 會以 0.5s、1s、2s 退避重試"""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = USER_AGENT
    return session

def fetch_html(session, url):
    """下載網頁並檢查狀態碼 (錯誤頁不再被當成「0 筆公告」默默略過)。
    預設驗證 SSL 憑證，只有憑證驗證失敗時才退回不驗證。"""
    try:
        response = session.get(url, timeout=TIMEOUT)
    except requests.exceptions.SSLError:
        print(f"⚠️ SSL 憑證驗證失敗，改以不驗證模式連線：{url}")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = session.get(url, timeout=TIMEOUT, verify=False)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.text

def news_url(classid="", dept="", page=1):
    if page == 1:
        return f"{NEWS_BASE}?classid={classid}&dept={dept}"
    return f"{NEWS_BASE}?page={page}&classid={classid}&DDLItem=&SCont=&SSdate=&SEdate=&dept={dept}"


# ==========================================
# 頁面解析 (純函式，可離線用存好的 HTML 測試)
# ==========================================
def parse_news_list(html, page_url, cat_name="所有訊息"):
    """解析新聞網列表頁：每筆公告是一列 <tr>，發布單位、類別、日期都在同列的 <span> 裡"""
    soup = BeautifulSoup(html, "html.parser")
    activities = []

    for link in soup.find_all("a", href=True):
        href = link['href']
        # 列表的 <a> 內含 <br/> 與多餘空白，title 屬性才是完整標題
        title = " ".join((link.get("title") or link.get_text()).split())

        # 過濾無效連結與網站導覽項目
        if not re.search(r'/id/\d+', href): continue
        if "classid" in href: continue
        if not is_valid_activity(title, href): continue
        title = clean_activity_title(title)

        row = link.find_parent("tr") or link.find_parent("li")

        def row_span(suffix):
            tag = row.find("span", id=re.compile(suffix)) if row else None
            return " ".join(tag.get_text().split()) if tag else ""

        raw_dept = row_span("lbl_sortname")
        dept_name = raw_dept if DEPT_RE.search(raw_dept) else "全校"
        class_name = row_span("lbl_classname")

        # 組合相對路徑為完整網址 (../id/123 相對於 /User/ 目錄)
        full_link = urljoin(page_url, href)

        # 只從同列的發布日期欄位取日期，避免誤抓標題中的截止日或別列的日期
        date = normalize_date(row_span("lbl_time"))
        if not date and row:
            for td_cell in row.find_all("td"):
                if td_cell.find("a") is not None:
                    continue
                if re.fullmatch(r'\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*', td_cell.get_text()):
                    date = normalize_date(td_cell.get_text())
                    break
        date = date or "近期"

        # 依據標題關鍵字自動配置相關標籤
        tags = []
        if "競賽" in title: tags.append("競賽")
        if "獎學金" in title: tags.append("獎學金")
        if "實習" in title: tags.append("實習")
        if cat_name != "所有訊息":
            tags.append(cat_name)
        # 列表本身就標示了類別 (如 研討會 / 徵才訊息)，「所有訊息」頁也能分類
        if class_name and class_name not in tags:
            tags.append(class_name)
        if dept_name != "全校": tags.append("系所活動")
        if not tags: tags.append("一般")

        activities.append({
            "title": title,
            "date": date,
            "link": full_link,
            "tags": " ".join(tags),
            "dept_target": dept_name,
            "img": "🏫"
        })
    return activities

def parse_dept_options(html):
    """從新聞網的「發布單位」下拉選單取出所有系所 (代碼, 名稱)，
    系所增減或改名時不必改程式"""
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", {"name": "dept"})
    if not select:
        return []
    options = []
    for opt in select.find_all("option"):
        code, name = (opt.get("value") or "").strip(), opt.get_text(strip=True)
        if code and DEPT_RE.search(name):
            options.append((code, name))
    return options

def parse_department_site(html, url, dept_name, default_tags):
    """解析系網公告列表：每筆公告是一個 <a class="item">，第一個 <p> 是標題，其餘欄位為單位/類別/點閱數/日期"""
    soup = BeautifulSoup(html, "html.parser")
    activities = []

    for link in soup.find_all("a", href=True):
        href = link['href']

        # 只收公告連結 (/tc/news/1-96678-n)，排除選單中「沿革與規模」「學生獲獎114年」等導覽頁
        if not DEPT_NEWS_LINK_RE.search(href): continue

        first_p = link.find("p")
        title = " ".join((first_p.get_text() if first_p else link.get_text()).split())
        if not is_valid_activity(title, href): continue
        title = clean_activity_title(title)

        full_link = urljoin(url, href)

        # 日期只在這筆公告自己的欄位內找 (跳過標題，避免抓到標題裡的活動日期)，
        # 也避免整個列表共用第一筆的日期
        date = None
        for p in reversed(link.find_all("p")[1:]):
            date = normalize_date(p.get_text())
            if date: break
        if not date and not first_p:
            container = link.parent
            if container and len(container.find_all("a", href=DEPT_NEWS_LINK_RE)) == 1:
                date = normalize_date(container.get_text(" "))
        date = date or "近期"

        activities.append({
            "title": title,
            "date": date,
            "link": full_link,
            "tags": " ".join(default_tags),
            "dept_target": dept_name,
            "img": "🌐"
        })
    return activities


# ==========================================
# 抓取任務
# ==========================================
def crawl_pages(label, url_for_page, parse, max_pages):
    """依序抓取 1..max_pages 頁；某頁沒有任何公告 (已到最後一頁) 就提早停止"""
    session = make_session()
    results = []
    for page in range(1, max_pages + 1):
        url = url_for_page(page)
        if url is None:
            break
        try:
            items = parse(fetch_html(session, url), url)
        except Exception as e:
            print(f"❌ {label} 第 {page} 頁抓取失敗: {e}")
            break
        print(f"✅ {label} 第 {page} 頁：{len(items)} 筆")
        if not items:
            break
        results.extend(items)
        if page < max_pages:
            time.sleep(REQUEST_DELAY)
    return results

def build_tasks(max_pages, include_dept_news=True):
    """列出所有抓取任務 (標籤, 函式)"""
    tasks = []
    for cat_name, classid in NEWS_CATEGORIES.items():
        tasks.append((f"新聞網【{cat_name}】", lambda c=cat_name, cid=classid: crawl_pages(
            f"新聞網【{c}】", lambda p: news_url(classid=cid, page=p),
            lambda html, url: parse_news_list(html, url, c), max_pages)))

    for template, dept_name, tags in DEPT_SITES:
        tasks.append((f"系網【{dept_name}】", lambda t=template, d=dept_name, tg=tags: crawl_pages(
            f"系網【{d}】", lambda p: t.format(p) if "{}" in t else (t if p == 1 else None),
            lambda html, url: parse_department_site(html, url, d, tg), max_pages)))

    if include_dept_news:
        # 「所有訊息」只涵蓋全校最新幾十則，各系自己的公告幾乎抓不到；
        # 新聞網可依發布單位篩選，逐系抓第一頁 (各系公告量少，再往後多半是過期公告)
        try:
            options = parse_dept_options(fetch_html(make_session(), news_url()))
        except Exception as e:
            print(f"❌ 無法取得系所清單，略過各系公告: {e}")
            options = []
        for code, name in options:
            tasks.append((f"新聞網系所【{name}】", lambda c=code, n=name: crawl_pages(
                f"新聞網系所【{n}】", lambda p: news_url(dept=c, page=p),
                lambda html, url: parse_news_list(html, url), 1)))
    return tasks

def _is_recent(item, cutoff):
    """日期不明 (近期) 的保留；有日期的需在 cutoff 之後"""
    try:
        return datetime.strptime(item['date'], "%Y/%m/%d") >= cutoff
    except (ValueError, TypeError):
        return True

def fetch_all_sources(max_pages=2, progress_callback=None, include_dept_news=True,
                      max_age_days=MAX_AGE_DAYS, max_workers=MAX_WORKERS):
    """整合新聞網分類、各系新聞與系網公告，並行抓取後彙整去重"""
    print(f"🚀 開始抓取 (每分類 {max_pages} 頁，並行 {max_workers} 個任務)...")
    started = time.time()
    tasks = build_tasks(max_pages, include_dept_news)
    total = len(tasks)
    results = [[] for _ in tasks]

    if progress_callback:
        progress_callback(0, total, f"準備抓取 {total} 個來源")

    # progress_callback 只在主執行緒呼叫 (Streamlit 元件不能從背景執行緒更新)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): (i, label) for i, (label, fn) in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), start=1):
            i, label = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"❌ {label} 發生錯誤: {e}")
            if progress_callback:
                progress_callback(done, total, f"已完成 {label} ({done}/{total})")

    # 依任務順序合併 (結果與完成先後無關)。
    # 系網與新聞網是同一則公告的不同網址 (csie .../news/1-96678-n ↔ news .../id/96678)，
    # 以「標題 + 日期」合併，避免同一活動在推薦清單出現兩次
    combined = [item for chunk in results for item in chunk]
    unique_data = _merge_duplicates(combined, lambda v: (v['title'], v['date']))

    if max_age_days:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        before = len(unique_data)
        unique_data = [v for v in unique_data if _is_recent(v, cutoff)]
        print(f"🧹 移除 {before - len(unique_data)} 筆超過 {max_age_days} 天的舊公告")

    print(f"\n🎉 所有來源抓取完畢！{total} 個來源，共 {len(unique_data)} 筆不重複資料，耗時 {time.time() - started:.1f} 秒。")
    return unique_data


def verify_stust_login(username, password):
    """模擬校務系統登入驗證流程"""
    login_url = "https://portal.stust.edu.tw/api/login"

    try:
        response = requests.post(login_url, data={'user_id': username, 'user_pw': password}, timeout=5)

        if response.status_code == 200:
            return True, {"dept": "食品系", "name": "王老吉", "year": "大三"}
        return False, None
    except Exception:
        raise ConnectionError("校務系統連線異常")


if __name__ == "__main__":
    data = fetch_all_sources(max_pages=2)
    if data:
        df = pd.DataFrame(data)
        df.sort_values(by="date", ascending=False, inplace=True)
        df.to_csv("activities_data.csv", index=False, encoding="utf-8-sig")
        print("\n📊 抓取到的前五筆資料預覽：")
        print(df[['dept_target', 'tags', 'title']].head(5))
