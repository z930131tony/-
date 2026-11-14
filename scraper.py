import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3
import re
import time
from urllib.parse import urljoin

# 略過安全連線憑證警告以利開發環境進行網頁資料抓取
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EXCLUDE_KEYWORDS = [
    "首頁", "網站導覽", "english", "回上一頁", "更多", "登入", "行事曆", "南臺科技大學",
    "隱私權政策", "privacy policy", "免試入學", "相關連結", "版權所有", "資訊安全政策",
    "個人資料保護", "校園地圖", "交通資訊", "聯絡我們", "常見問題", "faq", "系統操作手冊",
    "法規辦法", "檔案下載"
]

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

DATE_RE = re.compile(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})')
# 發布單位需以「系 / 研究所 / 學程」結尾才視為系所，避免「圖資處網路系統組」這類行政單位被誤判
DEPT_RE = re.compile(r'(系|研究所|學程)$')

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

def scrape_stust_dept_aware(max_pages=2, progress_callback=None, step_offset=0, total_steps=14):
    """建立主爬蟲模組負責抓取新聞網頁資訊"""
    base_url = "https://news.stust.edu.tw/User/RwdNewsList.aspx"
    all_activities = []
    
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 建立新聞網分類與代碼對應表
    categories = {
        "所有訊息": "",      
        "學術活動": "0015",
        "校園活動": "0003",
        "工讀機會": "0014",
        "徵才訊息": "0016"
    }

    print(f"🚀 開始執行【新聞網分類爬蟲】，共 {len(categories)} 個分類，每分類 {max_pages} 頁...")
    current_step = step_offset

    # 遍歷所有設定之新聞分類
    for cat_name, target_classid in categories.items():
        print(f"\n📂 開始抓取新聞網分類 ➡️ 【{cat_name}】")
        target_dept = ""

        # 執行各分類之分頁資料請求
        for page in range(1, max_pages + 1):
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"正在抓取新聞網【{cat_name}】第 {page}/{max_pages} 頁")

            if page == 1:
                page_url = f"{base_url}?classid={target_classid}&dept={target_dept}"
            else:
                page_url = f"{base_url}?page={page}&classid={target_classid}&DDLItem=&SCont=&SSdate=&SEdate=&dept={target_dept}"
            
            print(f"📡 正在連線至: 第 {page} 頁 ({page_url}) ...")
            
            try:
                response = session.get(page_url, headers=headers, verify=False, timeout=8)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 尋找目標網頁元素並提取相關文字與連結
                links = soup.find_all("a", href=True)
                page_data_count = 0

                for link in links:
                    href = link['href']
                    # 列表的 <a> 內含 <br/> 與多餘空白，title 屬性才是完整標題
                    title = " ".join((link.get("title") or link.get_text()).split())

                    # 過濾無效連結與網站導覽項目
                    if not re.search(r'/id/\d+', href): continue
                    if "classid" in href: continue
                    if not is_valid_activity(title, href): continue

                    title = clean_activity_title(title)

                    # 每筆公告是一列 <tr>：發布單位、類別、日期都在同列的 <span> 裡
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

                    all_activities.append({
                        "title": title,
                        "date": date,
                        "link": full_link,
                        "tags": " ".join(tags),
                        "dept_target": dept_name, 
                        "img": "🏫"
                    })
                    page_data_count += 1
                    
                print(f"✅ 第 {page} 頁解析完畢，找到 {page_data_count} 筆。")

            except Exception as e:
                print(f"❌ 第 {page} 頁抓取發生錯誤: {e}")
                break 
                
            # 設定輕量延遲保護目標伺服器
            if page < max_pages:
                time.sleep(0.3)
        
        time.sleep(0.3)

    # 同一則公告會同時出現在「所有訊息」與其分類頁：以連結為鍵合併標籤
    return _merge_duplicates(all_activities, lambda v: v['link']), current_step


def scrape_department_site(url_template, dept_name, default_tags, max_pages=1, progress_callback=None, step_offset=0, total_steps=14):
    """建立通用型爬蟲模組支援多系所網頁結構與分頁功能"""
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    activities = []
    
    print(f"\n📂 開始抓取系所來源 ➡️ 【{dept_name}】")
    current_step = step_offset

    for page in range(1, max_pages + 1):
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, f"正在抓取系所【{dept_name}】第 {page}/{max_pages} 頁")

        if "{}" in url_template:
            url = url_template.format(page)
        else:
            url = url_template
            if page > 1: break 
            
        print(f"📡 正在連線至: 第 {page} 頁 ({url}) ...")

        try:
            response = session.get(url, headers=headers, verify=False, timeout=8)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, "html.parser")

            links = soup.find_all("a", href=True)
            page_data_count = 0
            
            for link in links:
                href = link['href']

                # 只收公告連結 (/tc/news/1-96678-n)，排除選單中「沿革與規模」「學生獲獎114年」等導覽頁
                if not re.search(r'/news/\d+-\d+', href): continue

                # 公告列是整個 <a class="item">：第一個 <p> 是標題，其餘欄位為單位/類別/點閱數/日期
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
                    if container and len(container.find_all("a", href=re.compile(r'/news/\d+-\d+'))) == 1:
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
                page_data_count += 1
                
            print(f"✅ 第 {page} 頁解析完畢，找到 {page_data_count} 筆公告。")
            
        except Exception as e:
            print(f"❌ {dept_name} 第 {page} 頁抓取發生錯誤: {e}")
            break
            
        if page < max_pages and "{}" in url_template:
            time.sleep(0.3)

    unique_acts = list({v['link']: v for v in activities}.values())
    print(f"🎉 {dept_name} 全部解析完畢，共找到 {len(unique_acts)} 筆不重複公告。")
    return unique_acts, current_step


def fetch_all_sources(max_pages=2, progress_callback=None):
    """整合各爬蟲模組抓取之資料並進行彙整去重"""
    total_steps = (5 * max_pages) + (2 * max_pages) # 5類新聞 + 資工 + 視傳
    current_step = 0
    all_combined_data = []
    
    # 1. 呼叫新聞網主爬蟲模組
    news_data, current_step = scrape_stust_dept_aware(
        max_pages=max_pages, 
        progress_callback=progress_callback, 
        step_offset=current_step, 
        total_steps=total_steps
    )
    all_combined_data.extend(news_data)
    
    # 2. 呼叫資訊工程系網頁爬蟲模組
    csie_data, current_step = scrape_department_site(
        url_template="https://csie.stust.edu.tw/?npr=1-{}",
        dept_name="資工系",
        default_tags=["系網公告", "資工", "系所活動"],
        max_pages=max_pages,
        progress_callback=progress_callback,
        step_offset=current_step,
        total_steps=total_steps
    )
    all_combined_data.extend(csie_data)
    
    # 3. 呼叫視覺傳達設計系網頁爬蟲模組
    vc_data, current_step = scrape_department_site(
        url_template="https://vc.stust.edu.tw/?npr=209-{}",
        dept_name="視傳系", 
        default_tags=["系網公告", "視傳", "設計", "展覽"],
        max_pages=max_pages,
        progress_callback=progress_callback,
        step_offset=current_step,
        total_steps=total_steps
    )
    all_combined_data.extend(vc_data)
    
    # 系網公告與新聞網是同一則公告的不同網址 (csie .../news/1-96678-n ↔ news .../id/96678)，
    # 以「標題 + 日期」合併，避免同一活動在推薦清單出現兩次
    unique_data = _merge_duplicates(all_combined_data, lambda v: (v['title'], v['date']))
    
    print(f"\n🎉 所有來源抓取完畢！總共整合了 {len(unique_data)} 筆不重複資料。")
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
