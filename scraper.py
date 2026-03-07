import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3
import re
import time
import random

# 略過安全連線憑證警告以利開發環境進行網頁資料抓取
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scrape_stust_dept_aware(max_pages=2):
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

    # 遍歷所有設定之新聞分類
    for cat_name, target_classid in categories.items():
        print(f"\n📂 開始抓取新聞網分類 ➡️ 【{cat_name}】")
        target_dept = ""

        # 執行各分類之分頁資料請求
        for page in range(1, max_pages + 1):
            if page == 1:
                page_url = f"{base_url}?classid={target_classid}&dept={target_dept}"
            else:
                page_url = f"{base_url}?page={page}&classid={target_classid}&DDLItem=&SCont=&SSdate=&SEdate=&dept={target_dept}"
            
            print(f"📡 正在連線至: 第 {page} 頁 ({page_url}) ...")
            
            try:
                response = session.get(page_url, headers=headers, verify=False, timeout=10)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 尋找目標網頁元素並提取相關文字與連結
                links = soup.find_all("a", href=True)
                page_data_count = 0

                for link in links:
                    title = link.text.strip()
                    href = link['href']
                    
                    # 過濾無效連結與網站導覽項目
                    if not re.search(r'/id/\d+', href): continue
                    if "classid" in href or "Login" in href: continue
                    if len(title) < 5: continue

                    dept_name = "全校" 
                    parent_row = link.find_parent(["tr", "li", "div", "td"]) 
                    
                    if parent_row:
                        grand_parent = parent_row.find_parent(["tr", "li"])
                        search_target = grand_parent if grand_parent else parent_row
                        
                        dept_tag = search_target.find("span", id=re.compile("lbl_sortname"))
                        if dept_tag:
                            raw_dept = dept_tag.text.strip()
                            if "系" in raw_dept or "所" in raw_dept or "學程" in raw_dept:
                                dept_name = raw_dept
                            else:
                                dept_name = "全校" 

                    # 組合相對路徑為完整網址
                    clean_href = href.replace("..", "")
                    if not clean_href.startswith("http"):
                        full_link = "https://news.stust.edu.tw" + ("/" + clean_href if not clean_href.startswith("/") else clean_href)
                    else:
                        full_link = clean_href

                    # 提取公告發布日期
                    date = "近期"
                    if parent_row:
                        text_content = parent_row.parent.get_text() if parent_row.parent else parent_row.get_text()
                        date_match = re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text_content)
                        if date_match: date = date_match.group(0)

                    # 依據標題關鍵字自動配置相關標籤
                    tags = []
                    if "競賽" in title: tags.append("競賽")
                    if "獎學金" in title: tags.append("獎學金")
                    if "實習" in title: tags.append("實習")
                    
                    if cat_name != "所有訊息": 
                        tags.append(cat_name)
                        
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
                
            # 設定隨機延遲時間以避免伺服器負載過高
            if page < max_pages:
                time.sleep(random.uniform(1.0, 2.0))
        
        # 設定分類切換之延遲時間保護目標伺服器
        time.sleep(1.5)

    # 移除重複之活動資料
    return list({v['title']:v for v in all_activities}.values())


def scrape_department_site(url_template, dept_name, default_tags, max_pages=1):
    """建立通用型爬蟲模組支援多系所網頁結構與分頁功能"""
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    activities = []
    
    print(f"\n📂 開始抓取系所來源 ➡️ 【{dept_name}】")

    for page in range(1, max_pages + 1):
        # 替換網址字串中之分頁參數
        if "{}" in url_template:
            url = url_template.format(page)
        else:
            url = url_template
            # 若無分頁參數則結束該網站之抓取流程
            if page > 1: break 
            
        print(f"📡 正在連線至: 第 {page} 頁 ({url}) ...")

        try:
            response = session.get(url, headers=headers, verify=False, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, "html.parser")

            links = soup.find_all("a", href=True)
            page_data_count = 0
            
            for link in links:
                title = link.text.strip()
                href = link['href']

                # 過濾無效連結與網站導覽項目
                if len(title) < 8: continue
                if any(exclude in title for exclude in ["首頁", "網站導覽", "English", "回上一頁", "更多", "登入", "行事曆", "南臺科技大學"]): continue
                if href.startswith("#") or "javascript" in href or "mailto" in href: continue

                # 組合相對路徑為完整網址
                clean_href = href.replace("..", "")
                if not clean_href.startswith("http"):
                    base_domain = "/".join(url.split("/")[:3]) 
                    full_link = base_domain + ("/" + clean_href if not clean_href.startswith("/") else clean_href)
                else:
                    full_link = clean_href

                # 提取公告發布日期
                date = "近期"
                parent = link.find_parent()
                if parent:
                    text_content = parent.get_text()
                    date_match = re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text_content)
                    if date_match: date = date_match.group(0)

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
            
        # 設定隨機延遲時間以降低伺服器負載
        if page < max_pages and "{}" in url_template:
            time.sleep(random.uniform(1.0, 2.0))

    # 移除重複之活動資料
    unique_acts = list({v['title']: v for v in activities}.values())
    print(f"🎉 {dept_name} 全部解析完畢，共找到 {len(unique_acts)} 筆不重複公告。")
    return unique_acts


def fetch_all_sources(max_pages=2):
    """整合各爬蟲模組抓取之資料並進行彙整去重"""
    all_combined_data = []
    
    # 呼叫新聞網主爬蟲模組
    news_data = scrape_stust_dept_aware(max_pages=max_pages)
    all_combined_data.extend(news_data)
    
    # 呼叫資訊工程系網頁爬蟲模組
    csie_data = scrape_department_site(
        url_template="https://csie.stust.edu.tw/?npr=1-{}",
        dept_name="資工系",
        default_tags=["系網公告", "資工", "系所活動"],
        max_pages=max_pages 
    )
    all_combined_data.extend(csie_data)
    
    # 呼叫視覺傳達設計系網頁爬蟲模組
    vc_data = scrape_department_site(
        url_template="https://vc.stust.edu.tw/?npr=209-{}",
        dept_name="視傳系", 
        default_tags=["系網公告", "視傳", "設計", "展覽"],
        max_pages=max_pages
    )
    all_combined_data.extend(vc_data)
    
    # 以網址為鍵值過濾重複之活動資料
    unique_data = list({v['link']: v for v in all_combined_data}.values())
    
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
    # 執行爬蟲測試流程
    data = fetch_all_sources(max_pages=2)
    
    if data:
        df = pd.DataFrame(data)
        # 依據發布日期進行資料排序
        df.sort_values(by="date", ascending=False, inplace=True)
        # 輸出測試資料至本機檔案備查
        df.to_csv("activities_data.csv", index=False, encoding="utf-8-sig")

        print("\n📊 抓取到的前五筆資料預覽：")
        print(df[['dept_target', 'tags', 'title']].head(5))