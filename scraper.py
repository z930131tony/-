import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3
import re
import time
import random

# 停用 SSL 警告 (不建議在生產環境使用，但開發爬蟲時可略過學校憑證問題)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scrape_stust_dept_aware(max_pages=2):
    """
    終極版爬蟲：支援【多元分類】與【自動翻頁】，防封鎖機制升級
    :param max_pages: 每個分類預設抓取前 2 頁的資料 (避免一次抓太久被鎖)
    """
    base_url = "https://news.stust.edu.tw/User/RwdNewsList.aspx"
    all_activities = []
    
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # 🌟 多元爬取的靈魂：分類字典
    # ⚠️ 這裡的數字 (2, 3, 4) 是我暫時的猜測！
    # 👉 請你去南臺新聞網點擊左側的「演講與研習」等分類，看網址列的 classid 是多少，然後在這裡替換！
    categories = {
        "所有訊息": "",      
        "學術活動": "0015",
        "校園活動": "0003",
        "工讀機會": "0014",
        "徵才訊息": "0016"
    }

    print(f"🚀 開始執行【多元分類爬蟲】，共 {len(categories)} 個分類，每分類抓取 {max_pages} 頁...")

    # 第一層迴圈：切換分類
    for cat_name, target_classid in categories.items():
        print(f"\n📂 開始抓取分類 ➡️ 【{cat_name}】")
        target_dept = ""

        # 第二層迴圈：在該分類下翻頁
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
                
                links = soup.find_all("a", href=True)
                page_data_count = 0

                for link in links:
                    title = link.text.strip()
                    href = link['href']
                    
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

                    clean_href = href.replace("..", "")
                    if not clean_href.startswith("http"):
                        full_link = "https://news.stust.edu.tw" + ("/" + clean_href if not clean_href.startswith("/") else clean_href)
                    else:
                        full_link = clean_href

                    date = "近期"
                    if parent_row:
                        text_content = parent_row.parent.get_text() if parent_row.parent else parent_row.get_text()
                        date_match = re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text_content)
                        if date_match: date = date_match.group(0)

                    # --- 🌟 標籤邏輯大進化 ---
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
                    
                print(f"✅ 第 {page} 頁解析完畢，找到 {page_data_count} 筆有效公告。")

            except Exception as e:
                print(f"❌ 第 {page} 頁抓取發生錯誤: {e}")
                break 
                
            # 翻頁之間的禮貌性延遲
            if page < max_pages:
                sleep_time = random.uniform(1.5, 3.0)
                print(f"⏳ 休息 {sleep_time:.1f} 秒，準備進入下一頁...")
                time.sleep(sleep_time)
        
        # 分類之間的延遲稍微拉長一點，保護學校伺服器
        print(f"🛑 【{cat_name}】分類抓取完畢，休息 3 秒後切換下一個分類...\n")
        time.sleep(3)

    # 去重
    unique_data = list({v['title']:v for v in all_activities}.values())
    print(f"🎉 全部爬蟲執行完畢！共抓取 {len(unique_data)} 筆不重複的活動資料。")
    return unique_data


def verify_stust_login(username, password):
    """模擬登入南臺校務入口網站"""
    login_url = "https://portal.stust.edu.tw/api/login" 
    
    try:
        response = requests.post(login_url, data={'user_id': username, 'user_pw': password}, timeout=5)
        
        if response.status_code == 200:
            return True, {"dept": "食品系", "name": "王老吉", "year": "大三"}
        return False, None
    except Exception:
        raise ConnectionError("校務系統連線異常")


if __name__ == "__main__":
    # 測試執行：抓取 3 頁
    data = scrape_stust_dept_aware(max_pages=3)
    
    if data:
        df = pd.DataFrame(data)
        # 依日期排序
        df.sort_values(by="date", ascending=False, inplace=True)
        # 存成 CSV (方便開發時檢視，正式環境會存入 SQLite)
        df.to_csv("activities_data.csv", index=False, encoding="utf-8-sig")

        print("\n📊 抓取到的前五筆資料預覽：")
        print(df[['dept_target', 'date', 'title']].head(5))