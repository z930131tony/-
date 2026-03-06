import streamlit as st
import pandas as pd
import os
import time
import scraper
import database
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

database.init_db()
database.seed_mock_users()

# 1. 資料讀取層 (Data Layer)

@st.cache_data(ttl=1800, show_spinner=" 正在連線學校網站抓取最新活動...")
def fetch_live_data():
    """呼叫 scraper.py 進行即時抓取"""
    try:
        # 直接呼叫 scraper.py 裡面的函數
        data_list = scraper.scrape_stust_dept_aware()
        
        if data_list:
            # 將爬取到的資料寫入資料庫
            database.save_activities_to_db(data_list)
            # 從資料庫讀取最新狀態回傳
            return database.get_all_activities_df()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"爬蟲發生錯誤: {e}")
        return pd.DataFrame()

def load_data():
    """整合資料讀取與清洗"""
    # 呼叫上面的快取函數
    df = fetch_live_data()
    
    if df.empty:
        df = database.get_all_activities_df()
        if df.empty:
            st.error(" 無法抓取資料且資料庫為空。")
            return pd.DataFrame()
        else:
            st.warning(" 即時抓取失敗，顯示資料庫歷史備份資料。")

    try:
       
        df['tags'] = df['tags'].fillna("一般").apply(lambda x: str(x).split(" ") if isinstance(x, str) else x)
        df = enrich_activity_tags(df)

        if 'date' in df.columns:
            df.sort_values(by="date", ascending=False, inplace=True)
            
        return df
    except Exception as e:
        st.error(f"資料清洗錯誤: {e}")
        return pd.DataFrame()
    
# 模擬使用者資料庫
def get_user_info(student_id, password):
    return database.verify_user(student_id, password)


def enrich_activity_tags(df):
    """分析活動標題，自動為缺乏標籤的活動貼上精準的隱藏標籤"""
    
    # 建立「關鍵字 -> 標籤」的對應字典
    tag_rules = {
        "資安": ["資安", "漏洞", "駭客", "密碼", "防護"],
        "AI": ["AI", "人工智慧", "機器學習"],
        "徵才": ["徵才", "招募", "誠徵", "專任助理"],
        "研討會": ["研討會", "論壇", "講座", "說明會"],
        "半導體": ["半導體", "製程", "晶片", "電性"],
        "競賽": ["競賽", "比賽", "黑客松", "挑戰賽"]
    }
    
    def extract_tags(title, existing_tags):
        # 將原本的標籤轉成集合
        tags = set(existing_tags) if isinstance(existing_tags, list) else set([existing_tags])
        
        # 如果只有「一般」，就把它踢掉，因為這標籤沒意義
        if "一般" in tags and len(tags) == 1:
            tags.remove("一般")
            
        # 掃描標題，如果有命中關鍵字，就貼上對應的漂亮標籤
        for tag, keywords in tag_rules.items():
            if any(kw.lower() in str(title).lower() for kw in keywords):
                tags.add(tag)
                
        # 如果掃完還是空的（標題太奇怪），再塞回「一般」或「校園公告」
        if not tags:
            tags.add("校園公告")
            
        return list(tags)

    # 將擴增後的標籤覆蓋回 DataFrame
    df['tags'] = df.apply(lambda row: extract_tags(row['title'], row['tags']), axis=1)
    return df

# 推薦核心 

def get_user_profile_tags(department, past_courses):
    """將使用者背景轉化為能精準對應『學校公告標題』的興趣標籤"""
    user_tags = set()
    
    # 1. 根據系所，賦予常出現在公告標題的高頻關鍵字
    dept_tag_map = {
        "資工": ["資安", "漏洞", "AI", "資訊", "軟體", "網路", "系統"],
        "食品": ["食品", "衛生", "農業", "檢驗", "化學"],
        "企管": ["商業", "行銷", "管理", "創新", "產業"],
        "設計": ["文化", "設計", "藝術", "書展", "展覽"],
        "電機": ["半導體", "製程", "電性", "機電", "工程"],
        "應英": ["雙語", "EMI", "外語", "英文", "國際"]
    }

    for key, tags in dept_tag_map.items():
        if key in department:
            user_tags.update(tags)
            
    # 2. 針對修課紀錄/假資料 (fake_courses) 進行關鍵字轉換
    # 格式：(你模擬的課程關鍵字): "對應到 CSV 標題常出現的字"
    course_tag_map = {
        ("程式", "python", "java"): "軟體服務",
        ("ai", "人工智慧"): "AI",
        ("安全", "密碼"): "資安漏洞預警", # 確保食品系的"安全"不會去撞資安，而是資工的去撞
        ("行銷", "社群"): "商業同業",
        ("設計", "色彩"): "文化部",
        ("物理", "微積分"): "半導體",
    }

    for course in past_courses:
        for keywords, tag in course_tag_map.items():
            if any(kw.lower() in course.lower() for kw in keywords):
                user_tags.add(tag)
                
    # 3. 必殺技：加入大學生強關聯的「通用公告關鍵字」
    # 從你的 CSV 中發現，學校超多徵才跟研討會！加上這些可以大幅解決冷啟動問題
    user_tags.update(["徵才", "招募", "專任助理", "研討會", "獎學金", "計畫", "講座"])

    return user_tags

def calculate_recommendation(user_tags, user_dept, user_id, activities_df):
    """計算推薦分數 (混合式：規則權重 + TF-IDF 語意相似度)"""
    if activities_df.empty:
        return activities_df

    df = activities_df.copy()
    df['score'] = 0.0
    df['match_reason'] = ""
    
    # 將標題、標籤、系所融合成一段長文字
    df['content'] = df['title'] + " " + df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " + df['dept_target']
    
    # 2. 獲取使用者點擊歷史
    clicked_links = database.get_user_clicked_activities(user_id)
    clicked_titles = df[df['link'].isin(clicked_links)]['title'].tolist()
    
    
    # 使用者的系所、硬性標籤、點擊過的活動歷史
    user_profile_text = f"{user_dept} " + " ".join(user_tags) + " " + " ".join(clicked_titles)
    
    # 4. TF-IDF 向量化與 Cosine Similarity 計算
    all_documents = [user_profile_text] + df['content'].tolist()
    
    def jieba_tokenizer(text):
        return jieba.lcut(text)
        
    # 啟動 TF-IDF 轉換器
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        # 計算夾角相似度 (使用者輪廓 [0] 對比 所有活動 [1:])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception as e:
        cosine_sim = np.zeros(len(df)) # 萬一出錯的防呆機制
        
    # 5. 計算總分 (融合 AI 分數與硬性規則)
    for i, (index, row) in enumerate(df.iterrows()):
        rule_score = 0
        reasons = []
        target_dept = str(row['dept_target'])
        
        # --- 規則權重 (Rule-based) ---
        if target_dept == "全校":
            rule_score += 1
        elif user_dept in target_dept or target_dept in user_dept:
            rule_score += 10
            reasons.append(f"👑 {user_dept} 專屬")
            
        activity_tags = set(row['tags']) if isinstance(row['tags'], list) else set()
        overlap = user_tags.intersection(activity_tags)
        if overlap:
            rule_score += len(overlap) * 2
            
        # --- 機器學習權重 (ML-based) ---
        # cosine_sim 通常介於 0~1 之間，我們將其放大作為分數加成
        ml_score = cosine_sim[i] * 15 
        
        # 如果相似度夠高，或者命中標籤，就給予 AI 推薦理由
        if ml_score > 1.5 or overlap:
             reasons.append("🤖 AI 語意推薦")
            
        # 互動回饋：如果已經點擊過了，稍微降分，讓新活動浮上來 (適度探索)
        if row['link'] in clicked_links:
            rule_score -= 3
            reasons.append("👀 您已查看")
            
        # 總分 = 規則分 + AI 語意分
        df.at[index, 'score'] = rule_score + ml_score
        
        if reasons:
            # 移除重複的理由並組裝
            unique_reasons = list(dict.fromkeys(reasons))
            df.at[index, 'match_reason'] = " | ".join(unique_reasons)
        else:
            df.at[index, 'match_reason'] = "✨ 最新活動"
            
    return df.sort_values(by=['score', 'date'], ascending=[False, False])


# 模擬使用者資料庫 (Mock User DB)

def get_user_info(student_id, password):
    
    mock_users = {
        "4B1G0000": {"password": "123", "dept": "食品系", "year": "大三", "name": "王老吉"},
        "4B1G0078": {"password": "123", "dept": "資工系", "year": "大四", "name": "謝宇翔"},
        "4B1G0177": {"password": "123", "dept": "設計系", "year": "大一", "name": "范景程"},
        "admin":    {"password": "admin", "dept": "全校", "year": "教職員", "name": "管理員"}
    }
    if student_id in mock_users and mock_users[student_id]["password"] == password:
        return mock_users[student_id]
    return None


# 前端頁面邏輯 (UI)

st.set_page_config(page_title="南臺科大活動推薦平台", layout="wide")

def login_success(user_info):
    st.session_state['logged_in'] = True
    st.session_state['user_info'] = user_info
    st.success(f"登入成功！歡迎回來 {user_info['name']}")
    time.sleep(1)
    st.rerun()

def get_paged_data(df, page_size, page_key):
    """計算資料切片 (不在這裡畫 UI，只負責算資料)"""
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    
    # 初始化 Session State 來記住當前頁碼
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
        
    # 防呆：如果總資料量變少導致當前頁碼超過總頁數，就退回第一頁
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = 1
        
    current_page = st.session_state[page_key]
    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    
    return df.iloc[start_idx:end_idx], total_pages

def render_google_pagination(total_pages, page_key):
    """在畫面最下方畫出像 Google 一樣的橫向數字分頁列"""
    if total_pages <= 1:
        return # 只有一頁就不顯示分頁列
        
    current_page = st.session_state[page_key]
    st.markdown("<br><br>", unsafe_allow_html=True) # 增加與上方卡片的留白
    
    # 使用 [2, 5, 2] 比例讓分頁按鈕群保持在畫面正中間
    col_spacer1, col_main, col_spacer2 = st.columns([2, 5, 2])
    
    with col_main:
        # 動態計算要顯示的頁碼 (最多顯示 5 顆數字按鈕，確保畫面不會被撐爆)
        start_page = max(1, current_page - 2)
        end_page = min(total_pages, start_page + 4)
        start_page = max(1, end_page - 4)
        
        pages_to_show = list(range(start_page, end_page + 1))
        # 欄位數量 = 數字按鈕數量 + 上一頁 + 下一頁
        btn_cols = st.columns(len(pages_to_show) + 2)
        
        # ◀ 上一頁按鈕
        with btn_cols[0]:
            if st.button("◀", key=f"{page_key}_prev", disabled=(current_page == 1)):
                st.session_state[page_key] -= 1
                st.rerun()
                
        # 數字頁碼按鈕
        for i, p in enumerate(pages_to_show):
            with btn_cols[i+1]:
                if p == current_page:
                    # 當前頁碼：用 primary 藍色凸顯，且點擊不觸發重整
                    st.button(str(p), key=f"{page_key}_cur_{p}", type="primary")
                else:
                    # 其他頁碼：普通樣式，點擊後切換頁面
                    if st.button(str(p), key=f"{page_key}_btn_{p}"):
                        st.session_state[page_key] = p
                        st.rerun()
                        
        # ▶ 下一頁按鈕
        with btn_cols[-1]:
            if st.button("▶", key=f"{page_key}_next", disabled=(current_page == total_pages)):
                st.session_state[page_key] += 1
                st.rerun()

def render_activity_card(row, index, mode="recommend", current_user_id=None):
    """統一的活動卡片渲染函數，確保歷史紀錄與主畫面排版完全一致"""
    full_title = str(row['title']).strip()
    
    # 中英文標題切割邏輯
    regex_pattern = r'^(.*?(?:[\u4e00-\u9fa5」】)）！!？?📢]|\d+))\s*([A-Za-z\[【(（"“][^\u4e00-\u9fa5]*)$'
    match = re.search(regex_pattern, full_title)
    
    if match:
        title_tw = match.group(1).strip()
        title_en = match.group(2).strip()
    else:
        title_tw = full_title
        title_en = ""

    with st.container(border=True):
        # 統一使用 [6, 1] 比例，讓卡片更扁平精緻
        col_text, col_btn = st.columns([6, 1])
        
        with col_text:
            # 如果是歷史紀錄，顯示點擊時間
            if mode == "history":
                st.markdown(f"<small style='color:gray'>🕒 點擊於: {row.get('clicked_time', '')}</small>", unsafe_allow_html=True)
            
            # 顯示 AI 推薦標籤 (僅限推薦模式)
            if mode == "recommend" and 'match_reason' in row and row['match_reason']:
                st.markdown(f"<small style='background-color:#E8F0FE; color:#1967D2; padding:2px 6px; border-radius:4px;'>💡 {row['match_reason']}</small>", unsafe_allow_html=True)
                
            # 標題
            st.markdown(f"#### {title_tw}")
            if title_en:
                st.caption(f"{title_en}")
                
            # 將日期、對象、標籤整合在同一行 (節省高度)
            tags = row['tags'] if isinstance(row['tags'], list) else str(row['tags']).split(" ")
            tag_html = " ".join([f"<span style='color:#007BFF; font-size:12px;'>#{t}</span>" for t in tags])
            dept_target = row.get('dept_target', '全校')
            st.markdown(f"<small style='color:gray;'>📅 {row['date']} | 🎯 {dept_target} | {tag_html}</small>", unsafe_allow_html=True)
            
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) # 垂直置中調整
            if mode == "history":
                st.link_button("重新查看", row['link'], use_container_width=True)
            else:
                if st.button("查看詳情", key=f"btn_{mode}_{index}", use_container_width=True):
                    database.log_interaction(current_user_id, row['link'], "click")
                    st.success("✅ 已紀錄興趣")
                    st.markdown(f"[🔗 點此前往活動網頁]({row['link']})", unsafe_allow_html=True)

def show_history_page(student_id):
    st.title("📚 我的點擊歷史")
    st.write("以下是你最近感興趣查看過的活動：")
    
    conn = database.get_connection()
    query = """
        SELECT a.*, i.timestamp as clicked_time
        FROM user_interactions i
        JOIN activities a ON i.activity_link = a.link
        WHERE i.student_id = ?
        ORDER BY i.timestamp DESC
    """
    history_df = pd.read_sql_query(query, conn, params=(student_id,))
    conn.close()
    
    if history_df.empty:
        st.info("目前還沒有紀錄喔！趕快去首頁看看感興趣的活動吧。")
    else:
        st.caption(f"📅 共找到 {len(history_df)} 筆紀錄")
        
        paged_df, total_pages = get_paged_data(history_df, 10, "history_page")
        
        for index, row in paged_df.iterrows():
            render_activity_card(row, index, mode="history")
            
        render_google_pagination(total_pages, "history_page")

def login_page():
   
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>🎓 南臺科大活動推薦平台</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>請使用校務系統帳號登入</p>", unsafe_allow_html=True)
    
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container(border=True):
            # 這是原本的帳號密碼登入表單
            with st.form("login_form"):
                username = st.text_input("學號 / 教職員編號")
                password = st.text_input("密碼", type="password")
                submit = st.form_submit_button("登入系統", use_container_width=True)
            
            # 這是新增的訪客登入按鈕 (放在 form 的外面)
            st.markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 5px; color: gray; font-size: 14px;'>— 或 —</div>", unsafe_allow_html=True)
            guest_submit = st.button("👤 訪客登入 (免帳號)", use_container_width=True)
            
        # 處理一般登入邏輯
        if submit:
            with st.spinner("正在驗證身分..."):
                
                # --- 修改：完美結合開發者測試通道與模擬資料庫 ---
                # 直接呼叫你上面寫好的 get_user_info 來核對帳號密碼
                test_user_info = get_user_info(username, password)
                
                if test_user_info:
                    # 登入成功，提取需要的欄位 (排除密碼策安全)
                    mock_info = {
                        "dept": test_user_info["dept"], 
                        "name": test_user_info["name"], 
                        "year": test_user_info["year"]
                    }
                    database.sync_user_to_db(username, password, mock_info)
                    login_success({"student_id": username, **mock_info})
                    st.stop()
                try:
                    # 1. 優先嘗試校務系統驗證
                    success, info = scraper.verify_stust_login(username, password)
                    
                    if success:
                        # 登入成功，同步更新本地資料庫 (快取加密密碼)
                        database.sync_user_to_db(username, password, info)
                        user_info = {"student_id": username, **info}
                        
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = user_info
                        st.success(f"校務驗證成功！歡迎 {info['name']}")
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤。")
                        
                except ConnectionError:
                    # 2. 備援模式 (Fallback)：學校網站連不上時，檢查本地 SQLite
                    st.warning("⚠️ 校務系統連線異常，已自動切換至本地備援驗證。")
                    user_info = database.verify_local_user(username, password)
                    
                    if user_info:
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = user_info
                        st.success("本地驗證成功！(離線模式)")
                        st.rerun()
                    else:
                        st.error("本地無此帳號紀錄，請待網路恢復後重試。")
                
                
                
        # 處理訪客登入邏輯
        if guest_submit:
            # 創造一個假的訪客身分
            guest_info = {
                "student_id": "guest",
                "dept": "全校",
                "year": "訪客",
                "name": "訪客"
            }
            st.session_state['logged_in'] = True
            st.session_state['user_info'] = guest_info
            st.success("訪客登入成功！正在轉導...")
            time.sleep(1)
            st.rerun()


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False


if not st.session_state['logged_in']:
    login_page()
else:
    # 主畫面 (登入後) 
    user = st.session_state['user_info']
    
    
    st.sidebar.title(f"Hi, {user['name']}")
    st.sidebar.info(f"身份：{user['dept']} {user['year']}")
    
    if st.sidebar.button("登出系統"):
        st.session_state['logged_in'] = False
        st.rerun()

    menu = st.sidebar.radio("功能選單", ["活動推薦", "我的點擊紀錄"])

    search_keyword = ""
    selected_category = "所有分類"
    
    if menu == "活動推薦":
        st.sidebar.divider()
        st.sidebar.markdown("### 🔍 活動篩選器")
        search_keyword = st.sidebar.text_input("關鍵字搜尋", placeholder="例如：資安、多益...")
        # 這裡的選項對應我們爬蟲抓回來的分類標籤
        selected_category = st.sidebar.selectbox("活動分類", ["所有分類", "學術活動", "校園活動", "工讀機會", "徵才訊息"])

    if menu == "活動推薦":
        st.title(f"專屬推薦：{user['dept']} {user['year']}")
    else:
        
        show_history_page(user.get('student_id', 'guest'))

    if user.get('student_id') == 'admin' or user.get('name') == '管理員':
        with st.expander("🔧 管理員後台：查看 SQLite 資料庫原始資料", expanded=False):
            st.info(f"目前資料庫檔案位置：{os.path.abspath('stust_recommendation.db')}")
            
            # 讀取並顯示 Users 資料表
            st.write("👤 **使用者資料表 (Users)**")
            conn = database.get_connection()
            df_users = pd.read_sql_query("SELECT * FROM users", conn)
            st.dataframe(df_users, use_container_width=True)
            
            # 讀取並顯示 Activities 資料表
            st.write("📅 **活動資料表 (Activities)**")
            df_acts = pd.read_sql_query("SELECT * FROM activities", conn)
            st.dataframe(df_acts, use_container_width=True)
            
            # 顯示點擊紀錄
            st.write("🖱️ **使用者點擊紀錄 (Interactions)**")
            st.dataframe(pd.read_sql_query("SELECT * FROM user_interactions", conn), use_container_width=True)
            
            conn.close()

        st.divider()
    
    df_activities = load_data()
    
    if not df_activities.empty:
        fake_courses = []
        if "資工" in user['dept']: fake_courses = ["程式", "AI"]
        if "食品" in user['dept']: fake_courses = ["食品", "安全"]
        
        user_interest_tags = get_user_profile_tags(user['dept'], fake_courses)
        current_user_id = user.get('student_id', 'guest')

        recommended_df = calculate_recommendation(user_interest_tags, user['dept'], current_user_id, df_activities)
        
        if search_keyword:
            # 如果標題或標籤包含關鍵字，才保留該活動
            recommended_df = recommended_df[
                recommended_df['title'].str.contains(search_keyword, case=False, na=False) | 
                recommended_df['tags'].apply(lambda x: search_keyword in (x if isinstance(x, list) else str(x)))
            ]
            
        if selected_category != "所有分類":
            # 必須包含使用者選定的分類標籤才保留
            recommended_df = recommended_df[
                recommended_df['tags'].apply(lambda x: selected_category in (x if isinstance(x, list) else str(x)))
            ]

        # 透過檢查 match_reason 裡面有沒有包含 "您已查看" 來分類
        is_viewed = recommended_df['match_reason'].str.contains("您已查看", na=False)
        viewed_df = recommended_df[is_viewed]
        unviewed_df = recommended_df[~is_viewed]
        
        # 獨立顯示「您已查看」區塊 
        if not viewed_df.empty:
            st.markdown("### 👀 最近已查看")
            
            # 為了不讓主畫面太長最多只顯示 3 筆最新看過的
            for index, row in viewed_df.head(3).iterrows():
                render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
                
            if len(viewed_df) > 3:
                st.caption(f"👉 還有 {len(viewed_df) - 3} 筆紀錄，可至左側「我的點擊紀錄」分頁查看完整清單。")
                
            st.divider() # 分隔線
            
        # 顯示「專屬推薦」區塊 
        st.markdown("### ✨ 專屬推薦")
        st.write(f"為您找到 {len(unviewed_df)} 筆未查看的活動：")
        
        # 分頁
        paged_rec_df, rec_total_pages = get_paged_data(unviewed_df, 10, "recommend_page")
        
        for index, row in paged_rec_df.iterrows():
            render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
            
        render_google_pagination(rec_total_pages, "recommend_page")
    else:
        st.warning(" 資料庫為空，請確認後端爬蟲狀態。")