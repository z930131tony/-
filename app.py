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
import streamlit.components.v1 as components
import joblib
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai

# 🔐 專業做法：從 Streamlit 的安全保險箱讀取 API Key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except Exception as e:
    # 如果找不到 API Key，系統依然能正常開啟，只是 AI 對話框可能會報錯
    st.warning("⚠️ 找不到 API Key！請確認已設定 .streamlit/secrets.toml 或 Streamlit Cloud Secrets。")

database.init_db()
database.seed_mock_users()

# ==========================================
# 1. 資料讀取層
# ==========================================
@st.cache_data(ttl=1800,show_spinner=False)
def fetch_live_data(): 
    """呼叫外部模組進行即時抓取"""
    try:
        data_list = scraper.scrape_stust_dept_aware()
        if data_list:
            database.save_activities_to_db(data_list)
            return database.get_all_activities_df()
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"爬蟲發生錯誤: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="🧠 正在載入 AI 語意大腦，初次啟動請稍候...")
def load_semantic_model():
    """載入支援多國語言與中文的輕量級語意向量模型"""
    return SentenceTransformer('shibing624/text2vec-base-chinese')

def load_data():
    """整合資料讀取與清洗，並顯示客製化南臺小知識載入畫面"""
    placeholder = st.empty()
    
    html_code = """
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 25px; background-color: #E8F0FE; border-radius: 12px; border: 1px solid #D2E3FC;">
        <h3 style="color: #1967D2; margin-top: 0;">⏳ 正在連線學校網站抓取最新活動...</h3>
        <p style="color: #3C4043; font-size: 16px; margin-bottom: 0;">
            💡 <b>南臺冷知識：</b> <span id="fact-text">載入中...</span>
        </p>
    </div>
    <script>
        const facts = [
            "南臺科大創立於 1969 年，前身為「南台工業技藝專科學校」。",
            "南臺科大校訓為「信義誠實」。",
            "磅礴館的外觀設計靈感來自於「巨輪」，象徵乘風破浪、航向未來。",
            "校園內的天鵝池不僅是地標，更是許多南臺人共同的回憶與約會聖地。",
            "南臺科大圖書館館藏豐富，是南部私立科大中資源最充沛的圖書館之一。",
            "每年的校慶活動與迎新演唱會，都會在具代表性的「三連堂」盛大舉辦。",
            "南臺科大校地面積約 16.46 公頃，是一所充滿活力的綠色校園。",
            "南臺科大在各項產學合作與創新創業競賽中，常年名列全國私立科大前茅。"
        ];
        let currentIndex = Math.floor(Math.random() * facts.length);
        document.getElementById("fact-text").innerText = facts[currentIndex];
        setInterval(function() {
            let nextIndex;
            do {
                nextIndex = Math.floor(Math.random() * facts.length);
            } while (nextIndex === currentIndex);
            currentIndex = nextIndex;
            document.getElementById("fact-text").innerText = facts[currentIndex];
        }, 10000);
    </script>
    """
    with placeholder:
        components.html(html_code, height=130)

    df = fetch_live_data()
    placeholder.empty()

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

# ==========================================
# 2. 推薦核心與特徵工程
# ==========================================
def enrich_activity_tags(df):
    tag_rules = {
        "資安": ["資安", "漏洞", "駭客", "密碼", "防護"],
        "AI": ["AI", "人工智慧", "機器學習"],
        "徵才": ["徵才", "招募", "誠徵", "專任助理", "職缺", "面試"],
        "實習": ["實習", "intern"],
        "研討會": ["研討會", "論壇", "講座", "說明會"],
        "半導體": ["半導體", "製程", "晶片", "電性"],
        "競賽": ["競賽", "比賽", "黑客松", "挑戰賽"]
    }
    
    def extract_tags(title, existing_tags):
        tags = set(existing_tags) if isinstance(existing_tags, list) else set([existing_tags])
        if "一般" in tags and len(tags) == 1:
            tags.remove("一般")
        for tag, keywords in tag_rules.items():
            if any(kw.lower() in str(title).lower() for kw in keywords):
                tags.add(tag)
        if not tags:
            tags.add("校園公告")
        return list(tags)

    df['tags'] = df.apply(lambda row: extract_tags(row['title'], row['tags']), axis=1)
    return df

def get_user_profile_tags(department, past_courses, user_year=""):
    user_tags = set()
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
            
    course_tag_map = {
        ("程式", "python", "java"): "軟體服務",
        ("ai", "人工智慧"): "AI",
        ("安全", "密碼"): "資安漏洞預警", 
        ("行銷", "社群"): "商業同業",
        ("設計", "色彩"): "文化部",
        ("物理", "微積分"): "半導體",
    }
    for course in past_courses:
        for keywords, tag in course_tag_map.items():
            if any(kw.lower() in course.lower() for kw in keywords):
                user_tags.add(tag)
                
    if any(y in user_year for y in ["三", "四", "碩"]):
        user_tags.update(["實習", "徵才", "招募", "專任助理", "就業", "企業", "職缺"])
    elif any(y in user_year for y in ["一", "二"]):
        user_tags.update(["社團", "營隊", "志工", "迎新", "校園活動", "講座", "工讀"])

    user_tags.update(["研討會", "獎學金", "計畫", "講座"])
    return user_tags

def calculate_recommendation(user_tags, user_dept, user_id, activities_df, user_year=""):
    if activities_df.empty: return activities_df

    df = activities_df.copy()
    df['score'] = 0.0
    df['match_reason'] = ""

    ctr_model = None
    if os.path.exists("ctr_model.pkl"):
        ctr_model = joblib.load("ctr_model.pkl")
    
    df['content'] = df['title'] + " " + df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " + df['dept_target']
    
    clicked_links = database.get_user_clicked_activities(user_id)
    clicked_titles = df[df['link'].isin(clicked_links)]['title'].tolist()
    
    user_profile_text = f"{user_dept} {user_year} " + " ".join(user_tags) + " " + " ".join(clicked_titles)
    all_documents = [user_profile_text] + df['content'].tolist()
    
    def jieba_tokenizer(text):
        return jieba.lcut(text)
        
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        cosine_sim = np.zeros(len(df))
        
    for i, (index, row) in enumerate(df.iterrows()):
        rule_score = 0
        reasons = []
        target_dept = str(row['dept_target'])
        
        if target_dept == "全校":
            rule_score += 1
        elif user_dept in target_dept or target_dept in user_dept:
            rule_score += 10
            reasons.append(f"👑 {user_dept} 專屬")
            
        activity_tags = set(row['tags']) if isinstance(row['tags'], list) else set()
        overlap = user_tags.intersection(activity_tags)
        if overlap:
            rule_score += len(overlap) * 2
            
        if any(y in user_year for y in ["三", "四", "碩"]):
            activity_text = str(row['tags']) + str(row['title'])
            if any(kw in activity_text for kw in ["實習", "徵才", "就業", "招募", "企業", "職缺"]):
                rule_score += 15 
                reasons.append("💼 就業實習推薦")
                
        ml_score = cosine_sim[i] * 15 
        if ml_score > 1.5 or overlap:
             reasons.append("🤖 AI 語意推薦")
            
        if row['link'] in clicked_links:
            rule_score -= 3
            reasons.append("👀 您已查看")
            
        predict_score = 0
        if ctr_model:
            dept_match = 1 if (user_dept in target_dept or target_dept == "全校") else 0
            overlap_count = len(overlap)
            is_senior = 1 if any(y in user_year for y in ["三", "四", "碩"]) else 0
            is_internship = 1 if any(kw in str(row['tags']) for kw in ["實習", "徵才", "就業"]) else 0

            features = [[dept_match, overlap_count, is_senior * is_internship]]
            proba_result = ctr_model.predict_proba(features)[0]
            if len(proba_result) > 1:
                click_prob = proba_result[1]
            else:
                click_prob = 1.0 if ctr_model.classes_[0] == 1 else 0.0

            predict_score = click_prob * 20
            if click_prob > 0.6:
                reasons.append("🎯 AI 點擊率預測極高")

        df.at[index, 'score'] = rule_score + ml_score + predict_score
        
        if reasons:
            unique_reasons = list(dict.fromkeys(reasons))
            df.at[index, 'match_reason'] = " | ".join(unique_reasons)
        else:
            df.at[index, 'match_reason'] = "✨ 最新活動"

    return df.sort_values(by=['score', 'date'], ascending=[False, False])

# ==========================================
# 3. Agentic RAG 與技能 (Skill) 定義
# ==========================================
def skill_auto_register(student_id, activity_title):
    """這是一個 Skill：模擬 AI 自動幫學生報名活動，並寫入資料庫"""
    time.sleep(1.5) # 模擬網路延遲與報名處理
    return f"✅ 技能執行成功：已成功為學號 {student_id} 報名【{activity_title}】！已同步至校務系統。"

def render_agentic_rag_chat(user_info, recommended_df):
    """繪製具有檢索與行動能力的 AI 聊天室"""
    st.title("🤖 校園 AI 導覽員 (Agentic RAG)")
    st.info("我不只會用語意搜尋推薦活動，只要跟我說一聲，我還能啟動【Skill】直接幫你報名喔！")
    
    # 初始化對話紀錄與技能狀態
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_skill_activity" not in st.session_state:
        st.session_state.pending_skill_activity = None

    # 顯示歷史對話
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 技能觸發確認介面 (Persistent UI) ---
    if st.session_state.pending_skill_activity:
        st.divider()
        st.markdown("#### 🛠️ AI 請求授權執行技能：代客報名")
        target_act = st.session_state.pending_skill_activity
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button(f"⚡ 允許報名", type="primary"):
                with st.spinner(f"🤖 正在為您調用報名 Skill，寫入資料庫中..."):
                    skill_result = skill_auto_register(user_info.get('student_id', 'Unknown'), target_act)
                    st.success(skill_result)
                    st.balloons()
                    # 執行完畢後重置狀態
                    st.session_state.pending_skill_activity = None
                    time.sleep(2)
                    st.rerun()
        with col_btn2:
            if st.button("❌ 取消"):
                st.session_state.pending_skill_activity = None
                st.rerun()
        st.divider()

    # 接收使用者聊天輸入
    if prompt := st.chat_input("例如：我想找跟程式有關的活動，並幫我報名第一個！"):
        # 1. 顯示並儲存使用者的問題
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. 檢索與生成回覆
        with st.chat_message("assistant"):
            with st.spinner("🧠 思考中，並檢索南臺活動庫..."):
                # RAG 檢索 (Retrieval)：抓取分數最高的前 3 筆活動
                top_3 = recommended_df.head(3) if not recommended_df.empty else pd.DataFrame()
                context_str = ""
                for idx, row in top_3.iterrows():
                    context_str += f"- 活動：{row['title']}, 標籤：{row['tags']}\n"
                
                # RAG 生成 (Generation)：組合 Prompt 呼叫 LLM
                system_prompt = f"""
                你現在扮演南臺科大最罩、最熱心的學長/學姊。
                來找你諮詢的是 {user_info['dept']} 的 {user_info['name']}。
                
                【絕對遵守的規則】
                1. 說話語氣要自然、活潑，像大學生平常傳 LINE 聊天一樣，絕對不要像死板的客服機器人。
                2. 你【只能】根據以下的 [最新活動清單] 來回答問題，絕對不可以自己憑空捏造活動！
                3. 如果學生問的問題，在清單中找不到相關活動，請誠實跟他說「目前沒看到相關的耶」，然後順勢推薦清單裡的其他活動。
                4. 如果學生明確表示想要報名，請在回答的最後加上一句：「沒問題，我馬上幫你處理報名！」

                [最新活動清單]
                {context_str}
                """
                
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(system_prompt + "\n學生問：" + prompt)
                    ai_reply = response.text
                except Exception as e:
                    ai_reply = f"抱歉，我的大腦連線異常，請確認 API Key 是否設定正確：{e}"

                # 顯示 AI 回答
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})

                # Skill 意圖觸發判斷
                if any(keyword in prompt for keyword in ["報名", "參加", "加一", "+1", "報這個"]):
                    target = top_3.iloc[0]['title'] if not top_3.empty else "未知活動"
                    st.session_state.pending_skill_activity = target
                    st.rerun() # 重新整理以顯示上方的技能確認按鈕

# ==========================================
# 4. 模擬使用者資料庫 (帳號管理)
# ==========================================
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

def login_success(user_info):
    st.session_state['logged_in'] = True
    st.session_state['user_info'] = user_info
    st.success(f"登入成功！歡迎回來 {user_info['name']}")
    time.sleep(1)
    st.rerun()

# ==========================================
# 5. 前端介面與渲染組件
# ==========================================
st.set_page_config(page_title="南臺科大活動推薦平台", layout="wide")

def get_paged_data(df, page_size, page_key):
    total_pages = max(1, (len(df) - 1) // page_size + 1)
    if page_key not in st.session_state: st.session_state[page_key] = 1
    if st.session_state[page_key] > total_pages: st.session_state[page_key] = 1
    current_page = st.session_state[page_key]
    start_idx = (current_page - 1) * page_size
    return df.iloc[start_idx:start_idx + page_size], total_pages

def render_google_pagination(total_pages, page_key):
    if total_pages <= 1: return 
    current_page = st.session_state[page_key]
    st.markdown("<br>", unsafe_allow_html=True) 
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 上一頁", key=f"{page_key}_prev", disabled=(current_page == 1), use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
    with col_info:
        st.markdown(f"<div style='text-align: center; padding-top: 8px; color: #555;'>第 <b>{current_page}</b> / {total_pages} 頁</div>", unsafe_allow_html=True)
    with col_next:
        if st.button("下一頁 ▶", key=f"{page_key}_next", disabled=(current_page == total_pages), use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()

def render_activity_card(row, index, mode="recommend", current_user_id=None):
    full_title = str(row['title']).strip()
    regex_pattern = r'^(.*?)\s*([a-zA-Z\[\(][^\u4e00-\u9fa5]*)$'
    match = re.search(regex_pattern, full_title)
    if match and match.group(1).strip():
        title_tw, title_en = match.group(1).strip(), match.group(2).strip()
    else:
        title_tw, title_en = full_title, ""

    with st.container(border=True):
        col_text, col_btn = st.columns([6, 1])
        with col_text:
            if mode == "history":
                st.markdown(f"<small style='color:gray'>🕒 點擊於: {row.get('clicked_time', '')}</small>", unsafe_allow_html=True)
            if mode == "recommend" and 'match_reason' in row and row['match_reason']:
                st.markdown(f"<small style='background-color:#E8F0FE; color:#1967D2; padding:2px 6px; border-radius:4px;'>💡 {row['match_reason']}</small>", unsafe_allow_html=True)
            st.markdown(f"#### {title_tw}")
            if title_en: st.caption(f"{title_en}")
            tags = row['tags'] if isinstance(row['tags'], list) else str(row['tags']).split(" ")
            tag_html = " ".join([f"<span style='color:#007BFF; font-size:12px;'>#{t}</span>" for t in tags])
            dept_target = row.get('dept_target', '全校')
            st.markdown(f"<small style='color:gray;'>📅 {row['date']} | 🎯 {dept_target} | {tag_html}</small>", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True) 
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
            with st.form("login_form"):
                username = st.text_input("學號 / 教職員編號")
                password = st.text_input("密碼", type="password")
                submit = st.form_submit_button("登入系統", use_container_width=True)
            
            st.markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 5px; color: gray; font-size: 14px;'>— 或 —</div>", unsafe_allow_html=True)
            guest_submit = st.button("👤 訪客登入 (免帳號)", use_container_width=True)
            
        if submit:
            with st.spinner("正在驗證身分..."):
                test_user_info = get_user_info(username, password)
                if test_user_info:
                    mock_info = {"dept": test_user_info["dept"], "name": test_user_info["name"], "year": test_user_info["year"]}
                    database.sync_user_to_db(username, password, mock_info)
                    login_success({"student_id": username, **mock_info})
                    st.stop()
                try:
                    success, info = scraper.verify_stust_login(username, password)
                    if success:
                        database.sync_user_to_db(username, password, info)
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = {"student_id": username, **info}
                        st.success(f"校務驗證成功！歡迎 {info['name']}")
                        st.rerun()
                    else:
                        st.error("帳號或密碼錯誤。")
                except ConnectionError:
                    st.warning("⚠️ 校務系統連線異常，已自動切換至本地備援驗證。")
                    user_info = database.verify_local_user(username, password)
                    if user_info:
                        st.session_state['logged_in'] = True
                        st.session_state['user_info'] = user_info
                        st.success("本地驗證成功！(離線模式)")
                        st.rerun()
                    else:
                        st.error("本地無此帳號紀錄，請待網路恢復後重試。")
                        
        if guest_submit:
            guest_info = {"student_id": "guest", "dept": "全校", "year": "訪客", "name": "訪客"}
            st.session_state['logged_in'] = True
            st.session_state['user_info'] = guest_info
            st.success("訪客登入成功！正在轉導...")
            time.sleep(1)
            st.rerun()

# ==========================================
# 6. 主程式執行入口與選單路由
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    user = st.session_state['user_info']
    current_user_id = user.get('student_id', 'guest')
    
    st.sidebar.title(f"Hi, {user['name']}")
    st.sidebar.info(f"身份：{user['dept']} {user['year']}")
    
    if st.sidebar.button("登出系統"):
        st.session_state['logged_in'] = False
        st.rerun()

    # 🌟 核心選單新增 AI 導覽員
    menu = st.sidebar.radio("功能選單", ["活動推薦", "🤖 AI 導覽員", "我的點擊紀錄"])

    if user.get('student_id') == 'admin' or user.get('name') == '管理員':
        with st.expander("🔧 管理員後台：查看 SQLite 資料庫原始資料", expanded=False):
            st.info(f"目前資料庫檔案位置：{os.path.abspath('stust_recommendation.db')}")
            conn = database.get_connection()
            st.write("👤 **使用者資料表 (Users)**")
            st.dataframe(pd.read_sql_query("SELECT * FROM users", conn), use_container_width=True)
            st.write("📅 **活動資料表 (Activities)**")
            st.dataframe(pd.read_sql_query("SELECT * FROM activities", conn), use_container_width=True)
            st.write("🖱️ **使用者點擊紀錄 (Interactions)**")
            st.dataframe(pd.read_sql_query("SELECT * FROM user_interactions", conn), use_container_width=True)
            conn.close()

    # 🌟 資料統一在這裡載入，供所有分頁共用 (解決歷史紀錄底下出現推薦卡片的 Bug)
    df_activities = load_data()
    recommended_df = pd.DataFrame()
    
    if not df_activities.empty:
        fake_courses = []
        if "資工" in user['dept']: fake_courses = ["程式", "AI"]
        if "食品" in user['dept']: fake_courses = ["食品", "安全"]
        user_interest_tags = get_user_profile_tags(user['dept'], fake_courses, user.get('year', ''))
        recommended_df = calculate_recommendation(user_interest_tags, user['dept'], current_user_id, df_activities, user.get('year', ''))

    st.divider()
    
    # --- 根據選單顯示對應畫面 ---
    if menu == "活動推薦":
        st.sidebar.divider()
        st.sidebar.markdown("### 🔍 活動篩選器")
        search_keyword = st.sidebar.text_input("關鍵字搜尋", placeholder="例如：資安、多益...")
        selected_category = st.sidebar.selectbox("活動分類", ["所有分類", "學術活動", "校園活動", "工讀機會", "徵才訊息"])

        st.title(f"專屬推薦：{user['dept']} {user['year']}")
        
        if not recommended_df.empty:
            if search_keyword:
                with st.spinner(f"🧠 正在以 AI 語意分析您的需求：「{search_keyword}」..."):
                    semantic_model = load_semantic_model()
                    query_embedding = semantic_model.encode(search_keyword, convert_to_tensor=True)
                    activity_texts = (recommended_df['title'] + " " + 
                                    recommended_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))).tolist()
                    corpus_embeddings = semantic_model.encode(activity_texts, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    recommended_df['semantic_score'] = cos_scores.cpu().numpy()
                    recommended_df = recommended_df[
                        (recommended_df['semantic_score'] > 0.35) | 
                        (recommended_df['title'].str.contains(search_keyword, case=False, na=False))
                    ]
                    recommended_df = recommended_df.sort_values(by=['semantic_score', 'score'], ascending=[False, False])
                    for idx in recommended_df.index:
                        recommended_df.at[idx, 'match_reason'] = "🕵️ AI 語意精準命中"
            
            if selected_category != "所有分類":
                recommended_df = recommended_df[
                    recommended_df['tags'].apply(lambda x: selected_category in (x if isinstance(x, list) else str(x)))
                ]

            is_viewed = recommended_df['match_reason'].str.contains("您已查看", na=False)
            viewed_df = recommended_df[is_viewed]
            unviewed_df = recommended_df[~is_viewed]
            
            if not viewed_df.empty:
                st.markdown("### 👀 最近已查看")
                for index, row in viewed_df.head(3).iterrows():
                    render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
                if len(viewed_df) > 3:
                    st.caption(f"👉 還有 {len(viewed_df) - 3} 筆紀錄，可至左側「我的點擊紀錄」分頁查看完整清單。")
                st.divider() 
                
            st.markdown("### ✨ 專屬推薦")
            st.write(f"為您找到 {len(unviewed_df)} 筆未查看的活動：")
            paged_rec_df, rec_total_pages = get_paged_data(unviewed_df, 10, "recommend_page")
            for index, row in paged_rec_df.iterrows():
                render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
            render_google_pagination(rec_total_pages, "recommend_page")
        else:
            st.warning(" 資料庫為空，請確認後端爬蟲狀態。")

    elif menu == "🤖 AI 導覽員":
        # 呼叫 Agentic RAG 模組
        if not recommended_df.empty:
            render_agentic_rag_chat(user, recommended_df)
        else:
            st.warning("目前沒有活動資料，AI 導覽員無法為您服務。")

    elif menu == "我的點擊紀錄":
        # 顯示歷史紀錄分頁 (乾淨無重疊)
        show_history_page(current_user_id)
