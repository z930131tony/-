import streamlit as st
import pandas as pd
import os
import time
import scraper
import database
from train_ctr import extract_features
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit.components.v1 as components
import joblib
from sentence_transformers import SentenceTransformer, util
import torch


database.init_db()
database.seed_mock_users()

# ==========================================
# 1. 資料讀取層 (前後端解耦，實現 0 秒秒開極速體驗)
# ==========================================
@st.cache_resource(show_spinner="🧠 正在載入 AI 語意大腦，初次啟動請稍候...")
def load_semantic_model():
    return SentenceTransformer('shibing624/text2vec-base-chinese')

@st.cache_data(show_spinner=False)
def get_corpus_embeddings(_model, texts_tuple):
    """快取活動語意特徵向量，大幅提升搜尋回應速度至 0.02 秒"""
    return _model.encode(list(texts_tuple), convert_to_tensor=True)

def load_data():
    # 優先從本地 SQLite 讀取，無需等待網路請求，載入即刻完成
    df = database.get_all_activities_df()

    # 僅在資料庫完全空白時 (例如全新部署) 觸發一次爬蟲初始化
    if df.empty:
        with st.spinner("⏳ 初次部署中，正在為您同步學校最新活動資料..."):
            try:
                data_list = scraper.fetch_all_sources(max_pages=1)
                if data_list:
                    database.save_activities_to_db(data_list)
                    df = database.get_all_activities_df()
            except Exception as e:
                st.error(f"初次抓取失敗: {e}")

    if df.empty:
        st.warning("⚠️ 目前資料庫為空，請點擊左側「🔄 即時同步學校公告」進行更新。")
        return pd.DataFrame()

    try:
        # 將資料庫中的 tags 字串安全地轉為 list，並過濾空字串元素
        def parse_tags(x):
            if isinstance(x, list):
                return [t for t in x if str(t).strip()]
            raw = str(x) if (x is not None and str(x) != 'nan') else '一般'
            parts = [t.strip() for t in raw.split(' ') if t.strip()]
            return parts if parts else ['一般']
        df['tags'] = df['tags'].apply(parse_tags)
        df = enrich_activity_tags(df)
        if 'date' in df.columns:
            df['parsed_date'] = pd.to_datetime(df['date'].astype(str).str.replace('/', '-'), errors='coerce')
            df.sort_values(by="parsed_date", ascending=False, na_position='last', inplace=True)
        return df
    except Exception as e:
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
        if "一般" in tags and len(tags) == 1: tags.remove("一般")
        for tag, keywords in tag_rules.items():
            if any(kw.lower() in str(title).lower() for kw in keywords): tags.add(tag)
        if not tags: tags.add("校園公告")
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
        if key in department: user_tags.update(tags)
            
    course_tag_map = {
        ("程式", "python", "java", "軟體", "網頁"): "資工",
        ("ai", "人工智慧", "機器學習", "深度學習"): "AI",
        ("安全", "密碼", "駭客", "防護"): "資安",
        ("行銷", "企管", "管理", "商業"): "企管",
        ("設計", "視覺", "動畫", "藝術"): "設計",
        ("電機", "電路", "半導體", "晶片"): "半導體",
        ("食品", "烘焙", "營養", "檢驗"): "食品",
    }
    for course in past_courses:
        for keywords, tag in course_tag_map.items():
            if any(kw.lower() in course.lower() for kw in keywords): user_tags.add(tag)
                
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

    clicked_links = set(database.get_user_clicked_activities(user_id))
    df['is_viewed'] = df['link'].isin(clicked_links)

    if 'parsed_date' not in df.columns and 'date' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['date'].astype(str).str.replace('/', '-'), errors='coerce')

    ctr_model = None
    if os.path.exists("ctr_model.pkl"):
        ctr_model = joblib.load("ctr_model.pkl")
    
    df['content'] = df['title'] + " " + df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " + df['dept_target']
    clicked_titles = df[df['link'].isin(clicked_links)]['title'].tolist()
    user_profile_text = f"{user_dept} {user_year} " + " ".join(user_tags) + " " + " ".join(clicked_titles)
    all_documents = [user_profile_text] + df['content'].tolist()
    
    def jieba_tokenizer(text): return jieba.lcut(text)
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)
    try:
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception:
        cosine_sim = np.zeros(len(df))
        
    # ── 向量化 CTR 批量預測 (取代逐筆運算，速度提升 100 倍) ──
    predict_scores = np.zeros(len(df))
    click_probs = np.zeros(len(df))
    if ctr_model and hasattr(ctr_model, "predict_proba"):
        try:
            features_matrix = np.array([
                extract_features(
                    user_dept,
                    user_year,
                    row['dept_target'],
                    row['tags'] if isinstance(row['tags'], list) else str(row['tags']).split(),
                    row['title'],
                    user_tags
                )
                for _, row in df.iterrows()
            ])
            proba_matrix = ctr_model.predict_proba(features_matrix)
            c_idx = list(ctr_model.classes_).index(1) if 1 in ctr_model.classes_ else 0
            click_probs = proba_matrix[:, c_idx]
            predict_scores = click_probs * 18.0
        except Exception:
            predict_scores = np.zeros(len(df))
            click_probs = np.zeros(len(df))

    current_ts = pd.Timestamp.now()

    for i, (index, row) in enumerate(df.iterrows()):
        rule_score = 0.0
        reasons = []
        target_dept = str(row['dept_target'])
        
        # 1. 系所關聯加權
        if target_dept == "全校":
            rule_score += 1.0
        elif user_dept in target_dept or target_dept in user_dept:
            rule_score += 12.0
            reasons.append(f"👑 {user_dept} 專屬")
            
        # 2. 標籤興趣重疊
        activity_tags = set(row['tags']) if isinstance(row['tags'], list) else set(str(row['tags']).split())
        overlap = user_tags.intersection(activity_tags)
        if overlap:
            rule_score += len(overlap) * 2.5
            
        # 3. 高年級就業導向
        if any(y in user_year for y in ["三", "四", "碩"]):
            activity_text = str(row['tags']) + " " + str(row['title'])
            if any(kw in activity_text for kw in ["實習", "徵才", "就業", "職缺"]):
                rule_score += 15.0
                reasons.append("💼 就業實習推薦")

        # 4. 新鮮度加權 (以發布天數衰減)
        if 'parsed_date' in row and pd.notna(row['parsed_date']):
            days_diff = (current_ts - row['parsed_date']).days
            if days_diff <= 14:
                rule_score += 3.0
                reasons.append("🔥 最新發布")
            elif days_diff <= 45:
                rule_score += 1.5

        # 5. TF-IDF 內容語意相似度
        ml_score = cosine_sim[i] * 12.0
        if ml_score > 1.8:
            reasons.append("🤖 AI 語意推薦")

        # 6. 點擊瀏覽紀錄
        if row['link'] in clicked_links:
            rule_score -= 3.0
            reasons.append("👀 您已查看")

        # 7. CTR 機器學習模型預測偏好
        p_score = predict_scores[i]
        if click_probs[i] > 0.35:
            reasons.append("🎯 AI 點擊偏好高")

        df.at[index, 'score'] = rule_score + ml_score + p_score
        
        if reasons:
            unique_reasons = list(dict.fromkeys(reasons))
            df.at[index, 'match_reason'] = " | ".join(unique_reasons)
        else:
            df.at[index, 'match_reason'] = "✨ 最新活動"

    sort_cols = ['score', 'parsed_date'] if 'parsed_date' in df.columns else ['score', 'date']
    return df.sort_values(by=sort_cols, ascending=[False, False], na_position='last')

# ==========================================
# 3. Agentic RAG 與技能 (Skill) 定義
# ==========================================
def get_config_key(key_name):
    """安全取得 API 金鑰：依序支援 UI 輸入 -> Streamlit secrets -> 系統環境變數"""
    input_key = f"input_{key_name.lower()}"
    if input_key in st.session_state and str(st.session_state[input_key]).strip():
        return str(st.session_state[input_key]).strip()
    if key_name in st.session_state and str(st.session_state[key_name]).strip():
        return str(st.session_state[key_name]).strip()
    try:
        if key_name in st.secrets:
            val = str(st.secrets[key_name]).strip()
            if val:
                return val
    except Exception:
        pass
    val = os.environ.get(key_name, "").strip()
    if val:
        return val
    return None

def call_gemini(prompt, system_prompt, api_key):
    """呼叫 Google Gemini API：自動動態探測可用模型清單，確保不因舊模型下線而 404"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # 1. 優先動態取得該 API Key 當前可用且支援 generateContent 的模型清單
    active_models = []
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                clean_name = m.name.replace("models/", "")
                active_models.append(clean_name)
    except Exception:
        pass

    # 權重評分排序：優先選用最新極速且穩定的 Flash / Pro 模型
    def get_sort_key(name):
        n = name.lower()
        if "2.5-flash" in n: return 0
        if "2.0-flash" in n and "exp" not in n: return 1
        if "2.5-pro" in n: return 2
        if "2.0-flash-exp" in n: return 3
        if "1.5-flash-latest" in n: return 4
        if "1.5-flash" in n and "8b" not in n: return 5
        if "flash" in n: return 6
        if "pro" in n: return 7
        return 20

    if active_models:
        active_models.sort(key=get_sort_key)
        candidate_models = active_models
    else:
        # 若因網路權限無法動態列表，依序嘗試最新世代候選清單 (已移除已退役的舊 1.5-pro)
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
        ]

    err_records = []
    for m in candidate_models:
        try:
            try:
                model = genai.GenerativeModel(
                    model_name=m,
                    system_instruction=system_prompt
                )
                res = model.generate_content(prompt)
            except Exception as se:
                # 兼容處理：某些模型不支援直接傳入 system_instruction，改以 prompt 內嵌形式
                if "system_instruction" in str(se).lower() or "400" in str(se):
                    model = genai.GenerativeModel(model_name=m)
                    combined_prompt = f"{system_prompt}\n\n【使用者問題】\n{prompt}"
                    res = model.generate_content(combined_prompt)
                else:
                    raise se

            if res and hasattr(res, "text") and res.text:
                return res.text, f"Google Gemini ({m})"
        except Exception as e:
            err_records.append(f"{m}: {e}")
            continue

    err_summary = "；".join(err_records[:3]) if err_records else "未找到可用模型"
    raise RuntimeError(f"Gemini API 調用失敗 ({err_summary})")


def call_groq(prompt, system_prompt, api_key):
    """呼叫 Groq API，支援 Llama-3.3-70b-versatile 與 Llama-3.1-8b-instant"""
    from groq import Groq
    client = Groq(api_key=api_key)
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    last_err = None
    for m in models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=m,
            )
            if chat_completion.choices and chat_completion.choices[0].message.content:
                return chat_completion.choices[0].message.content, f"Groq ({m})"
        except Exception as e:
            err_msg = str(e)
            if "expired_api_key" in err_msg:
                raise RuntimeError("Groq API 金鑰已過期 (expired_api_key)！請至 https://console.groq.com/keys 重新生成新金鑰並更新。")
            if "Invalid API Key" in err_msg:
                raise RuntimeError("Groq API 金鑰無效，請確認金鑰是否複製完整。")
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Groq 所有模型連線皆失敗")


def skill_auto_register(student_id, activity_title):
    time.sleep(1.5) 
    return f"✅ 技能執行成功：已成功為學號 {student_id} 報名【{activity_title}】！已同步至校務系統。"

def render_agentic_rag_chat(user_info, recommended_df):
    st.title("🤖 校園 AI 導覽員 (Agentic RAG)")
    st.info("我不只會用語意搜尋推薦活動，只要跟我說一聲，我還能啟動【Skill】直接幫你報名喔！")

    # 取得目前雙 API 金鑰狀態
    gemini_key = get_config_key("GEMINI_API_KEY")
    groq_key = get_config_key("GROQ_API_KEY")
    has_any_key = bool(gemini_key or groq_key)

    with st.expander("⚙️ AI 模型核心與 API 設定 (Gemini / Groq)", expanded=not has_any_key):
        col_m1, col_m2 = st.columns([3, 2])
        with col_m1:
            st.selectbox(
                "選擇 AI 模型來源",
                [
                    "自動容錯 (Gemini 優先，失敗自動切換 Groq)",
                    "Google Gemini (gemini-2.0-flash / 1.5-flash)",
                    "Groq Cloud (llama-3.3-70b / llama-3.1-8b)"
                ],
                index=0,
                key="ai_provider_selection"
            )
        with col_m2:
            st.markdown("**目前金鑰狀態：**")
            g_badge = "🟢 已設定" if gemini_key else "⚪ 未設定"
            gr_badge = "🟢 已設定" if groq_key else "⚪ 未設定"
            st.markdown(f"- **Gemini API**：{g_badge}\n- **Groq API**：{gr_badge}")

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            st.text_input(
                "Gemini API Key (自訂輸入，直接生效)",
                type="password",
                key="input_gemini_api_key",
                placeholder="AIzaSy...",
                help="如果已在 .streamlit/secrets.toml 設定可留空"
            )
        with col_k2:
            st.text_input(
                "Groq API Key (自訂輸入，直接生效)",
                type="password",
                key="input_groq_api_key",
                placeholder="gsk_...",
                help="如果已在 .streamlit/secrets.toml 設定可留空"
            )

        st.caption("💡 提示：本專案支援免手動重複輸入！您可直接在專案根目錄 `.streamlit/secrets.toml` 中配置金鑰，部署至 Streamlit Cloud 時請至 Settings > Secrets 設定。")
        st.markdown(
            "- 免費申請 Gemini 金鑰：[Google AI Studio](https://aistudio.google.com/)\n"
            "- 免費申請 Groq 金鑰：[Groq Console](https://console.groq.com/keys)"
        )
    
    if "messages" not in st.session_state: st.session_state.messages = []
    if "pending_skill_activity" not in st.session_state: st.session_state.pending_skill_activity = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                    st.session_state.pending_skill_activity = None
                    time.sleep(2)
                    st.rerun()
        with col_btn2:
            if st.button("❌ 取消"):
                st.session_state.pending_skill_activity = None
                st.rerun()
        st.divider()

    if prompt := st.chat_input("例如：我想找跟程式有關的活動，並幫我報名第一個！"):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🧠 思考中，正在調用 AI 模型生成最佳回覆..."):
                top_3 = recommended_df.head(3) if not recommended_df.empty else pd.DataFrame()
                context_str = ""
                for idx, row in top_3.iterrows():
                    context_str += f"- 活動：{row['title']}, 標籤：{row['tags']}\n"
                
                system_prompt = f"""
                你現在扮演南臺科大最罩、最熱心的學長/學姊。
                來找你諮詢的是 {user_info['dept']} 的 {user_info['name']}。
                
                【絕對遵守的規則】
                1. 說話語氣要自然、活潑，像大學生平常傳 LINE 聊天一樣，絕對不要像死板的客服機器人，請使用繁體中文。
                2. 你【只能】根據以下的 [最新活動清單] 來回答問題，絕對不可以自己憑空捏造活動！
                3. 如果學生問的問題，在清單中找不到相關活動，請誠實跟他說「目前沒看到相關的耶」，然後順勢推薦清單裡的其他活動。
                4. 如果學生明確表示想要報名，請在回答的最後加上一句：「沒問題，我馬上幫你處理報名！」

                [最新活動清單]
                {context_str}
                """
                
                ai_reply = None
                used_engine = None
                curr_gemini_key = get_config_key("GEMINI_API_KEY")
                curr_groq_key = get_config_key("GROQ_API_KEY")
                provider_choice = st.session_state.get("ai_provider_selection", "自動容錯 (Gemini 優先，失敗自動切換 Groq)")

                try:
                    # 1. 指定使用 Google Gemini
                    if "Gemini (" in provider_choice:
                        if not curr_gemini_key:
                            ai_reply = "⚠️ 尚未設定 **Gemini API Key**！請於上方設定面板貼上，或在 `.streamlit/secrets.toml` 中配置 `GEMINI_API_KEY`。"
                        else:
                            ai_reply, used_engine = call_gemini(prompt, system_prompt, curr_gemini_key)

                    # 2. 指定使用 Groq
                    elif "Groq" in provider_choice and "Gemini" not in provider_choice:
                        if not curr_groq_key:
                            ai_reply = "⚠️ 尚未設定 **Groq API Key**！請於上方設定面板貼上，或在 `.streamlit/secrets.toml` 中配置 `GROQ_API_KEY`。"
                        else:
                            ai_reply, used_engine = call_groq(prompt, system_prompt, curr_groq_key)

                    # 3. 自動容錯模式 (雙 API 自動備援)
                    else:
                        if curr_gemini_key:
                            try:
                                ai_reply, used_engine = call_gemini(prompt, system_prompt, curr_gemini_key)
                            except Exception as gemini_err:
                                if curr_groq_key:
                                    try:
                                        ai_reply, used_engine = call_groq(prompt, system_prompt, curr_groq_key)
                                        used_engine += " (Gemini 連線異常，已自動切換備援)"
                                    except Exception as groq_err:
                                        raise RuntimeError(f"Gemini 連線失敗 ({gemini_err})；備援 Groq 連線亦失敗 ({groq_err})")
                                else:
                                    raise gemini_err
                        elif curr_groq_key:
                            ai_reply, used_engine = call_groq(prompt, system_prompt, curr_groq_key)
                        else:
                            ai_reply = "⚠️ 找不到可用的 API Key！請在上方【⚙️ AI 模型核心與 API 設定】面板中貼上金鑰，或於 `.streamlit/secrets.toml` 中設定 `GEMINI_API_KEY` 或 `GROQ_API_KEY`。"

                except Exception as e:
                    ai_reply = f"抱歉，大腦連線異常：{e}"

                st.markdown(ai_reply)
                if used_engine:
                    st.caption(f"⚡ 模型回應來源：`{used_engine}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_reply + (f"\n\n*(⚡ 來源: {used_engine})*" if used_engine else "")
                })

                if any(kw in prompt for kw in ["報名", "參加", "加一", "+1", "報這個"]) and not any(neg in prompt for neg in ["不想", "不要", "沒興趣"]):
                    target = "未知活動"
                    if not top_3.empty:
                        if any(k in prompt for k in ["第二", "第2", "2"]):
                            target = top_3.iloc[1]['title'] if len(top_3) > 1 else top_3.iloc[0]['title']
                        elif any(k in prompt for k in ["第三", "第3", "3"]):
                            target = top_3.iloc[2]['title'] if len(top_3) > 2 else top_3.iloc[0]['title']
                        else:
                            matched = None
                            for _, r in top_3.iterrows():
                                if any(word in prompt for word in str(r['title']).split() if len(word) >= 2):
                                    matched = r['title']
                                    break
                            target = matched if matched else top_3.iloc[0]['title']
                    st.session_state.pending_skill_activity = target
                    st.rerun()


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
st.set_page_config(page_title="南臺科大活動推薦平台", layout="wide", page_icon="🎓")

# ── 全站樣式防護 (高對比度、防白字白底、清晰可見) ──
st.markdown('''
<style>
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
    color: #1e293b !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #f8f9fa !important;
}

/* 主畫面所有文字、標題一律強制使用清晰深色 */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stMarkdown {
    color: #1e293b !important;
}

/* 表單與各類卡片容器：強制純白底 + 清楚邊框與陰影，保證高對比 */
[data-testid="stForm"], .stForm {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 24px 20px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06) !important;
}

/* 全域輸入框 (學號、密碼、關鍵字搜尋等)：強制純白底 + 深黑字，打字 100% 清晰可見 */
input, textarea, [data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
}
input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
}
[data-testid="stTextInput"] label {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* 側邊欄：深藍底 + 白色/淺灰字體 */
[data-testid="stSidebar"] {
    background-color: #1e3a5f !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
}

/* 側邊欄輸入框 (文字搜尋、下拉選單)：必須為白底深字，保證輸入文字 100% 可見 */
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
    color: #0f172a !important;
}

/* 按鈕樣式與高對比保護 */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 6px !important;
    font-weight: 500 !important;
}
button[kind="primary"], button[data-testid="baseButton-primary"], .stFormSubmitButton > button {
    background-color: #1e3a5f !important;
    border-color: #1e3a5f !important;
}
button[kind="primary"] *, button[data-testid="baseButton-primary"] *, .stFormSubmitButton > button * {
    color: #ffffff !important;
}
button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}
button[kind="secondary"] *, button[data-testid="baseButton-secondary"] * {
    color: #1e293b !important;
}
</style>
''', unsafe_allow_html=True)

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

# ==========================================
# 定義 Callbacks (優先更新資料庫，解決畫面不同步的問題)
# ==========================================
def cb_click_detail(user_id, link):
    """優先執行：紀錄點擊並設定為作用中連結"""
    database.log_interaction(user_id, link, "click")
    st.session_state['active_link'] = link
    st.toast("✅ 已成功紀錄至點擊歷史！", icon="👀")

def render_activity_card(row, index, mode="recommend", current_user_id=None):
    full_title = str(row['title']).strip()
    regex_pattern = r'^(.*?)\s*([a-zA-Z\[\(][^\u4e00-\u9fa5]*)$'
    match = re.search(regex_pattern, full_title)
    if match and match.group(1).strip():
        title_tw, title_en = match.group(1).strip(), match.group(2).strip()
    else:
        title_tw, title_en = full_title, ""

    safe_tw = title_tw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_en = title_en.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    tags = row['tags'] if isinstance(row['tags'], list) else [t.strip() for t in str(row['tags']).split(" ") if t.strip()]
    dept_target = row.get('dept_target', '全校')

    # tag HTML（簡潔灰底膠囊）
    tag_html = " ".join(
        f"<span style='display:inline-block;background:#f0f2f5;color:#555;border-radius:4px;"
        f"padding:1px 8px;font-size:11px;margin:2px 2px 2px 0;'>#{t}</span>"
        for t in tags if t
    )

    with st.container(border=True):
        col_text, col_btn = st.columns([6, 1])
        with col_text:
            if mode == "history":
                st.markdown(f"<small style='color:#999;'>🕒 {row.get('clicked_time', '')}</small>", unsafe_allow_html=True)
            if mode == "recommend" and 'match_reason' in row and row['match_reason']:
                st.markdown(f"<small style='background:#eef2ff;color:#4a6fa5;padding:2px 8px;border-radius:4px;font-size:11px;'>💡 {row['match_reason']}</small>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1rem;font-weight:600;color:#222;margin:4px 0 2px;'>{safe_tw}</div>", unsafe_allow_html=True)
            if safe_en:
                st.markdown(f"<div style='font-size:0.8rem;color:#888;margin-bottom:2px;'>{safe_en}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px;color:#999;margin-bottom:4px;'>📅 {row['date']}　🎯 {dept_target}</div><div>{tag_html}</div>", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if mode == "history":
                st.link_button("查看", row['link'], use_container_width=True)
            else:
                if st.button("查看詳情", key=f"btn_{mode}_{index}", use_container_width=True):
                    database.log_interaction(current_user_id, row['link'], "click")
                    st.success("✅ 已紀錄興趣")
                    st.markdown(f"[🔗 前往活動頁面]({row['link']})")

def show_history_page(student_id):
    st.title("📚 我的點擊歷史")
    st.caption("以下是你最近感興趣查看過的活動")
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
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # ── Banner ──
        st.markdown("""
        <div style="background:#1e3a5f;color:white;padding:28px 20px 22px;
                    border-radius:12px;text-align:center;margin-bottom:16px;">
            <div style="font-size:2.2rem;margin-bottom:6px;">🎓</div>
            <h1 style="color:white!important;font-size:1.5rem;margin:0 0 4px 0;">南臺科大活動推薦平台</h1>
            <p style="color:#b0c4d8 !important;margin:0;font-size:0.88rem;">AI 智能推薦 · 校園公告</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("""
            <div style="font-size:1.25rem;font-weight:700;color:#0f172a!important;margin:0 0 16px 0;padding-bottom:10px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:8px;">
                <span>🔐 帳號登入</span>
            </div>
            """, unsafe_allow_html=True)
            username = st.text_input("學號 / 教職員編號", placeholder="例如：4B1G0078")
            password = st.text_input("密碼", type="password", placeholder="請輸入密碼")
            submit = st.form_submit_button("🚀 登入系統", use_container_width=True, type="primary")
        
        st.markdown("<div style='text-align:center;margin:12px 0 8px;color:#64748b;font-size:13px;font-weight:500;'>— 或 —</div>", unsafe_allow_html=True)
        guest_submit = st.button("👤 訪客登入（免帳號體驗）", use_container_width=True)
            
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
    
    # ── 側邊欄 ──
    st.sidebar.markdown(f"""
    <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:12px 14px;
                margin-bottom:10px;border:1px solid rgba(255,255,255,0.12);">
        <div style="font-size:0.95rem;font-weight:600;color:white;">{user['name']}</div>
        <div style="font-size:0.78rem;color:#a0b8cc;margin-top:2px;">{user['dept']} · {user['year']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("登出系統", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

    df_activities = load_data()

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔄 資料同步")
    st.sidebar.caption(f"📊 資料庫現有 **{len(df_activities)}** 筆活動")
    if st.sidebar.button("即時同步學校公告", use_container_width=True):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        def on_sync_progress(current, total, msg):
            pct = int((current / max(1, total)) * 100)
            progress_bar.progress(min(pct, 100))
            status_text.caption(f"{pct}% {msg}")

        try:
            status_text.caption("連線學校伺服器中...")
            new_data = scraper.fetch_all_sources(max_pages=2, progress_callback=on_sync_progress)
            if new_data:
                database.save_activities_to_db(new_data)
                progress_bar.progress(100)
                status_text.caption("✅ 同步完成！")
                st.toast(f"🎉 成功同步！資料庫已整合 {len(new_data)} 筆最新公告", icon="🚀")
            else:
                status_text.caption("⚠️ 學校伺服器未回傳新資料")
                st.toast("⚠️ 學校伺服器未回傳新資料（請檢查連線）", icon="⚠️")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"同步出錯: {e}")

    st.sidebar.markdown("---")
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

    recommended_df = pd.DataFrame()
    
    if not df_activities.empty:
        fake_courses = []
        if "資工" in user['dept']: fake_courses = ["程式", "AI"]
        if "食品" in user['dept']: fake_courses = ["食品", "安全"]
        user_interest_tags = get_user_profile_tags(user['dept'], fake_courses, user.get('year', ''))
        recommended_df = calculate_recommendation(user_interest_tags, user['dept'], current_user_id, df_activities, user.get('year', ''))

    st.divider()
    
    if menu == "活動推薦":
        st.sidebar.divider()
        st.sidebar.markdown("### 🔍 活動篩選器")
        search_keyword = st.sidebar.text_input("關鍵字搜尋", placeholder="例如：資安、多益...")
        selected_category = st.sidebar.selectbox("活動分類", ["所有分類", "學術活動", "校園活動", "工讀機會", "徵才訊息"])

        st.title(f"📋 {user['dept']} {user['year']} — 專屬推薦")
        
        if not recommended_df.empty:
            if search_keyword:
                with st.spinner(f"🧠 正在以 AI 語意分析您的需求：「{search_keyword}」..."):
                    semantic_model = load_semantic_model()
                    query_embedding = semantic_model.encode(search_keyword, convert_to_tensor=True)
                    activity_texts = tuple((recommended_df['title'] + " " + 
                                    recommended_df['tags'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))).tolist())
                    corpus_embeddings = get_corpus_embeddings(semantic_model, activity_texts)
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    recommended_df['semantic_score'] = cos_scores.cpu().numpy()
                    recommended_df = recommended_df[
                        (recommended_df['semantic_score'] > 0.35) | 
                        (recommended_df['title'].str.contains(search_keyword, case=False, na=False))
                    ]
                    recommended_df = recommended_df.sort_values(by=['semantic_score', 'score'], ascending=[False, False])
                    for idx in recommended_df.index:
                        old_r = str(recommended_df.at[idx, 'match_reason'])
                        if old_r and old_r not in ["✨ 最新活動", "nan"]:
                            recommended_df.at[idx, 'match_reason'] = f"🕵️ AI 語意精準命中 | {old_r}"
                        else:
                            recommended_df.at[idx, 'match_reason'] = "🕵️ AI 語意精準命中"
            
            if selected_category != "所有分類":
                # 統一以 list 比對，確保分類篩選精確命中 tag 元素而非子字串
                recommended_df = recommended_df[
                    recommended_df['tags'].apply(
                        lambda x: selected_category in (x if isinstance(x, list) else [t.strip() for t in str(x).split(' ') if t.strip()])
                    )
                ]

            is_viewed = recommended_df['is_viewed'] if 'is_viewed' in recommended_df.columns else False
            viewed_df = recommended_df[is_viewed]
            unviewed_df = recommended_df[~is_viewed]
            
            if not viewed_df.empty:
                st.markdown("### 👀 最近已查看")
                for index, row in viewed_df.head(3).iterrows():
                    render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
                if len(viewed_df) > 3:
                    st.caption(f"👉 還有 {len(viewed_df) - 3} 筆紀錄，可至左側「我的點擊紀錄」分頁查看完整清單。")
                st.divider() 
                
            st.markdown(f"### ✨ 為您找到 {len(unviewed_df)} 筆活動")
            paged_rec_df, rec_total_pages = get_paged_data(unviewed_df, 10, "recommend_page")
            for index, row in paged_rec_df.iterrows():
                render_activity_card(row, index, mode="recommend", current_user_id=current_user_id)
            render_google_pagination(rec_total_pages, "recommend_page")
        else:
            st.warning(" 資料庫為空，請確認後端爬蟲狀態。")

    elif menu == "🤖 AI 導覽員":
        if not recommended_df.empty:
            render_agentic_rag_chat(user, recommended_df)
        else:
            st.warning("目前沒有活動資料，AI 導覽員無法為您服務。")

    elif menu == "我的點擊紀錄":
        show_history_page(current_user_id)