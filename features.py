"""推薦系統共用的特徵工程。

app.py (線上推薦) 與 train_ctr.py (離線訓練) 都從這裡匯入，確保訓練與推論用的是
同一套標籤與特徵，避免 training/serving skew；train_ctr.py 也因此不必 import app
(import app 會把整個 Streamlit 頁面與語意模型都跑一次)。
"""
import re

TAG_RULES = {
    "資安": ["資安", "漏洞", "駭客", "密碼", "防護"],
    "AI": ["AI", "人工智慧", "機器學習"],
    "徵才": ["徵才", "招募", "誠徵", "專任助理", "職缺", "面試"],
    "實習": ["實習", "intern", "internship"],
    "研討會": ["研討會", "論壇", "講座", "說明會"],
    "半導體": ["半導體", "製程", "晶片", "電性"],
    "競賽": ["競賽", "比賽", "黑客松", "挑戰賽"]
}

COURSE_TAG_MAP = {
    ("程式", "python", "java", "軟體", "網頁"): "資工",
    ("ai", "人工智慧", "機器學習", "深度學習"): "AI",
    ("安全", "密碼", "駭客", "防護"): "資安",
    ("行銷", "企管", "管理", "商業"): "企管",
    ("設計", "視覺", "動畫", "藝術"): "設計",
    ("電機", "電路", "半導體", "晶片"): "半導體",
    ("食品", "烘焙", "營養", "檢驗"): "食品",
}

DEPT_TAG_MAP = {
    "資工": ["資安", "漏洞", "AI", "資訊", "軟體", "網路", "系統"],
    "食品": ["食品", "衛生", "農業", "檢驗", "化學"],
    "企管": ["商業", "行銷", "管理", "創新", "產業"],
    "設計": ["文化", "設計", "藝術", "書展", "展覽"],
    "電機": ["半導體", "製程", "電性", "機電", "工程"],
    "應英": ["雙語", "EMI", "外語", "英文", "國際"]
}

SENIOR_MARKERS = ["三", "四", "碩"]
CAREER_KEYWORDS = ['實習', '徵才', '就業', '職缺', '工讀']


def keyword_in(keyword, text):
    """關鍵字比對：英文關鍵字需整字命中 (不分大小寫)，中文維持子字串比對。
    避免 "intern" 命中 "International"、"AI" 命中 "Fair" / "Email" / "Taiwan"。"""
    text = str(text or '')
    if keyword.isascii():
        return re.search(rf'(?<![A-Za-z]){re.escape(keyword)}(?![A-Za-z])', text, re.IGNORECASE) is not None
    return keyword in text


def is_senior(user_year):
    return any(y in str(user_year or '') for y in SENIOR_MARKERS)


def get_fake_courses(department):
    """依系所配置模擬修課紀錄 (訓練與推論共用同一份，避免特徵不一致)"""
    department = str(department or '')
    if "資工" in department: return ["程式", "AI"]
    if "食品" in department: return ["食品", "安全"]
    return []


def enrich_activity_tags(df):
    def extract_tags(title, existing_tags):
        tags = set(existing_tags) if isinstance(existing_tags, (list, set)) else set(str(existing_tags or '').split())
        if "一般" in tags and len(tags) == 1: tags.remove("一般")
        for tag, keywords in TAG_RULES.items():
            if any(keyword_in(kw, title) for kw in keywords): tags.add(tag)
        if not tags: tags.add("校園公告")
        return sorted(tags)

    df['tags'] = df.apply(lambda row: extract_tags(row['title'], row['tags']), axis=1)
    return df


def get_user_profile_tags(department, past_courses, user_year=""):
    department = str(department or '')
    user_year = str(user_year or '')
    user_tags = set()
    for key, tags in DEPT_TAG_MAP.items():
        if key in department: user_tags.update(tags)

    for course in past_courses or []:
        for keywords, tag in COURSE_TAG_MAP.items():
            if any(keyword_in(kw, course) for kw in keywords): user_tags.add(tag)

    if is_senior(user_year):
        user_tags.update(["實習", "徵才", "招募", "專任助理", "就業", "企業", "職缺"])
    elif any(y in user_year for y in ["一", "二"]):
        user_tags.update(["社團", "營隊", "志工", "迎新", "校園活動", "講座", "工讀"])

    user_tags.update(["研討會", "獎學金", "計畫", "講座"])
    return user_tags


def dept_matches(user_dept, act_dept):
    """使用者系所與活動對象系所是否吻合 (空字串或「全校」不算吻合)"""
    user_dept = str(user_dept or '').strip()
    act_dept = str(act_dept or '').strip()
    if not user_dept or not act_dept or '全校' in (user_dept, act_dept):
        return False
    return user_dept in act_dept or act_dept in user_dept


def extract_features(user_dept, user_year, act_dept, act_tags_list, act_title, user_tags):
    """
    特徵工程模組：將文字、系所與標籤轉換為機器學習特徵向量
    回傳特徵維度 (長度 5)：
    1. dept_score: 系所吻合度 (2.0: 直接吻合, 1.0: 全校, 0.0: 他系)
    2. tag_overlap: 興趣標籤與活動標籤重疊數
    3. is_senior: 是否為大三、大四或碩士高年級
    4. is_career: 是否為實習/徵才/職缺/工讀
    5. senior_career: 高年級與就業活動的交叉特徵 (互動項)
    """
    act_dept = str(act_dept or '全校')

    if act_dept == '全校':
        dept_score = 1.0
    elif dept_matches(user_dept, act_dept):
        dept_score = 2.0
    else:
        dept_score = 0.0

    tags_set = set(act_tags_list) if isinstance(act_tags_list, (list, set)) else set(str(act_tags_list or '').split())
    overlap = float(len(set(user_tags).intersection(tags_set)))

    senior = 1.0 if is_senior(user_year) else 0.0
    search_txt = ' '.join(tags_set) + ' ' + str(act_title or '')
    is_career = 1.0 if any(kw in search_txt for kw in CAREER_KEYWORDS) else 0.0
    senior_career = senior * is_career

    return [dept_score, overlap, senior, is_career, senior_career]


def click_probability(model, features_matrix):
    """回傳「點擊 (類別 1)」的機率；模型若沒學過正樣本就回傳全 0，
    而不是誤把「不點擊」的機率當成點擊率。"""
    import numpy as np
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return np.zeros(len(features_matrix))
    return model.predict_proba(features_matrix)[:, classes.index(1)]
