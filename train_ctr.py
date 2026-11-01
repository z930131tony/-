import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

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
    user_dept = str(user_dept or '')
    
    if user_dept in act_dept or act_dept in user_dept:
        dept_score = 2.0
    elif act_dept == '全校':
        dept_score = 1.0
    else:
        dept_score = 0.0
        
    tags_set = set(act_tags_list) if isinstance(act_tags_list, (list, set)) else set(str(act_tags_list or '').split())
    overlap = float(len(set(user_tags).intersection(tags_set)))
    
    is_senior = 1.0 if any(y in str(user_year or '') for y in ['三', '四', '碩']) else 0.0
    search_txt = str(act_tags_list or '') + ' ' + str(act_title or '')
    is_career = 1.0 if any(kw in search_txt for kw in ['實習', '徵才', '就業', '職缺', '工讀']) else 0.0
    senior_career = is_senior * is_career
    
    return [dept_score, overlap, is_senior, is_career, senior_career]

def train_model():
    """從資料庫萃取資料並訓練平衡化隨機森林點擊率 (CTR) 預測模型"""
    print("🚀 開始讀取資料庫...")
    conn = sqlite3.connect("stust_recommendation.db")
    
    df_users = pd.read_sql_query("SELECT * FROM users", conn)
    df_acts = pd.read_sql_query("SELECT * FROM activities", conn)
    df_interactions = pd.read_sql_query("SELECT * FROM user_interactions", conn)
    conn.close()

    if df_interactions.empty:
        print("⚠️ 警告：目前沒有任何點擊紀錄，請先至網站瀏覽點擊活動產生數據。")
        return

    print("📊 建構特徵訓練矩陣與標籤...")
    from app import get_user_profile_tags
    
    clicked_pairs = set(zip(df_interactions['student_id'], df_interactions['activity_link']))
    X = []
    y = []

    for _, user in df_users.iterrows():
        u_id = user['student_id']
        u_dept = user['dept']
        u_year = user['year']
        
        # 依系所配置合理之探索課程以獲取全貌標籤
        fake_courses = ['程式', 'AI'] if '資工' in u_dept else (['食品', '安全'] if '食品' in u_dept else ['設計'])
        u_tags = get_user_profile_tags(u_dept, fake_courses, u_year)

        for _, act in df_acts.iterrows():
            act_link = act['link']
            act_dept = act['dept_target']
            act_tags = act['tags'].split() if isinstance(act['tags'], str) else []
            act_title = act['title']

            features = extract_features(u_dept, u_year, act_dept, act_tags, act_title, u_tags)
            X.append(features)
            y.append(1 if (u_id, act_link) in clicked_pairs else 0)

    X = np.array(X)
    y = np.array(y)

    pos_count = int(np.sum(y))
    print(f"📈 樣本總數：{len(y)} 筆，其中正樣本（點擊）：{pos_count} 筆")

    # 負樣本下採樣 (Negative Downsampling) 解決 99.7% 極度不平衡問題
    np.random.seed(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if pos_count > 0:
        sample_size = min(len(neg_idx), max(pos_count * 5, 25))
        sampled_neg_idx = np.random.choice(neg_idx, size=sample_size, replace=False)
        train_idx = np.concatenate([pos_idx, sampled_neg_idx])
    else:
        train_idx = np.arange(len(y))

    X_train_sub = X[train_idx]
    y_train_sub = y[train_idx]

    print("🌲 開始訓練隨機森林 CTR 預測器...")
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train_sub, y_train_sub)

    feature_names = ["系所吻合度", "標籤重疊數", "高年級指標", "就業實習指標", "高年級實習交叉項"]
    print("\n🔍 各特徵重要度 (Feature Importances)：")
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"  - {name}: {imp * 100:.2f}%")

    # 驗證測試樣本的機率輸出
    high_match = extract_features("資工系", "大四", "資工系", ["資工", "實習"], "資工系企業實習說明會", ["資工", "實習", "AI"])
    low_match = extract_features("資工系", "大四", "視傳系", ["視傳", "藝術"], "視傳畢業設計展", ["資工", "實習", "AI"])
    
    p_high = model.predict_proba([high_match])[0]
    p_low = model.predict_proba([low_match])[0]
    c_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    
    print(f"\n🎯 高度符合活動預測點擊率：{p_high[c_idx] * 100:.1f}%")
    print(f"🎯 無關活動預測點擊率：{p_low[c_idx] * 100:.1f}%")

    joblib.dump(model, "ctr_model.pkl")
    print("\n💾 模型已成功儲存為 ctr_model.pkl")

if __name__ == "__main__":
    train_model()
