import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def extract_features(user_dept, user_year, activity_dept, activity_tags_str, user_tags_list):
    """特徵工程將文字資料轉換為機器學習模型看得懂的數字"""
    # 特徵 1: 系所是否吻合 (1為是 0為否)
    dept_match = 1 if (user_dept in activity_dept or activity_dept == "全校") else 0
    
    # 特徵 2: 標籤重疊數量
    activity_tags = activity_tags_str.split(" ") if isinstance(activity_tags_str, str) else []
    overlap_count = len(set(user_tags_list).intersection(set(activity_tags)))
    
    # 特徵 3: 是否為高年級實習專屬
    is_senior = 1 if any(y in user_year for y in ["三", "四", "碩"]) else 0
    is_internship = 1 if any(kw in activity_tags_str for kw in ["實習", "徵才", "就業"]) else 0
    senior_internship_match = is_senior * is_internship
    
    return [dept_match, overlap_count, senior_internship_match]

def train_model():
    """從資料庫萃取資料並訓練隨機森林預測模型"""
    print("開始讀取資料庫...")
    conn = sqlite3.connect("stust_recommendation.db")
    
    df_users = pd.read_sql_query("SELECT * FROM users", conn)
    df_acts = pd.read_sql_query("SELECT * FROM activities", conn)
    df_interactions = pd.read_sql_query("SELECT * FROM user_interactions", conn)
    conn.close()

    if df_interactions.empty:
        print("警告目前沒有任何點擊紀錄請先去網站上隨便點擊幾個活動產生數據")
        return

    print("建構訓練資料集...")
    X = []
    y = []

    # 為了模擬真實情況我們需要正樣本與負樣本
    clicked_pairs = set(zip(df_interactions['student_id'], df_interactions['activity_link']))

    for _, user in df_users.iterrows():
        student_id = user['student_id']
        user_dept = user['dept']
        user_year = user['year']
        
        # 簡單模擬使用者的興趣標籤
        user_tags = ["資安", "AI"] if "資工" in user_dept else ["一般"]

        for _, act in df_acts.iterrows():
            act_link = act['link']
            act_dept = act['dept_target']
            act_tags = act['tags']

            # 提取數字特徵
            features = extract_features(user_dept, user_year, act_dept, act_tags, user_tags)
            X.append(features)
            
            # 若有點擊紀錄標記為 1 否則標記為 0
            if (student_id, act_link) in clicked_pairs:
                y.append(1)
            else:
                y.append(0)

    # 將特徵轉換為陣列
    X = np.array(X)
    y = np.array(y)

    # 切割訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("開始訓練隨機森林分類器...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 評估模型準確率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型訓練完成準確率為 {accuracy * 100:.2f}%")

    # 將訓練好的模型存檔
    joblib.dump(model, "ctr_model.pkl")
    print("模型已儲存為 ctr_model.pkl")

if __name__ == "__main__":
    train_model()