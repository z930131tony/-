import sqlite3
import pandas as pd
import bcrypt

DB_NAME = "stust_recommendation.db"

def get_connection():
    """建立並回傳資料庫連線"""
    # 加入 check_same_thread=False 防止 Streamlit 在多執行緒下報錯
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    """初始化資料庫與資料表"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 建立使用者表 (Users)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            student_id TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            dept TEXT,
            year TEXT,
            name TEXT
        )
    ''')
    
    # 建立活動表 (Activities)
    # 使用 link 作為 UNIQUE，避免爬蟲重複寫入相同的活動
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            date TEXT,
            link TEXT UNIQUE NOT NULL,
            tags TEXT,
            dept_target TEXT,
            img TEXT
        )
    ''')

    # 建立使用者互動紀錄表 (user_interactions)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            activity_link TEXT,
            interaction_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """將密碼加密 (使用 bcrypt)"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    """檢查密碼是否與資料庫中的雜湊值符合"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except ValueError:
        # 防呆：避免資料庫裡殘留未加密的明文密碼導致程式崩潰
        return False

def seed_mock_users():
    """寫入模擬的使用者資料 (如果資料庫是空的)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 檢查是否已經有使用者
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        # 🚨 修復：寫入資料庫前，將密碼全部經過 bcrypt 雜湊處理
        mock_users = [
            ("4B1G0000", hash_password("123"), "食品系", "大三", "王老吉"),
            ("4B1G0078", hash_password("123"), "資工系", "大四", "謝宇翔"),
            ("4B1G0177", hash_password("123"), "設計系", "大一", "范景程"),
            ("admin", hash_password("admin"), "全校", "教職員", "管理員")
        ]
        cursor.executemany('''
            INSERT INTO users (student_id, password, dept, year, name) 
            VALUES (?, ?, ?, ?, ?)
        ''', mock_users)
        conn.commit()
    conn.close()

def verify_user(student_id, password):
    """(統一版) 驗證登入並回傳使用者資訊"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 🚨 修復：不能直接用 AND password = ? 查，必須先撈出雜湊密碼再比對
    cursor.execute('''
        SELECT student_id, password, dept, year, name 
        FROM users WHERE student_id = ?
    ''', (student_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    # row[1] 是資料庫裡的雜湊密碼
    if row and check_password(password, row[1]):
        return {"student_id": row[0], "dept": row[2], "year": row[3], "name": row[4]}
    return None

def verify_local_user(username, password):
    """本地備援驗證：當學校斷網時使用"""
    # 因為邏輯一樣，直接呼叫修復後的 verify_user 即可
    return verify_user(username, password)

def sync_user_to_db(username, password, info):
    """登入成功後，將最新資訊與加密後的密碼同步回本地 SQLite"""
    conn = get_connection()
    cursor = conn.cursor()
    hashed_pwd = hash_password(password)
    
    cursor.execute('''
        INSERT OR REPLACE INTO users (student_id, password, dept, year, name)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, hashed_pwd, info.get('dept', ''), info.get('year', ''), info.get('name', '')))
    
    conn.commit()
    conn.close()

def save_activities_to_db(activities_list):
    """將爬蟲抓到的活動寫入資料庫"""
    if not activities_list: return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    for act in activities_list:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO activities (title, date, link, tags, dept_target, img)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (act['title'], act['date'], act['link'], act['tags'], act['dept_target'], act['img']))
        except Exception as e:
            print(f"寫入活動失敗: {e}")
            
    conn.commit()
    conn.close()

def get_all_activities_df():
    """從資料庫讀取所有活動，並轉為 Pandas DataFrame 供推薦系統使用"""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM activities", conn)
    conn.close()
    return df

def log_interaction(student_id, activity_link, interaction_type="click"):
    """紀錄使用者點擊活動的行為"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_interactions (student_id, activity_link, interaction_type)
        VALUES (?, ?, ?)
    ''', (student_id, activity_link, interaction_type))
    conn.commit()
    conn.close()

def get_user_clicked_activities(student_id):
    """獲取使用者點擊過的活動連結清單"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT activity_link 
        FROM user_interactions 
        WHERE student_id = ?
    ''', (student_id,))
    links = [row[0] for row in cursor.fetchall()]
    conn.close()
    return links