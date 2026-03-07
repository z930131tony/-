import sqlite3
import pandas as pd
import bcrypt

DB_NAME = "stust_recommendation.db"

def get_connection():
    """建立並回傳資料庫連線設定防止多執行緒存取衝突"""
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    """初始化資料庫與建立所需之資料表結構"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 建立使用者資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            student_id TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            dept TEXT,
            year TEXT,
            name TEXT
        )
    ''')
    
    # 建立活動資料表並設定連結欄位為唯一值避免重複寫入
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

    # 建立使用者互動歷史紀錄表
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
    """執行密碼雜湊加密運算"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    """驗證輸入密碼與資料庫中雜湊值是否吻合"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except ValueError:
        # 處理資料庫殘留明文密碼所導致的例外狀況
        return False

def seed_mock_users():
    """於資料庫為空時寫入系統預設之測試帳號資料"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 檢查使用者資料表是否已有紀錄
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        # 將測試帳號的密碼全數經過雜湊加密處理後再行寫入
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
    """執行登入驗證程序並回傳符合之使用者資訊"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 提取特定學號之使用者紀錄以進行後續密碼比對
    cursor.execute('''
        SELECT student_id, password, dept, year, name 
        FROM users WHERE student_id = ?
    ''', (student_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    # 比對輸入明文與資料庫儲存之雜湊密碼
    if row and check_password(password, row[1]):
        return {"student_id": row[0], "dept": row[2], "year": row[3], "name": row[4]}
    return None

def verify_local_user(username, password):
    """執行離線模式之本地端登入驗證"""
    return verify_user(username, password)

def sync_user_to_db(username, password, info):
    """將登入成功之使用者資訊與加密密碼同步儲存至本地資料庫"""
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
    """將爬蟲取得之活動資料批次寫入資料庫"""
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
    """提取資料庫內所有活動紀錄並轉換為資料框架格式供系統存取"""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM activities", conn)
    conn.close()
    return df

def log_interaction(student_id, activity_link, interaction_type="click"):
    """將使用者與系統互動之行為紀錄存入資料庫"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_interactions (student_id, activity_link, interaction_type)
        VALUES (?, ?, ?)
    ''', (student_id, activity_link, interaction_type))
    conn.commit()
    conn.close()

def get_user_clicked_activities(student_id):
    """查詢特定使用者曾點擊瀏覽過之活動連結清單"""
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