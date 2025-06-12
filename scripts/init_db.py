# scripts/init_db.py
import sqlite3
import os
import logging
logger = logging.getLogger(f"cinfer.{__name__}")

# Determine the path to the database from configuration or a default
# For simplicity, using a fixed path here. In a real app, get this from ConfigManager.
DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'db')
DB_PATH = os.path.join(DB_DIR, 'cinfer.db') # As per 8.3 data/db/

TABLE_DEFINITIONS = {
    "users": """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        status TEXT DEFAULT 'active'
    );
    """, # From 5.3.1
    "models": """
    CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        remark TEXT,
        engine_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        params_path TEXT,
        input_schema TEXT,
        output_schema TEXT,
        created_by TEXT DEFAULT 'system',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'draft',
        FOREIGN KEY (created_by) REFERENCES users (id) ON DELETE SET NULL
    );
    """, # From 5.3.2
    "auth_tokens": """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        token_value_hash TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """, # From 5.3.3
    "access_tokens": """
    CREATE TABLE IF NOT EXISTS access_tokens (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        token_value_hash TEXT UNIQUE NOT NULL,
        token_value_view TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'active', 
        allowed_models TEXT NOT NULL DEFAULT '[]',
        ip_whitelist TEXT NOT NULL DEFAULT '[]',
        rate_limit INTEGER DEFAULT 100, 
        monthly_limit INTEGER,
        used_count INTEGER DEFAULT 0,
        remark TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """, # From 5.3.3
    "inference_logs": """
    CREATE TABLE IF NOT EXISTS inference_logs (
        id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        access_token_id TEXT,
        request_id TEXT,
        client_ip TEXT,
        request_data TEXT,
        response_data TEXT,
        status TEXT,
        error_message TEXT,
        latency_ms REAL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
        FOREIGN KEY (access_token_id) REFERENCES access_tokens (id) ON DELETE SET NULL
    );
    """ # From 5.3.4
}

INDEX_DEFINITIONS = [


    # models 表新增的索引
    "CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);", # 如果经常按名称搜索模型
    "CREATE INDEX IF NOT EXISTS idx_models_created_by ON models(created_by);", # 如果经常按创建者搜索模型

    # auth_tokens 表的索引
    # token_value_hash 是 UNIQUE，会自动创建索引
    "CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id ON auth_tokens(user_id);", # 频繁按用户ID查找会话令牌
    "CREATE INDEX IF NOT EXISTS idx_auth_tokens_expires_at ON auth_tokens(expires_at);", # 清理过期令牌或检查时有用

    # access_tokens 表的索引
    # token_value_hash 是 UNIQUE，会自动创建索引
    "CREATE INDEX IF NOT EXISTS idx_access_tokens_user_id ON access_tokens(user_id);", # 按用户ID查找API密钥
    "CREATE INDEX IF NOT EXISTS idx_access_tokens_name ON access_tokens(name);", # 如果API密钥有可搜索的名称


    # inference_logs 表
    "CREATE INDEX IF NOT EXISTS idx_inference_logs_model_id ON inference_logs(model_id);",
    "CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at ON inference_logs(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_inference_logs_status ON inference_logs(status);",
]

def initialize_database():
    """Creates all tables and indexes in the SQLite database."""
    # Ensure the database directory exists
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        logger.info(f"Created directory: {DB_DIR}")

    conn = None
    try:
        logger.info(f"Initializing database at: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Enable Foreign Keys
        cursor.execute("PRAGMA foreign_keys = ON;")

        logger.info("Creating tables...")
        for table_name, definition in TABLE_DEFINITIONS.items():
            logger.info(f"  Creating table {table_name}...")
            cursor.execute(definition)

        logger.info("Creating indexes...")
        for index_def in INDEX_DEFINITIONS:
            logger.info(f"  Executing: {index_def.split('ON')[0].strip()}...") # Short print
            cursor.execute(index_def)

        conn.commit()
        logger.info("Database initialization completed successfully.")
    except sqlite3.Error as e:
        logger.error(f"An error occurred during database initialization: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    initialize_database()