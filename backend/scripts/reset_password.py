#!/usr/bin/env python3
# scripts/reset_password.py

import os
import sys
import argparse
import sqlite3
import logging
from datetime import datetime


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.security import get_password_hash

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("password_reset")

def get_db_connection():
    """获取数据库连接"""
    db_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'db')
    db_path = os.path.join(db_dir, 'cinfer.db')
    
    if not os.path.exists(db_path):
        logger.error(f"数据库文件不存在: {db_path}")
        sys.exit(1)
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"连接数据库时出错: {e}")
        sys.exit(1)

def list_users(conn):
    """列出所有用户"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, is_admin, status, created_at FROM users")
        users = cursor.fetchall()
        
        if not users:
            print("系统中没有用户")
            return
            
        print("\n用户列表:")
        print("-" * 80)
        print(f"{'ID':<36} | {'用户名':<15} | {'管理员':<6} | {'状态':<8} | {'创建时间'}")
        print("-" * 80)
        
        for user in users:
            print(f"{user['id']:<36} | {user['username']:<15} | {'是' if user['is_admin'] else '否':<6} | {user['status']:<8} | {user['created_at']}")
            
        print("-" * 80)
    except sqlite3.Error as e:
        logger.error(f"获取用户列表时出错: {e}")
        sys.exit(1)

def reset_password(conn, username, new_password):
    """重置指定用户的密码"""
    try:
        # 检查用户是否存在
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if not user:
            logger.error(f"用户 '{username}' 不存在")
            return False
            
        # 生成新的密码哈希
        password_hash = get_password_hash(new_password)
        
        # 更新密码
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username)
        )
        conn.commit()
        
        logger.info(f"已成功重置用户 '{username}' 的密码")
        return True
    except sqlite3.Error as e:
        logger.error(f"重置密码时出错: {e}")
        conn.rollback()
        return False

def main():
    parser = argparse.ArgumentParser(description="用户密码重置工具")
    parser.add_argument("--list", action="store_true", help="列出所有用户")
    parser.add_argument("--username", type=str, help="要重置密码的用户名")
    parser.add_argument("--password", type=str, help="新密码")
    
    args = parser.parse_args()
    
    # 获取数据库连接
    conn = get_db_connection()
    
    try:
        if args.list:
            list_users(conn)
        elif args.username and args.password:
            if reset_password(conn, args.username, args.password):
                print(f"密码重置成功: 用户 '{args.username}'")
            else:
                print(f"密码重置失败: 用户 '{args.username}'")
        else:
            parser.print_help()
    finally:
        conn.close()

if __name__ == "__main__":
    main() 