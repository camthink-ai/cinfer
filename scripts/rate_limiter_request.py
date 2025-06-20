import requests
import time

def get_user_input():
    """
    通过交互式提示获取用户配置，并进行验证。
    """
    print("--- API 请求配置 ---")
    
    # 1. 获取URL (必填)
    while True:
        url = input("👉 请输入目标API的完整URL: ")
        if url.strip():  # .strip()确保用户输入的不是纯空格
            break
        print("❌ URL不能为空，请重新输入。")
        
    # 2. 获取Token (必填)
    while True:
        token = input("👉 请输入x-access-token: ")
        if token.strip():
            break
        print("❌ Token不能为空，请重新输入。")

    # 3. 获取频率 (可选, 有默认值)
    while True:
        freq_str = input("👉 请输入请求频率 (每秒次数, 默认: 1.0): ")
        if not freq_str:
            frequency = 1.0
            break
        try:
            frequency = float(freq_str)
            if frequency > 0:
                break
            else:
                print("❌ 频率必须是大于零的数字。")
        except ValueError:
            print("❌ 无效输入，请输入一个数字 (例如: 2 或 0.5)。")

    # 4. 获取总次数 (可选, 有默认值)
    while True:
        count_str = input("👉 请输入总请求次数 (默认: 10): ")
        if not count_str:
            count = 10
            break
        try:
            count = int(count_str)
            if count > 0:
                break
            else:
                print("❌ 请求次数必须是大于零的整数。")
        except ValueError:
            print("❌ 无效输入，请输入一个整数。")
            
    return url, token, frequency, count

def send_request(session: requests.Session, url: str):
    """
    使用提供的会话对象和URL发送单次API请求并打印结果。
    """
    try:
        start_time = time.time()
        response = session.get(url)
        response.raise_for_status()
        duration = time.time() - start_time
        print(f"✅ Success | Status: {response.status_code} | Duration: {duration:.3f}s")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed  | Error: {e}")

def main():
    """
    主函数，获取用户输入并控制请求循环。
    """
    # 从交互式提示中获取配置
    url, token, frequency, count = get_user_input()

    delay = 1.0 / frequency
    
    headers = {
        'accept': 'application/json',
        'x-access-token': token
    }
    
    with requests.Session() as session:
        session.headers.update(headers)
        
        print("\n" + "-" * 50)
        print(f"🚀 配置确认，准备发送请求...")
        print(f"   - 目标URL: {url}")
        print(f"   - 请求频率: {frequency} req/s")
        print(f"   - 请求间隔: {delay:.3f} s")
        print(f"   - 总请求数: {count}")
        print("   - (按 Ctrl+C 随时停止)")
        print("-" * 50)

        try:
            for i in range(count):
                print(f"[{i + 1:>{len(str(count))}}/{count}] ", end="")
                
                loop_start_time = time.time()
                send_request(session, url)
                
                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0, delay - elapsed_time)
                
                if i < count - 1:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n🛑 用户手动中断。")
        finally:
            print("-" * 50)
            print("✅ 脚本执行完毕。")

if __name__ == "__main__":
    main()