#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爬虫程序：获取电站历史数据
作用：自动登录网站并获取指定时间段内的电站数据
"""

import os
import json
import time
import logging
import requests
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("crawler.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("电站数据爬虫")

# 默认配置
DEFAULT_CONFIG = {
    "website": {
        "login_url": "http://hnfbs.cnnchn.com.cn/login",
        "api_base_url": "http://hnfbs.cnnchn.com.cn/station-s/station/statistic/history/day",
        "dashboard_url": "http://hnfbs.cnnchn.com.cn/sub-monitor/plant/plantDetail",
    },

    "weather": {
        "weather_url":      "http://hnfbs.cnnchn.com.cn/dict-s/weather/record/day",
        "regionNationId":   "44",
        "regionLevel1":     "156924",
        "regionLevel2":     "158933",
        "lan":              "zh",
    },

    "user": {"username": "Zhouliguo", "password": "Solar@2025"},
    "crawler": {"default_station_id": "197", "data_dir": "data", "retry_times": "3", "retry_interval": "5"},
    "date_range": {
        "start_year": "2025",
        "start_month": "5",
        "start_day": "13",
        "end_year": "2025",
        "end_month": "6",
        "end_day": "22",
    },

}


class ConfigManager:
    """配置管理类，用于读取和管理配置"""

    def __init__(self, config_file: str = "config.ini"):
        """初始化配置管理器

        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()

        # 如果配置文件不存在，创建默认配置
        if not os.path.exists(config_file):
            self._create_default_config()
        else:
            self.config.read(config_file, encoding="utf-8")
        logger.info("当前配置文件加载完毕，内容如下：")
        for section in self.config.sections():
            logger.info(f"[{section}]")
            for key, val in self.config.items(section):
                logger.info(f"  {key} = {val}")

    def _create_default_config(self):
        """创建默认配置文件"""
        for section, options in DEFAULT_CONFIG.items():
            self.config[section] = options

        with open(self.config_file, "w", encoding="utf-8") as f:
            self.config.write(f)

        logger.info(f"已创建默认配置文件: {self.config_file}")

    def get(self, section: str, option: str, fallback: Any = None) -> str:
        """获取配置值

        Args:
            section: 配置节
            option: 配置项
            fallback: 默认值

        Returns:
            配置值
        """
        return self.config.get(section, option, fallback=fallback)

    def get_int(self, section: str, option: str, fallback: int = 0) -> int:
        """获取整型配置值"""
        return self.config.getint(section, option, fallback=fallback)


class StationCrawler:
    """电站数据爬虫类"""

    def __init__(self, config_manager: ConfigManager):
        """初始化爬虫

        Args:
            config_manager: 配置管理器
        """
        self.config = config_manager
        print(self.config.config.sections())  
        self.driver = None
        self.cookie = None

        # 确保数据目录存在
        self.data_dir = self.config.get("crawler", "data_dir")
        
        os.makedirs(self.data_dir, exist_ok=True)

    def init_browser(self) -> None:
        """初始化浏览器"""
        chrome_options = Options()
        # 可以添加浏览器选项，例如无头模式等
        # chrome_options.add_argument('--headless')

        logger.info("正在初始化浏览器...")
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)

    def login(self) -> bool:
        """登录网站

        Returns:
            登录是否成功
        """
        if self.driver is None:
            self.init_browser()

        login_url = self.config.get("website", "login_url")
        username = self.config.get("user", "username")
        password = self.config.get("user", "password")

        try:
            logger.info(f"正在访问登录页面: {login_url}")
            self.driver.get(login_url)

            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@placeholder='请输入用户名']"))
            )
            # 输入用户名和密码
            logger.info(f"正在输入用户名: {username}")
            self.driver.find_element(By.XPATH, "//input[@placeholder='请输入用户名']").send_keys(username)
            logger.info("正在输入密码")
            self.driver.find_element(By.XPATH, "//input[@placeholder='请输入密码']").send_keys(password)

            # 等待用户手动输入验证码并登录
            logger.info("请在浏览器中输入验证码并点击登录按钮")
            input("请在弹窗中输入验证码登录后按回车键继续...")

            # 获取cookie
            self.cookie = self.driver.get_cookies()
            if self.cookie and len(self.cookie) > 0:
                logger.info("登录成功，已获取Cookie")
                return True
            else:
                logger.error("登录失败，未能获取Cookie")
                return False

        except Exception as e:
            logger.error(f"登录过程中发生错误: {str(e)}")
            return False

    def get_station_data(self, station_id: str, year: int, month: int, day: int) -> Optional[Dict]:
        """获取指定日期的电站数据

        Args:
            station_id: 电站ID
            year: 年份
            month: 月份
            day: 日期

        Returns:
            电站数据，获取失败则返回None
        """
        if not self.cookie:
            logger.error("尚未获取Cookie，请先登录")
            return None

        api_base_url = self.config.get("website", "api_base_url")
        dashboard_url = self.config.get("website", "dashboard_url")

        # 格式化月份和日期，确保是两位数
        month_str = f"{month:02d}"
        day_str = f"{day:02d}"

        url = f"{api_base_url}/{station_id}?year={year}&month={month_str}&day={day_str}"
        logger.debug(f"正在请求电站数据: {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Accept-Encoding": "gzip, deflate",
            "Authorization": f"Bearer {self.cookie[0]['value']}",
            "Connection": "keep-alive",
            "Referer": f"{dashboard_url}/{station_id}/dashboard",
        }

        retry_times = self.config.get_int("crawler", "retry_times", 3)
        retry_interval = self.config.get_int("crawler", "retry_interval", 5)

        # 添加重试机制
        for attempt in range(retry_times):
            try:
                logger.info(
                    f"正在获取电站 {station_id} 在 {year}-{month_str}-{day_str} 的数据 (尝试 {attempt+1}/{retry_times})"
                )
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"请求失败，状态码: {response.status_code}")

            except Exception as e:
                logger.error(f"请求过程中发生错误: {str(e)}")

            # 如果不是最后一次尝试，则等待一段时间后重试
            if attempt < retry_times - 1:
                logger.info(f"将在 {retry_interval} 秒后重试...")
                time.sleep(retry_interval)

        logger.error(f"获取电站 {station_id} 在 {year}-{month_str}-{day_str} 的数据失败，已达到最大重试次数")
        return None

    def get_weather_data(self, date: datetime, station_id: str) -> Optional[Dict]:
        """单天天气接口调用"""
        # 1. 确保 date 是 datetime
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")
        # 2. 读取配置
        base = self.config.get("weather", "weather_url")
        rn   = self.config.get("weather", "regionNationId")
        rl1  = self.config.get("weather", "regionLevel1")
        rl2  = self.config.get("weather", "regionLevel2")
        lan  = self.config.get("weather", "lan")
        # 3. 拼 URL
        y = date.year
        m = f"{date.month:02d}"
        d = f"{date.day:02d}"
        url = (
            f"{base}"
            f"?regionNationId={rn}"
            f"&regionLevel1={rl1}"
            f"&regionLevel2={rl2}"
            f"&year={y}&month={m}&day={d}"
            f"&lan={lan}"
        )
        logger.info(f"Fetching weather: {url}")

        # 4. 构造请求头（去掉 “…” 等非 ASCII 字符）
        headers = {
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept":        "application/json, text/plain, */*",
            "Authorization": f"Bearer {self.cookie[0]['value']}",
            "Referer":       f"{self.config.get('website','dashboard_url')}/{station_id}/dashboard",
        }

        # 5. 请求并返回 JSON
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"Weather API returned {resp.status_code}")
        except Exception as e:
            logger.error(f"Weather request error: {e}")
        return None


    def get_data_for_period(self, station_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """获取指定时间段内的电站数据

        Args:
            station_id: 电站ID
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            时间段内的电站数据列表
        """
        logger.info(
            f"开始获取电站 {station_id} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据"
        )

        current_date = start_date
        success_count = 0
        fail_count = 0

        # while current_date <= end_date:
        #     data = self.get_station_data(station_id, current_date.year, current_date.month, current_date.day)

        #     if data:
        #         # 保存数据到文件 data/station_{station_id}/{date}.json
        #         date_str = current_date.strftime("%Y-%m-%d")
        #         station_dir = Path(self.data_dir) / f"station_{station_id}"
        #         station_dir.mkdir(parents=True, exist_ok=True)
        #         filename = station_dir / f"{date_str}.json"
        #         logger.info(f"正在保存数据到 {filename}")
        #         with open(filename, "w", encoding="utf-8") as f:
        #             json.dump(data, f, ensure_ascii=False, indent=4)

        #         logger.info(f"数据已保存到 {filename}")

        #         logger.info(
        #             f"已获取 {current_date.strftime('%Y-%m-%d')} 的数据，成功: {success_count}，失败: {fail_count}"
        #         )
        #         success_count += 1
        #     else:
        #         fail_count += 1

        #     current_date += timedelta(days=1)
        #     time.sleep(1)  # 避免请求过于频繁，休眠1秒


        while current_date <= end_date:
            # 1) 抓电站数据
            station_data = self.get_station_data(station_id,
                                                current_date.year,
                                                current_date.month,
                                                current_date.day)
            # 2) 抓天气数据
            weather_data = self.get_weather_data(current_date, station_id)

            date_str = current_date.strftime("%Y-%m-%d")

            # —— 保存电站数据 —— #
            station_dir = Path(self.data_dir) / f"station_{station_id}"
            station_dir.mkdir(exist_ok=True, parents=True)
            with open(station_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
                json.dump(station_data, f, ensure_ascii=False, indent=4)

            # —— 保存天气数据 —— #
            weather_dir = Path(self.data_dir) / "weather"
            weather_dir.mkdir(exist_ok=True)
            with open(weather_dir / f"{date_str}.json", "w", encoding="utf-8") as f:
                json.dump(weather_data, f, ensure_ascii=False, indent=4)

            # 进度日志
            logger.info(f"{date_str} 数据保存完毕。")

            current_date += timedelta(days=1)
            time.sleep(1)


        logger.info(f"数据获取完成，成功: {success_count}，失败: {fail_count}")

    def close(self) -> None:
        """关闭浏览器"""
        if self.driver:
            logger.info("正在关闭浏览器...")
            self.driver.quit()
            self.driver = None


def main():
    """主函数"""
    # 初始化配置
    config_manager = ConfigManager(config_file="D:\八所\功率预测\功率\config.ini")

    # 创建爬虫实例
    crawler = StationCrawler(config_manager)

    try:
        # 登录
        if not crawler.login():
            logger.error("登录失败，程序退出")
            return

        # 获取配置的电站ID和日期范围
        station_id = config_manager.get("crawler", "default_station_id")
        start_year = config_manager.get_int("date_range", "start_year")
        start_month = config_manager.get_int("date_range", "start_month")
        start_day = config_manager.get_int("date_range", "start_day")
        end_year = config_manager.get_int("date_range", "end_year")
        end_month = config_manager.get_int("date_range", "end_month")
        end_day = config_manager.get_int("date_range", "end_day")

        # 创建日期对象
        start_date = datetime(start_year, start_month, start_day)
        print(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
        end_date = datetime(end_year, end_month, end_day)

        # 获取数据
        crawler.get_data_for_period(station_id, start_date, end_date)

    except Exception as e:
        logger.error(f"程序运行过程中发生错误: {str(e)}")
    finally:
        # 关闭浏览器
        crawler.close()


if __name__ == "__main__":
    main()
