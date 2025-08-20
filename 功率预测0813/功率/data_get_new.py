#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爬虫程序：获取电站历史数据 + 天气数据
作用：自动登录网站并获取指定时间段内的电站数据 & 天气数据
"""

import os
import json
import time
import logging
import requests
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
    handlers=[
        logging.FileHandler("crawler.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
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
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            self._create_default_config()
        else:
            self.config.read(config_file, encoding="utf-8")
        logger.info("当前配置文件加载完毕：%s", config_file)

    def _create_default_config(self):
        for section, opts in DEFAULT_CONFIG.items():
            self.config[section] = opts
        with open(self.config_file, "w", encoding="utf-8") as f:
            self.config.write(f)
        logger.info("已创建默认配置文件: %s", self.config_file)

    def get(self, section: str, option: str, fallback: Any = None) -> str:
        return self.config.get(section, option, fallback=fallback)

    def get_int(self, section: str, option: str, fallback: int = 0) -> int:
        return self.config.getint(section, option, fallback=fallback)

class StationCrawler:
    def __init__(self, config: ConfigManager, station_id: str):
        self.config = config
        self.station_id = station_id
        self.driver = None
        self.cookie = None
        self.session = requests.Session()
        # 数据目录
        self.data_dir = self.config.get("crawler", "data_dir")
        os.makedirs(self.data_dir, exist_ok=True)

    def init_browser(self):
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )
        logger.info("浏览器初始化完成")

    def login(self) -> bool:
        if not self.driver:
            self.init_browser()
        login_url = self.config.get("website", "login_url")
        self.driver.get(login_url)
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@placeholder='请输入用户名']"))
        )
        self.driver.find_element(By.XPATH, "//input[@placeholder='请输入用户名']").send_keys(
            self.config.get("user", "username")
        )
        self.driver.find_element(By.XPATH, "//input[@placeholder='请输入密码']").send_keys(
            self.config.get("user", "password")
        )
        logger.info("请手动输入验证码并登录，登录后按回车继续...")
        input()
        # 获取 cookie 并写入 requests session
        self.cookie = self.driver.get_cookies()
        for ck in self.cookie:
            self.session.cookies.set(ck['name'], ck['value'])
        logger.info("登录成功，Session 已设置 Cookie")
        return True

    def get_station_data(self, station_id: str, date: datetime) -> Optional[Dict]:
        y, m, d = date.year, f"{date.month:02d}", f"{date.day:02d}"
        url = (
            f"{self.config.get('website','api_base_url')}/{station_id}"
            f"?year={y}&month={m}&day={d}"
        )
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Authorization': f"Bearer {self.session.cookies.get_dict().get(self.cookie[0]['name'])}",
        }
        resp = self.session.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Station API %s 返回 %s", url, resp.status_code)
        return None

    def fetch_area_ids(self) -> Tuple[int,int,int]:
        url = (
            f"{self.config.get('website','dashboard_url')}/searchAreaByIds"
            f"?plantIds={self.station_id}"
        )
        resp = self.session.get(url)
        data = resp.json().get('data', [{}])[0]
        return data.get('regionNationId'), data.get('regionLevel1'), data.get('regionLevel2')

    def get_weather_data(self, date: datetime) -> Optional[List[Dict]]:
        rn, rl1, rl2 = self.fetch_area_ids()
        y, m, d = date.year, f"{date.month:02d}", f"{date.day:02d}"
        url = (
            f"{self.config.get('weather','weather_url')}"
            f"?regionNationId={rn}&regionLevel1={rl1}&regionLevel2={rl2}"
            f"&year={y}&month={m}&day={d}&lan=zh"
        )
        resp = self.session.get(url)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Weather API %s 返回 %s", url, resp.status_code)
        return None

    def get_data_for_period(
        self, station_id: str, start: datetime, end: datetime
    ) -> None:
        current = start
        while current <= end:
            sd = self.get_station_data(station_id, current)
            wd = self.get_weather_data(current)
            date_str = current.strftime('%Y-%m-%d')
            # 保存电站数据
            sd_dir = Path(self.data_dir)/f"station_{station_id}"
            sd_dir.mkdir(parents=True, exist_ok=True)
            with open(sd_dir/f"{date_str}.json","w",encoding='utf-8') as f:
                json.dump(sd or {}, f, ensure_ascii=False, indent=2)
            # 保存天气数据
            wd_dir = Path(self.data_dir)/"weather"
            wd_dir.mkdir(parents=True, exist_ok=True)
            with open(wd_dir/f"{date_str}.json","w",encoding='utf-8') as f:
                json.dump(wd or [], f, ensure_ascii=False, indent=2)
            logger.info(f"{date_str} 数据已保存")
            current += timedelta(days=1)
            time.sleep(1)

    def close(self):
        if self.driver:
            self.driver.quit()


def main():
    cfg = ConfigManager(config_file="config.ini")
    station_id = cfg.get('crawler','default_station_id')
    start = datetime(
        cfg.get_int('date_range','start_year'),
        cfg.get_int('date_range','start_month'),
        cfg.get_int('date_range','start_day')
    )
    end = datetime(
        cfg.get_int('date_range','end_year'),
        cfg.get_int('date_range','end_month'),
        cfg.get_int('date_range','end_day')
    )
    crawler = StationCrawler(cfg, station_id)
    try:
        crawler.login()
        crawler.get_data_for_period(station_id, start, end)
    finally:
        crawler.close()

if __name__ == '__main__':
    main()
