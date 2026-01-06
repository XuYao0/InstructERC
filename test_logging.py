#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日志功能脚本

这个脚本用于快速验证日志系统是否正常工作
"""

import logging
import sys
import os
from datetime import datetime


def setup_logger(log_file=None):
    """设置日志系统"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/test_{timestamp}.log"
    
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def main():
    """测试日志功能"""
    logger, log_file = setup_logger()
    
    logging.info("="*60)
    logging.info("日志功能测试")
    logging.info("="*60)
    logging.info(f"日志文件路径: {log_file}")
    logging.info("="*60)
    logging.info("")
    
    # 测试各种日志级别
    logging.info("✅ INFO 级别日志测试")
    logging.warning("⚠️  WARNING 级别日志测试")
    logging.error("❌ ERROR 级别日志测试")
    
    # 测试中文字符
    logging.info("\n测试中文字符:")
    logging.info("  你好，世界！")
    logging.info("  这是一个中文测试")
    logging.info("  情感标签: neutral, joyful, sad, angry")
    
    # 测试格式化输出
    logging.info("\n测试格式化输出:")
    accuracy = 0.8523
    f1_score = 0.7891
    logging.info(f"  准确率: {accuracy*100:.2f}%")
    logging.info(f"  F1 分数: {f1_score*100:.2f}%")
    
    # 测试多行输出
    logging.info("\n测试多行输出:")
    report = """
              precision    recall  f1-score   support

     neutral     0.8500    0.9000    0.8743       100
      joyful     0.7800    0.8200    0.8000        50
         sad     0.7200    0.6800    0.7000        30
    """
    logging.info(report)
    
    # 测试特殊字符
    logging.info("\n测试特殊字符:")
    logging.info("  ✓ ✗ ★ ♥ → ← ↑ ↓")
    logging.info("  🎯 🚀 ✨ 📊 📈 🔧")
    
    logging.info("\n" + "="*60)
    logging.info("✅ 日志功能测试完成!")
    logging.info(f"📁 日志已保存到: {log_file}")
    logging.info("="*60)
    
    # 验证日志文件是否存在
    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        logging.info(f"\n✅ 验证通过: 日志文件存在，大小 {file_size} 字节")
    else:
        logging.error(f"\n❌ 验证失败: 日志文件不存在")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

