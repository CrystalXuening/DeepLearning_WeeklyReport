# !/usr/bin/python
# -*- coding: UTF-8 -*-

import csv  # CSV(Comma-Separated Values)即逗号分隔值，是纯文本

def save(data):
    # 保存预测数据到result.csv
    with open('./result/result.csv', 'w') as csv_writer:
        # 以写模式打开result.csv，叫做csv_writer
        writer = csv.writer(csv_writer)
        # 写入多行，但注意如果不指定newline='',则每写入一行将有一空行被写入
        writer.writerows(data)


