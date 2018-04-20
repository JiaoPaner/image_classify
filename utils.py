# -*- coding: utf-8 -*-
# @Time    : 2018/4/14 17:57
# @Author  : jiaopan
# @Email   : jiaopaner@163.com
import configparser

def configUtil(confir_file,name, key):
    """
    读取配置文件
    param confir_file:配置文件
    param name:配置项名
    param key:配置键名
    """
    conf = configparser.ConfigParser()
    try:conf.read(confir_file)
    except:print(confir_file+"不存在或"+name+"/"+"key未配置")
    return conf.get(name, key)