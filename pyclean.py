#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : pyclean.py
# Create date : 2016-08-14 03:00
# Modified date : 2019-03-18 16:52
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################

import os
import etc

def delete_log_dir():
    os.popen("rm -rf ./log")

def delete_log_db():
    log_db_name = etc.DATABASE_NAME
    log_db_path = etc.DATABASE_PATH
    os.popen("rm ./%s/%s.db" % (log_db_path,log_db_name))

def delete_sqltest_db():
    os.popen("rm ./db/sqltest.db")

def delete_pydb_test_db():
    os.popen("rm ./db/pydb_test.db")

def delete_pyc():
    os.popen("rm ./*.pyc")

def delete_pycache():
    os.popen("rm -rf ./__pycache__")

def delete_tmp_file():
    tmp_file_name = etc.TMP_FILE_PATH = "./tmp_file/"
    os.popen("rm -rf %s" % tmp_file_name)

def delete_model_dir():
    os.popen("rm -rf model_dir")

def delete_tensorboard_logs():
    os.popen("rm -rf tensorboard_logs")

if __name__ == "__main__":
    delete_log_dir()
    #delete_log_db()
    #delete_sqltest_db()
    #delete_pydb_test_db()
    delete_pyc()
    delete_tmp_file()
    delete_pycache()

    delete_model_dir()
    delete_tensorboard_logs()
