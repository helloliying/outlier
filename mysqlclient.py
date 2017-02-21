#-*-coding:utf8-*-
#! /usr/bin/python
#encoding=utf-8
import json
import pymysql


class MysqlClient:
	
 	def __init__(self,host,user,passwd,db,port):
 		self.db = 'dianping'
 		self.conn = pymysql.connect(host=host,user=user,passwd=passwd,db=db,port=port,charset='utf8')
 		self.cur = self.conn.cursor()

 	def close_Conn(self):
 		self.cur.close()
 		self.conn.close()
	
 	def insert(self,dic_list,table_name,**postDic):
 		sql_join = "insert into  "+ table_name +"(" + ','.join(dic_list) + ") values  (\'%("+')s\',\'%('.join(dic_list)+") s\' )"
 		sql = sql_join % postDic
 		self.cur.execute(sql)
 		self.conn.commit()

 	def getData(self,sql):
 		self.cur.execute(sql);
 		data = self.cur.fetchall()
 		return data

