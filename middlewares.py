//Develop the middleware
import random 
from scrapy.conf import settings
//hahah 
class ProxyMiddleware(object):
	// 这个方法中的代码会在每次爬虫访问网页之前执行
	def process_request(self,request,spider):
		proxy = random.choice(setting['PROXIES'])
		request.meta['proxy'] = proxy


// 打开settings.py，首先添加几个代理IP
PROXIES = ['https://114.217.243.25:8118','https://125.37.175.233:8118',
          'http://1.85.116.218:8118']



//Activiate middleware
//   settings.py
DOWNLOADER_MIDDLEWARES = {
  'AdvanceSpider.middlewares.ProxyMiddleware': 543,
}



