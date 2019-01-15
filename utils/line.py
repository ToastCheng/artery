from requests import get


class LineBot:
	def __init__(self):
		self.url = "https://asia-northeast1-linebot-toefl.cloudfunctions.net/push_message"

	def print(self, message):
		return get(self.url, params={"message": message})




if __name__ == "__main__":
	bot = LineBot()
	bot.print("process start")
	for i in range(1000000000):
		ii = 0
	bot.print("")

