import matplotlib.pyplot as plt

def wav_show(data):
	plt.figure(figsize=(16,8))
	plt.plot(data) 
	plt.show()

def wav_show_compare(data1,data2):
	plt.figure(figsize=(16,8))
	plt.subplot(211)
	plt.plot(data1)
	plt.subplot(212)
	plt.plot(data2)
	plt.show()

def spectro_show(data):
	fig = plt.figure(figsize=(16,8))
	heat_map = plt.pcolor(data)
	fig.colorbar(mappable=heat_map)
	plt.ylabel('Windows')
	plt.xlabel("Features")
	plt.tight_layout()
	plt.show()

