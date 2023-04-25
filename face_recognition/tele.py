import matplotlib.pyplot as plt
losses = {"train":[], "val":[]}
acces = {"train":[], "val":[]}
acces['train'].append([1,3,2,5,6,2,3,5,])
acces['val'].append([5,8,9,6,32,12,])
losses['train'].append([4,6,7,1,1,6,9,1,])
losses['val'].append([1,7,8,2,6,1,2])

plt.title('matplotlib.pyplot.plot() example 1')

plt.plot(acces["train"],label="train_acc",color = 'r')
plt.plot(losses["train"],label="train_loss", color = 'g')

plt.plot(acces["val"],label="val_acc", color = 'b')
plt.plot(losses["val"],label="val_loss", color = 'm')

plt.xlabel("epoch")
plt.ylabel("values")
plt.draw()
plt.show()

