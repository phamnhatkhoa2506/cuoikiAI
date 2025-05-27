from numpy import genfromtxt
import matplotlib.pyplot as plt
# pick data
data = genfromtxt('data.csv', delimiter=',')
areas = data[:, 0]
prices = data[:, 1]
N = areas.size
plt.scatter(areas, prices)
plt.xlabel('Diện tích nhà ')
plt.ylabel('Giá nhà ')
plt.xlim(3, 7)
plt.ylim(4, 10)
plt.show()
# param
max_epoch = 10
lr = 0.001
w,b = -0.34, 0.45
losses = []
for _ in range(max_epoch):
    for i in range(N):
        # select sample
        x = areas[i]
        y = prices[i:i+1]
        # predict y_hat
        y_hat = w * x + b
        # compute loss
        loss = (y_hat - y) * (y_hat - y)
        losses.append(loss)
        # gradient
        d_w = 2 * x * (y_hat - y)
        d_b = 2 * (y_hat - y)
        # update
        w = w - lr * d_w
        b = b - lr * d_b
print("loss:\n", losses[-1])
plt.plot(losses) # test with losses[3:]
plt.xlabel('iteration')
plt.ylabel('losses')
plt.show()

x_data = range(2, 8)
y_data = [x*w + b for x in x_data]
plt.plot(x_data, y_data)
plt.scatter(areas, prices)
plt.xlabel('Diện tích nhà')
plt.ylabel('Giá nhà ')

plt.xlim(3, 7)
plt.ylim(4, 10)
plt.show()