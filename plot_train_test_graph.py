# Example
# str1 = "this is string example....wow!!!";
# str2 = "exam";
#
# print str1.find(str2)
# print str1.find(str2, 10)
# print str1.find(str2, 40)

# Results
# 15
# 15
# -1


# to extract the string


import matplotlib.pyplot as plt
import numpy as np

f = open('black-white.txt', 'r')

train_acc = []
test_acc = []
train_loss = []
val_loss = []


for line in f :
    index = line.find("val_loss")
    if line.find("val_loss:")!= -1:
        val_loss.append(float(line[index+10:index+16]))
        train_acc.append(float(line[index -9:index - 3]))
        test_acc.append(float(line[-6:-1]))
        train_loss.append(float(line[index -23 :index - 17]))



print(train_acc)
print(len(train_acc))



epoch = np.arange(1,101)

# x-axis values
x = epoch
# y-axis values
y1 = train_loss
y2 = val_loss
y3 = train_acc
y4 = test_acc

print(y1)
print(y2)
# plotting points as a scatter plot
plt.plot(np.array(y3), label="train_acc", lw=2, color="navy")
plt.plot(np.array(y4), label="test_acc", lw=2, color="darkorange")


#plt.ylim(plt.ylim()[::-1])
# plt.xlim([0, 100])
# plt.ylim([0.0, 0.9])

# x-axis label
plt.xlabel('x - axis')
# frequency label
plt.ylabel('y - axis')
# plot title
plt.title('Train vs Test accuracy')
# showing legend
plt.legend()

# function to show the plot
plt.show()

################################## Loss Funxtion show

# plotting points as a scatter plot
plt.plot(np.array(y1), label="train_loss", lw=2, color="navy")
plt.plot(np.array(y2), label="val_loss", lw=2, color="darkorange")


#plt.ylim(plt.ylim()[::-1])
# plt.xlim([0, 100])
# plt.ylim([0.0, 0.9])

# x-axis label
plt.xlabel('x - axis')
# frequency label
plt.ylabel('y - axis')
# plot title
plt.title('Train Loss vs Validation Loss')
# showing legend
plt.legend()

# function to show the plot
plt.show()




