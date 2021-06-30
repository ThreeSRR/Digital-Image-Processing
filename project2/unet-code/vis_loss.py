from matplotlib import pyplot as plt
import csv

data = csv.reader(open('experiments/UNet_vessel_seg/log_2021-06-20-21-49.csv').readlines())

epoch, train_loss, val_loss, val_acc, val_f1, val_auc_roc = [], [], [], [], [], []
for line in data:
    if line[0] == 'epoch':
        continue
    epoch.append(int(line[0]))
    train_loss.append(float(line[1]))
    val_loss.append(float(line[2]))
    val_acc.append(float(line[3]))
    val_f1.append(float(line[4]))
    val_auc_roc.append(float(line[5]))

print(epoch)
print(train_loss)
print(val_loss)
print(val_acc)
print(val_f1)
print(val_auc_roc)

plt.plot(epoch, train_loss, label='train_loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.title("Loss against Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig('train_loss.png')

# ===============================
plt.clf()
plt.plot(epoch, val_acc, label='val_acc')
# plt.plot(epoch, val_f1, label='val_f1')
plt.plot(epoch, val_auc_roc, label='val_auc_roc')
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()

plt.savefig('train_acc.png')

# ===============================
plt.clf()
plt.plot(epoch, val_f1, label='val_f1')
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.tight_layout()

plt.savefig('train_f1.png')

