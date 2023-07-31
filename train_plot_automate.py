import matplotlib.pyplot as plt
import os

# 設定圖片和標題的列表
image_folder = 'result/train/'
start_epoch = 100
end_epoch = 1000

images = [os.path.join(image_folder, f'loss_epoch{i}.png') for i in range(start_epoch, end_epoch + 100, 100)]
titles = [f'Epoch={i}' for i in range(start_epoch, end_epoch + 100, 100)]

plt.figure(dpi=100)
# 創建一個2x5的子圖表格
fig, axes = plt.subplots(2, 5, figsize=(24, 12))

# 將圖片和標題逐一添加到子圖中
for i, ax in enumerate(axes.flat):
    # 讀取圖片並顯示
    img = plt.imread(images[i])
    ax.imshow(img)
    # 設定標題
    ax.set_title(titles[i])
    # 隱藏軸刻度
    ax.axis('off')

# 調整子圖之間的間距
plt.tight_layout()

# 增加大標題
plt.suptitle('Loss', fontsize=18, fontweight='bold')

plt.savefig(os.path.join(image_folder, 'final/loss.png'))
# 顯示圖片
plt.show()
