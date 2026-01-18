import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. 读取图片
img = mpimg.imread('experiments_plot2/resnet34_exp4_50_pda3/test_results/pda_maps_paper/regression/class2_val_sample_0000_beta2_0.0867_beta3_0.2000_files_23_beta3.png')
#experiments_plot2/resnet34_exp4_50_pda3/test_results/pda_maps_paper/regression/class2_val_sample_0000_beta2_0.0867_beta3_0.2000_files_23_beta3.png
#experiments_plot2/resnet34_exp4_50_pda3/test_results/pda_maps_paper/classification/class0_val_sample_0090_beta2_0.1133_beta3_0.1333_files_21_cls.png
# 2. 显示图片并裁剪
plt.figure(dpi=300) # 保持高分辨率
# 假设原来的标题大概占据顶部 10% 的位置，我们可以切掉它
# img[y1:y2, x1:x2] -> img[50: , :] 表示切掉顶部前50行像素(具体数值需尝试)
# 也可以不裁剪，直接画个白框盖住，但裁剪更自然
cropped_img = img[130:, :] # 这里的60取决于图片分辨率，试着调整一下

plt.imshow(cropped_img)
plt.axis('off') # 关闭坐标轴（因为原图里已经有坐标轴了）

# 3. 添加新标题
plt.title('b', fontsize=14, fontweight='bold')

# 4. 保存
plt.tight_layout()
plt.savefig('regression.pdf', bbox_inches='tight')
plt.show()