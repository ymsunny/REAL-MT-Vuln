import json
import matplotlib.pyplot as plt
import numpy as np

# 定义类型名称映射和顺序
TYPE_MAPPING = {
    'no_hint': 'No Context',
    'groundtruth_meaning': 'Gold Meaning',
    'perturbation_meaning': 'Structure-Perturbed Gold Meaning',
    'literal_meaning': 'Literal Meaning',
    'perturbation_literal_meaning': 'Semantic-Perturbed Literal Meaning',
    'opposite_meaning': 'Opposite Meaning'
}

# 定义显示顺序
TYPE_ORDER = [
    'No Context',
    'Gold Meaning',
    'Structure-Perturbed Gold Meaning',
    'Literal Meaning',
    'Semantic-Perturbed Literal Meaning',
    'Opposite Meaning'
]

# 定义颜色方案
def get_color_scheme(types):
    # 使用更鲜明的配色
    colors = ['#2ecc71', '#3498db', '#e74c3c',    # 翠绿色、天蓝色、鲜红色
              '#9b59b6', '#f1c40f', '#1abc9c']    # 紫色、金黄色、青绿色
    
    # 创建颜色映射字典，按照TYPE_ORDER的顺序
    type_to_color = {}
    for i, t in enumerate(TYPE_ORDER):
        type_to_color[t] = plt.cm.colors.to_rgba(colors[i])
    
    return type_to_color

# 读取JSON文件
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 收集所有可能的类型，排除similar_literal_meaning
        types = set()
        for item in data:
            types.update(item['idiom_attention_weight'].keys())
            types.update(item['meaning_attention_weight'].keys())
        types.discard('similar_literal_meaning')  # 移除不需要的类型
        types = sorted(list(types))
        
        # 收集每个样本的数据
        sample_data = []
        for item in data:
            for type_name in types:
                if type_name in item['idiom_attention_weight'] and type_name in item['meaning_attention_weight']:
                    sample_data.append({
                        'idiom': item['idiom_attention_weight'][type_name],
                        'meaning': item['meaning_attention_weight'][type_name],
                        'type': TYPE_MAPPING.get(type_name, type_name),  # 使用映射后的名称
                        'source': item['source']
                    })
        return sample_data, TYPE_ORDER  # 返回固定顺序的类型列表

# 语言对列表
language_pairs = ['fi-en', 'ja-en', 'fr-en']

# 创建一个大图，包含三个子图（每个语言对一个）
fig, axes = plt.subplots(1, 3, figsize=(24, 9))  # 稍微减小图的高度
#fig.suptitle('Idiom vs Meaning Attention Weight Scatter Plot', fontsize=16, y=1.02)

# 获取颜色映射
type_to_color = get_color_scheme(TYPE_ORDER)

# 设置全局字体大小
FONT_SIZE = {
    'title': 24,    # 标题（语言对）
    'label': 20,    # 坐标轴标签
    'legend': 18,   # 图例
    'tick': 16      # 刻度值
}

# 为每个语言对创建散点图
for idx, lang_pair in enumerate(language_pairs):
    file_path = f'{lang_pair}_meaning_en_Qwen2.5-7B-Instruct_2.json'
    sample_data, _ = load_json_data(file_path)
    
    # 按类型分组数据
    data_by_type = {}
    for item in sample_data:
        if item['type'] not in data_by_type:
            data_by_type[item['type']] = {'idiom': [], 'meaning': []}
        data_by_type[item['type']]['idiom'].append(item['idiom'])
        data_by_type[item['type']]['meaning'].append(item['meaning'])
    
    ax = axes[idx]
    
    # 按指定顺序绘制散点图
    for type_name in TYPE_ORDER:
        if type_name in data_by_type:
            data = data_by_type[type_name]
            ax.scatter(data['idiom'], data['meaning'], 
                      c=[type_to_color[type_name]], label=type_name if idx == 0 else "", alpha=0.6)
            
            # 添加趋势线
            z = np.polyfit(data['idiom'], data['meaning'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(data['idiom']), max(data['idiom']), 100)
            ax.plot(x_trend, p(x_trend), '--', color=type_to_color[type_name], alpha=0.5)
    
    # 添加对角线
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x' if idx == 0 else "")
    
    # 设置图表属性
    ax.set_title(f'{lang_pair}', fontsize=FONT_SIZE['title'])
    ax.set_xlabel('Idiom Attention Weight', fontsize=FONT_SIZE['label'])
    if idx == 0:  # 只在最左边的子图设置y轴标签
        ax.set_ylabel('Meaning Attention Weight', fontsize=FONT_SIZE['label'])
    else:  # 其他子图隐藏y轴标签
        ax.set_ylabel('')
    ax.grid(True, linestyle='--', alpha=0.3)
    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE['tick'])

# 获取第一个子图的图例并重新排序
handles, labels = axes[0].get_legend_handles_labels()
# 重新排序图例
order = []
for t in TYPE_ORDER + ['y=x']:  # 添加y=x到最后
    try:
        order.append(labels.index(t))
    except ValueError:
        continue

# 在图表上方添加图例
fig.legend([handles[idx] for idx in order],
          [labels[idx] for idx in order],
          loc='upper center', bbox_to_anchor=(0.5, 1.08),  # 减小与图表的距离
          ncol=len(order), frameon=False,
          fontsize=FONT_SIZE['legend'])

# 调整布局
plt.tight_layout()
# 调整子图之间的间距和与x轴标签的距离
plt.subplots_adjust(bottom=0.2, wspace=0.2)  # 增加底部和子图之间的间距
plt.savefig('attention_weights_scatter.pdf', bbox_inches='tight', dpi=300)
plt.close()

# 创建箱线图比较不同类型
fig, axes = plt.subplots(1, 3, figsize=(24, 9))  # 稍微减小图的高度
#fig.suptitle('Distribution of Attention Weights by Type', fontsize=16, y=1.02)

for idx, lang_pair in enumerate(language_pairs):
    file_path = f'{lang_pair}_meaning_en_Qwen2.5-7B-Instruct_2.json'
    sample_data, _ = load_json_data(file_path)
    
    # 按类型分组数据
    data_by_type = {}
    for item in sample_data:
        if item['type'] not in data_by_type:
            data_by_type[item['type']] = {'idiom': [], 'meaning': []}
        data_by_type[item['type']]['idiom'].append(item['idiom'])
        data_by_type[item['type']]['meaning'].append(item['meaning'])
    
    ax = axes[idx]
    
    # 准备箱线图数据
    labels = []
    idiom_data = []
    meaning_data = []
    box_colors = []
    
    # 使用指定顺序
    for type_name in TYPE_ORDER:
        if type_name in data_by_type:
            labels.append(type_name)
            idiom_data.append(data_by_type[type_name]['idiom'])
            meaning_data.append(data_by_type[type_name]['meaning'])
            box_colors.append(type_to_color[type_name])
    
    # 绘制箱线图
    positions = np.arange(len(labels)) * 3
    width = 1.0
    
    # 创建箱线图，确保标签数量匹配
    bp1 = ax.boxplot(idiom_data, positions=positions-width/2, widths=width,
                     patch_artist=True)
    bp2 = ax.boxplot(meaning_data, positions=positions+width/2, widths=width,
                     patch_artist=True)
    
    # 设置箱线图颜色
    for i, box in enumerate(bp1['boxes']):
        box.set(facecolor=box_colors[i], alpha=0.3)
    for i, box in enumerate(bp2['boxes']):
        box.set(facecolor=box_colors[i], alpha=0.3)
    
    # 设置图表属性
    ax.set_title(f'{lang_pair}', fontsize=FONT_SIZE['title'])
    if idx == 0:  # 只在最左边的子图设置y轴标签
        ax.set_ylabel('Attention Weight', fontsize=FONT_SIZE['label'])
    else:  # 其他子图隐藏y轴标签
        ax.set_ylabel('')
    
    # 设置x轴标签位置和文本
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=FONT_SIZE['tick'])
    # 设置y轴刻度字体大小
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZE['tick'])

# 在图表上方添加图例
legend_elements = [
    plt.Line2D([0], [0], color='white', markerfacecolor='white', alpha=0.3, label='Idiom Attention'),
    plt.Line2D([0], [0], color='white', markerfacecolor='white', alpha=0.3, label='Meaning Attention')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08),  # 减小与图表的距离
          ncol=2, frameon=False, fontsize=FONT_SIZE['legend'])

# 调整布局
plt.tight_layout()
# 调整子图之间的间距和与x轴标签的距离
plt.subplots_adjust(bottom=0.3, wspace=0.2)  # 增加底部和子图之间的间距
plt.savefig('attention_weights_distribution.pdf', bbox_inches='tight', dpi=300)
plt.close()
