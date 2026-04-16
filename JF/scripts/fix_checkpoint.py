import torch

# 1. 指向你最原始的那个未修改过的权重文件
original_ckpt_path = "logs/rsl_rl/Mvr_10dof/best/model_2500.pt"
checkpoint = torch.load(original_ckpt_path)

# 获取旧版的总权重字典
if 'model_state_dict' in checkpoint:
    old_state_dict = checkpoint['model_state_dict']
else:
    print("未找到 model_state_dict，可能文件结构有异，请检查。")
    exit()

# 准备新版的空字典
new_actor_dict = {}
new_critic_dict = {}

print("开始深度转换网络层名称...")

# 2. 遍历旧字典，替换名称并分发到新字典
for old_key, tensor_value in old_state_dict.items():
    # 处理 Actor 的层
    if old_key.startswith('actor.'):
        new_key = old_key.replace('actor.', 'mlp.')
        new_actor_dict[new_key] = tensor_value
    
    # 处理 Actor 的标准差 (std)
    elif old_key == 'std':
        new_actor_dict['distribution.std_param'] = tensor_value
        
    # 处理 Critic 的层
    elif old_key.startswith('critic.'):
        # 新版 Critic 也是独立的 MLPModel，所以前缀也叫 mlp.
        new_key = old_key.replace('critic.', 'mlp.')
        new_critic_dict[new_key] = tensor_value
        
    # 如果有其他额外的参数（比如自适应模块的权重等），可以按需在这里添加 elif
    else:
        print(f"警告: 忽略了未知的旧权重层: {old_key}")

# 3. 将新的字典写回 checkpoint
checkpoint['actor_state_dict'] = new_actor_dict
checkpoint['critic_state_dict'] = new_critic_dict

# 移除旧的 model_state_dict 避免冗余
del checkpoint['model_state_dict']

# 4. 保存为新文件
new_ckpt_path = "logs/rsl_rl/Mvr_10dof/best/model_2500_fixed.pt"
torch.save(checkpoint, new_ckpt_path)

print(f"转换大功告成！已保存至: {new_ckpt_path}")
print("现在你可以使用这个新文件运行 play.py 了。")