# 创建一个名为 test_highway_env.py 的文件
import gymnasium as gym
import highway_env

print("注册的环境:")
for env_id in [env_id for env_id in gym.envs.registry.keys() if env_id.startswith('highway-') or env_id.startswith('merge-')]:
    print(f"  - {env_id}")

print("\n创建原始合流环境...")
env = gym.make('merge-v0', render_mode="rgb_array")
print("环境创建成功")

print("\n检查环境类属性...")
print("所有属性和方法:")
for attr in dir(env):
    if not attr.startswith('__'):
        print(f"  - {attr}")

print("\n检查env.unwrapped的属性...")
for attr in dir(env.unwrapped):
    if not attr.startswith('__'):
        print(f"  - {attr}")

print("\n尝试重置环境...")
try:
    obs, info = env.reset()
    print("环境重置成功!")
    print(f"观测类型: {type(obs)}")
    print(f"观测形状: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
except Exception as e:
    print(f"环境重置出错: {e}")
    import traceback
    traceback.print_exc()