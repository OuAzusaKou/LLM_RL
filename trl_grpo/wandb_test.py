import wandb
import numpy as np
import time
import os

def main():
    # 使用指定的 key 登录 wandb
    wandb.login(key="64ae865da9e03328b96e0202de725fe607d6dd12")
    
    # 初始化 wandb，使用相同的 entity 和 project，设置为离线模式
    wandb.init(
        entity="940257123",
        project="bridgevla",
        name="wandb-test-run",
        # mode="offline",  # 设置为离线模式
        config={
            "learning_rate": 0.01,
            "architecture": "test",
            "dataset": "test-data",
            "epochs": 5,
        }
    )
    
    # 模拟训练循环
    for epoch in range(5):
        # 模拟一些训练指标
        train_loss = np.random.normal(0.5, 0.1)
        val_loss = np.random.normal(0.4, 0.1)
        accuracy = np.random.uniform(0.8, 0.95)
        
        # 记录指标
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        })
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, accuracy={accuracy:.4f}")
        time.sleep(1)  # 添加一些延迟以便观察
    
    # 完成运行
    wandb.finish()
    
    print("\n离线数据已保存。要同步到线上，请运行以下命令：")
    print("wandb sync wandb/latest-run")

if __name__ == "__main__":
    main()
