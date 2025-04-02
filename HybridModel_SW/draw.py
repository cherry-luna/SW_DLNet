import torch
from torchinfo import summary
from torchview import draw_graph


class ModelVisualizer:
    """模型可视化工具类"""
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape # (B,L)

    def visualize(self):
        """执行完整可视化流程"""
        self._summary()
        self._model_visualize()

    def _summary(self):
        """打印模型摘要"""
        print("\n模型结构摘要：")
        summary(
            self.model,
            input_data=[  # 修正参数名并添加数据长度
                torch.randn(128, *self.input_shape),  # [batch, channels, length]
                torch.tensor([self.input_shape[-1]])  # 数据长度参数
            ],
            col_names=["input_size", "output_size", "num_params"],
            verbose=1
        )

    def _model_visualize(self):
        """使用HiddenLayer生成架构图"""
        # 生成符合模型输入的虚拟数据
        dummy_input = torch.zeros(128, *self.input_shape)  # [batch, channels, length]
        dummy_len = torch.tensor([self.input_shape[-1]]).long()  # 数据长度参数

        model_graph = draw_graph(
            self.model,
            (dummy_input, dummy_len),
            depth=4,
            device='cpu',
            save_graph = True,
            filename = "architecture"
        )


if __name__ == "__main__":
    from model_arch import HybridModel

    model = HybridModel()
    input_shape = (2, 4096)

    # 执行可视化
    visualizer = ModelVisualizer(model, input_shape)
    visualizer.visualize()

    print("可视化完成！请检查以下文件：")
    print("- architecture.png")