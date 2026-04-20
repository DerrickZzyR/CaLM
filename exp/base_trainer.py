from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, args):
        self.args = args

    # --- 模板方法 (主流程骨架) ---
    def run(self):
        """
        模板方法：流程固定。
        pretrain -> train -> test
        """
        if self.args.phase in ['pretrain', 'all']:
            print("开始预训练...")
            self.pretrain(self.args)
        if self.args.phase in ['train', 'all']:
            print("开始正式训练...")
            self.train(self.args)
        if self.args.phase in ['test', 'all']:
            print("开始测试...")
            self.test(self.args)

    # --- 抽象方法 (子类必须实现) ---
    @abstractmethod
    def pretrain(self, args):
        pass

    @abstractmethod
    def train(self, args):
        pass

    @abstractmethod
    def test(self, args):
        pass