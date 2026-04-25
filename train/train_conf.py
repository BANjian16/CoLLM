import torch, torch.nn.functional as F
from datasets.cmapss import CMAPSSDataset
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection


# 论文中的置信度监督目标：
# Q* = 1 - tanh(|prediction - target| / alpha)
# 直观上讲，预测误差越小，Q* 越接近 1；预测误差越大，Q* 越接近 0。
# 这样就把“回归误差”转换成了“可学习的连续置信度标签”。
def Qstar(p,y,a=5): return 1-torch.tanh(torch.abs(p-y)/a)

device = torch.device('cpu')
# 加载已经预训练好的两个基础预测器。
# 这对应论文的分阶段训练：先训练 S 和 L，再训练置信度相关模块。
S = SmallModel().to(device); S.load_state_dict(torch.load('small.pt'))
L = GPT2TimeSeries().to(device); L.load_state_dict(torch.load('large.pt'))

# 初始化 CoLLM 在基础预测器之上新增的两个置信度模块：
# 1. Fz：为小模型学习模糊置信度 Q_s
# 2. Rf：为大模型学习自反思置信度 Q_l
Fz = FuzzyDecisionAgent(32,50).to(device)
Rf = SelfReflection(768,12).to(device)


# 此阶段只优化 Fz 和 Rf。
# 论文强调的训练策略正是：
# 先学会“怎么预测”，再在冻结预测器的前提下学习“什么时候信任哪个模型”。
opt = torch.optim.Adam(list(Fz.parameters())+list(Rf.parameters()),1e-3)
loader = torch.utils.data.DataLoader(CMAPSSDataset('../data/CMAPSS/train_FD001.txt'),128,True)
EPOCHS = 100
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        # 在不更新 S 和 L 参数的情况下抽取：
        # 1. 两个模型的预测值 ys / yl
        # 2. 两个模型的中间表示 ps / pl
        # 后者将分别作为 Fz 与 Rf 的输入特征。
        with torch.no_grad():
            ys, ps = S(x)
            yl, pl = L(x)
        # 训练目标分为两部分：
        # 1. Fz(phi_s) 拟合由小模型误差构造的监督标签 Q*_s
        # 2. Rf(phi_l) 拟合由大模型误差构造的监督标签 Q*_l
        # 两者都采用均方误差损失，对应论文公式中的 confidence regression。
        loss = F.mse_loss(Fz(ps),Qstar(ys,y)) + F.mse_loss(Rf(pl),Qstar(yl,y))
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * x.size(0)
    epoch_loss /= len(loader.dataset)
    print(f'Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}')

# 保存训练好的模糊决策代理与自反思模块，
# 供后续 CoLLM 协同推理阶段直接加载使用。
torch.save(Fz.state_dict(),'fuzzy.pt'); torch.save(Rf.state_dict(),'reflect.pt')
