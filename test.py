# ======================= Pytorch Lib =============================
import torch
from torchvision import transforms
# ======================= My Lib ===================================
from lib.NLEDN import NLEDN
from lib.data_loader import DataSet
from lib.utils import calc_psnr, calc_ssim
# ======================= Config File ===============================
import config as cfg
# ======================= Origin Lib ================================
import os
import time

# ======================= Config ===================================
print('-' * 40)
print('cuda number:', cfg.CUDA_NUMBER, '\n')
print('test dir:', cfg.test_dir)

# ======================= DataSet ===================================
dataset = DataSet(cfg)
test_batches = dataset.test_loader.__len__()
test_samples = dataset.test_dataset.__len__()

print('Test: %d batches, %d samples' % (test_batches, test_samples))
print('-' * 40 + '\n')

# ==================== Network ======================
net = NLEDN()

model_path = os.path.join(cfg.weight_path)
print('Loading weights From %s' % model_path)
net.load_state_dict(torch.load(model_path))

# ================== Network to GPU =========================
if torch.cuda.is_available():
    net.cuda(cfg.CUDA_NUMBER)

total_pnsr = 0
total_ssim = 0


torch.set_grad_enabled(False)
net.eval()
for batch_index, (img_batch, label_batch, name_list) in enumerate(dataset.test_loader):
    print('[%d/%d]' % (batch_index, test_batches), name_list[0])

    if torch.cuda.is_available():
        img_batch = img_batch.cuda(cfg.CUDA_NUMBER)
        label_batch = label_batch.cuda(cfg.CUDA_NUMBER)

    label_res_batch = img_batch - label_batch

    # ------------------------ Res Predict ------------------------
    prediction_res_batch = net(img_batch)

    prediction_batch = img_batch - prediction_res_batch
    prediction_batch = torch.clamp(prediction_batch, 0, 1)

    # ------------------------ Save Image And Calc Metric------------------------
    prediction_PIL = transforms.ToPILImage()(prediction_batch[0].cpu().data)
    label_batch_PIL = transforms.ToPILImage()(label_batch[0].cpu().data)

    pnsr = calc_psnr(prediction_PIL, label_batch_PIL)
    ssim = calc_ssim(prediction_PIL, label_batch_PIL)
    total_pnsr += pnsr
    total_ssim += ssim

    img = torch.cat([img_batch, prediction_batch, label_batch], dim=3)
    img = transforms.ToPILImage()(img[0].cpu().data)

    img.save(os.path.join(cfg.test_compare_results_dir, name_list[0]), format='png')
    prediction_PIL.save(os.path.join(cfg.test_results_dir, name_list[0]), format='png')

# ------------------------ Calc Mean Metric ------------------------
mean_pnsr = total_pnsr / test_batches
mean_ssim = total_ssim / test_batches
print('PNSR:%.4f SSIM:%.4f' % (mean_pnsr, mean_ssim))
