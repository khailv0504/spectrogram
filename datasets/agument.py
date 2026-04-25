import random
import torch

class RFAugment:
    def __init__(self):
        pass

    def add_controlled_awgn(self, spec, original_snr_db, target_snr_db):
        """
        spec: Tensor ảnh (đã chuẩn hóa)
        original_snr_db: SNR thực tế của file (từ metadata, 1-8dB)
        target_snr_db: SNR mục tiêu sau khi augment (phải nhỏ hơn original_snr_db)
        """
        # 1. Nếu target cao hơn original, không làm gì cả (vì không thể làm sạch ảnh bằng cách cộng nhiễu)
        if target_snr_db >= original_snr_db:
            return spec

        # 2. Tính công suất tổng hiện tại (P_signal + P_noise_old)
        current_total_power = torch.mean(spec ** 2)

        # 3. Từ SNR_orig, tính tỷ lệ giữa Signal và Noise hiện tại
        # SNR = 10 * log10(Ps/Pn) => Ps/Pn = 10^(SNR/10)
        snr_orig_linear = 10 ** (original_snr_db / 10)

        # Giải hệ: Ps + Pn = current_total_power AND Ps/Pn = snr_orig_linear
        # => Pn_old = current_total_power / (snr_orig_linear + 1)
        # => Ps = current_total_power - Pn_old
        pn_old = current_total_power / (snr_orig_linear + 1)
        ps = current_total_power - pn_old

        # 4. Tính tổng công suất nhiễu cần có để đạt target_snr
        snr_target_linear = 10 ** (target_snr_db / 10)
        pn_new = ps / snr_target_linear

        # 5. Lượng nhiễu cần cộng thêm vào
        pn_added = pn_new - pn_old

        # Nếu vì lý do sai số mà pn_added < 0 thì dừng
        if pn_added <= 0:
            return spec

        # 6. Tạo nhiễu và cộng vào
        noise = torch.randn_like(spec) * torch.sqrt(pn_added)
        return spec + noise

    def __call__(self, sample):
        spec = sample["image"]
        metadata = sample["meta"]
        snr = metadata[0]
        doppler = metadata[1]

        # 1. Doppler shift (dựa trên metadata)
        shift = int(doppler)
        spec = torch.roll(spec, shift, dims=1)

        # 2. Random SNR augmentation
        reduction = random.uniform(0, 2)
        snr_aug = snr - reduction
        spec = self.add_controlled_awgn(spec, snr, snr_aug)

        # 3. SpecAugment
        spec = self.mask(spec)

        spec = torch.clamp(spec, 0, 1)

        sample["image"] = spec
        return sample

    def mask(self, spec):
        h, w = spec.shape[-2:]

        # time mask
        t = random.randint(0, w // 4)
        t0 = random.randint(0, w - t)
        spec[:, t0:t0+t] = 0

        # freq mask
        f = random.randint(0, h // 4)
        f0 = random.randint(0, h - f)
        spec[f0:f0+f, :] = 0

        return spec