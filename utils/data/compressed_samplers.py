import random

from torch.utils.data import BatchSampler


class VideoBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle_variants=False, clip_repeat=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants
        self.clip_repeat = clip_repeat

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        all_batches = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            pool = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        pool.append((v['name'], variant, f, v['ds_root'], yi, xi))
            random.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)

        for _ in range(self.clip_repeat):
            random.shuffle(all_batches)
            yield from all_batches

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size) * self.clip_repeat


class ValVideoBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        for v in self.dataset.videos:
            variant = v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            pool = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        pool.append((v['name'], variant, f, v['ds_root'], yi, xi))
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                yield chunk

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size)


class SequentialVideoBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle_variants=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        video_batch_lists = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            video_id = hash(f"{v['name']}_{v['ds_root']}_{variant}") & 0x7FFFFFFF

            items = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        items.append((v['name'], variant, f, v['ds_root'], yi, xi, video_id))

            batches = []
            for i in range(0, len(items), self.batch_size):
                chunk = items[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
            if batches:
                video_batch_lists.append(batches)

        random.shuffle(video_batch_lists)
        for batches in video_batch_lists:
            yield from batches

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size)
