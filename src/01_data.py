"""
Dataset indexing, raster loading/caching, normalization, and the PyTorch Dataset class.

Original notebook cell(s): [3, 4, 5]
Depends on (run first / shares globals with): 00_config.py
"""

# ----- (notebook cell 3) -----
def foldertodf(sitedir: Path, nositedir: Path) -> pd.DataFrame:
    rows = [{'filepath': str(f), 'label': l}
            for l, d in [(1, sitedir), (0, nositedir)]
            for f in sorted(d.glob(f'*{CFG.IMGSUFFIX}'))]
    return pd.DataFrame(rows)

print('Scanning dataset folders...')
traindf = foldertodf(CFG.TRAINSITE, CFG.TRAINNOSITE)
valdf   = foldertodf(CFG.VALSITE,   CFG.VALNOSITE)
testdf  = foldertodf(CFG.TESTSITE,  CFG.TESTNOSITE)

print(f"Train: {len(traindf)} | Val: {len(valdf)} | Test: {len(testdf)}")
print(f"Train class balance: {traindf['label'].value_counts().to_dict()}")

# ----- (notebook cell 4) -----
def loadrawpatch(tifpath: str):
    with rasterio.open(tifpath) as src:
        img = src.read(masked=True).astype(np.float32).filled(0.0)
    sat = img[CFG.SATBANDS] / CFG.SAT_SCALE
    sat = np.clip(sat, 0.0, 1.0)
    return sat, img[CFG.LIDARBANDS]


def buildrawcache(df: pd.DataFrame, splitname: str):
    print(f"Caching {len(df)} {splitname} patches into RAM...")
    sats, lidars, labels = [], [], []
    for row in df.itertuples():
        s, l = loadrawpatch(row.filepath)
        sats.append(s); lidars.append(l); labels.append(float(row.label))
    return sats, lidars, labels


def computelidarstats(lidars):
    n = len(CFG.LIDARBANDS)
    total = np.zeros(n, np.float64)
    totalsq = np.zeros(n, np.float64)
    count = 0
    for l in lidars:
        ld = l.astype(np.float64)
        total   += ld.sum(axis=(1, 2))
        totalsq += (ld ** 2).sum(axis=(1, 2))
        count   += l.shape[1] * l.shape[2]
    mean = total / count
    var = np.maximum(totalsq / count - mean ** 2, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def normalizelidarperpatch(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for c, ch in enumerate(arr):
        mean, std = float(np.nanmean(ch)), float(np.nanstd(ch))
        out[c] = ch - mean if std < 1e-4 or np.isnan(std) else (ch - mean) / (std + 1e-6)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def finalizecache(sats, lidars, labels, lmean, lstd):
    out = []
    for i in range(len(sats)):
        l = lidars[i]
        if CFG.LIDARNORM == 'global':
            ln = (l - lmean[:, None, None]) / (lstd[:, None, None] + 1e-6)
            ln = np.clip(ln, -CFG.LIDARCLIP, CFG.LIDARCLIP)
        else:
            ln = normalizelidarperpatch(l)
        ln = np.nan_to_num(ln, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out.append((np.concatenate([sats[i], ln], axis=0), labels[i]))
        sats[i] = None; lidars[i] = None
    return out

# ----- (notebook cell 5) -----
class MultiModalDataset(Dataset):
    def __init__(self, cached, istrain: bool = True):
        self.cached = cached
        self.istrain = istrain
        self.randomerase = RandomErasing(p=CFG.ERASEPROB, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0)
        self.randomcrop = RandomResizedCrop(CFG.IMGSIZE, scale=CFG.CROPSCALE, ratio=(0.85, 1.18), antialias=True)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        img, label = self.cached[idx]
        t = torch.from_numpy(img).clone()
        nsat = len(CFG.SATBANDS)

        if CFG.IMAGENET_NORM and nsat >= 3:
            t[:3] = (t[:3] - self.rgb_mean) / self.rgb_std

        if self.istrain:
            t = self.randomcrop(t)
            if random.random() > 0.5: t = TF.hflip(t)
            if random.random() > 0.5: t = TF.vflip(t)
            k = random.choice([0, 1, 2, 3])
            if k: t = torch.rot90(t, k, dims=(1, 2))

            if random.random() > 0.5:
                t[:nsat] += torch.randn_like(t[:nsat]) * CFG.SATNOISESTD
                t[nsat:] += torch.randn_like(t[nsat:]) * CFG.LIDARNOISESTD

            t = self.randomerase(t)

            if random.random() < CFG.RGBDROPPROB:
                satmean = t[0:nsat].mean(dim=(1, 2), keepdim=True)
                t[0:nsat] = satmean.expand_as(t[0:nsat])
            if random.random() < CFG.LIDARDROPPROB:
                t[nsat:] = 0.0
        else:
            if t.shape[-1] != CFG.IMGSIZE or t.shape[-2] != CFG.IMGSIZE:
                t = TF.resize(t, [CFG.IMGSIZE, CFG.IMGSIZE],
                              interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        return t, torch.tensor(label, dtype=torch.float32)
