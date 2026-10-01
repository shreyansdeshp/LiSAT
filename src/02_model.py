"""
ConvNeXt-Tiny backbone adapted to an 11-channel (RGB + LiDAR) input.

Original notebook cell(s): [8]
Depends on (run first / shares globals with): 00_config.py
"""

# ----- (notebook cell 8) -----
def createconvnextmodel(inchannels=11, numclasses=1, dropout=CFG.DROPOUT):
    model = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT,
        stochastic_depth_prob=CFG.STOCHDEPTH,
    )

    oldconv = model.features[0][0]
    oldinchannels  = oldconv.weight.shape[1]
    oldoutchannels = oldconv.out_channels

    newconv = nn.Conv2d(
        in_channels=inchannels,
        out_channels=oldoutchannels,
        kernel_size=oldconv.kernel_size,
        stride=oldconv.stride,
        padding=oldconv.padding,
        bias=oldconv.bias is not None,
    )

    with torch.no_grad():
        if inchannels == oldinchannels:
            newconv.weight[:] = oldconv.weight
        elif oldinchannels == 3 and inchannels > 3:
            newconv.weight[:, 0:3, :, :] = oldconv.weight
            meanweight = oldconv.weight.mean(dim=1)
            for c in range(3, inchannels):
                newconv.weight[:, c, :, :] = meanweight * 0.25
            newconv.weight[:, 3:inchannels, :, :] *= (oldinchannels / inchannels)
            newconv.weight[:, 3:inchannels, :, :] += torch.randn_like(
                newconv.weight[:, 3:inchannels, :, :]) * CFG.STEMNOISESTD
        else:
            nn.init.kaiming_normal_(newconv.weight, nonlinearity='relu')

        if oldconv.bias is not None:
            newconv.bias.copy_(oldconv.bias)

    model.features[0][0] = newconv
    infeatures = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(infeatures, numclasses),
    )
    return model
