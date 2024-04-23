from .unet_parts import *
import torchvision

class UNet(nn.Module):
    def __init__(self, n_channels=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        bilinear = False
        self.n_slices = 12
        self.dim_embed = 128

        vgg = torchvision.models.vgg16_bn(pretrained=True)

        vgg_features = vgg.features
        self.down1 = vgg_features[:4]
        self.down2 = vgg_features[4:11]
        self.down3 = vgg_features[11:21]
        self.down4 = vgg_features[21:31]
        self.down5 = vgg_features[31:41]
        self.down5_ = vgg_features[41:44]

        self.trans_c = nn.Conv2d(512 + self.dim_embed, 512, 1)
        self.up1 = (Up(512, 256, bilinear))
        self.trans_up1 = nn.Conv2d(512, 256, 1)
        self.up2 = (Up(256, 128, bilinear))
        self.trans_up2 = nn.Conv2d(256, 128, 1)
        self.up3 = (Up(128, 64, bilinear))
        self.trans_up3 = nn.Conv2d(128, 64, 1)
        self.up4 = (Up(64, 32, bilinear))
        self.trans_up4 = nn.Conv2d(64, 32, 1)
        self.outc = (OutConv(32, 3))
        self.emds = torch.nn.Embedding(self.n_slices, self.dim_embed)


    def expand_bs(self, x):
        n_bs, n_c, n_w, n_h = x.shape
        x_tile = x.view(n_bs, 1, n_c, n_w, n_h).expand(-1, self.n_slices, -1, -1, -1).reshape(n_bs * self.n_slices, n_c, n_w, n_h)
        return x_tile

    def forward(self, x):
        feats = []

        x1 = self.down1(x) # 64, img_size, img_size
        x2 = self.down2(x1) # 128, img_size // 2, img_size // 2
        x3 = self.down3(x2) # 256, img_size // 4, img_size // 4
        x4 = self.down4(x3) # 512, img_size // 8, img_size // 8
        x5 = self.down5(x4) # 512, img_size // 16, img_size // 16
        x5_ = self.down5_(x5) # 512, img_size // 32, img_size // 32

        n_bs, n_c, n_w, n_h = x5.shape

        embs_tile = self.emds.weight.view(1, self.n_slices, self.dim_embed, 1, 1).expand(n_bs, self.n_slices, self.dim_embed, n_w, n_h).reshape(n_bs * self.n_slices, self.dim_embed, n_w, n_h)

        x5_tile = self.expand_bs(x5)

        latent = torch.cat([x5_tile, embs_tile], 1)
        latent = self.trans_c(latent)
        feats.append(latent)

        x = self.up1(latent, self.trans_up1(self.expand_bs(x4)))
        feats.append(x)
        x = self.up2(x, self.trans_up2(self.expand_bs(x3)))
        feats.append(x)
        x = self.up3(x, self.trans_up3(self.expand_bs(x2)))
        feats.append(x)
        x = self.up4(x, self.trans_up4(self.expand_bs(x1)))
        feats.append(x)
        out = self.outc(x)
        return feats, out
