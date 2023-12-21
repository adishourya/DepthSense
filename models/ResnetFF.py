import torch
class ResnetFF(nn.Module):
    def __init__(self, encoder_name='resnext50_32x4d', in_channels=3, classes=1):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels)
        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32),
            n_blocks=(3, 2, 1, 0),
            use_batchnorm=True,
            center=False,
            attention_type=None,  # You can change this to 'scse' or 'cbam' for attention mechanisms
        )
        self.segmentation_head = smp.unet.model.UnetHead(
            in_channels=self.decoder.out_channels[-1], out_channels=classes, activation=None
        )

    def trainable_encoder(self, trainable=True):
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        logits = self.segmentation_head(decoder_output[-1])
        return logits

    def _num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

