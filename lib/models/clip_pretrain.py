import pdb

import torch
import torch.nn as nn
from transformers import CLIPPreTrainedModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


class CLIP_visual_encoder(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    def __init__(self, config: CLIPVisionConfig, num_class=100, freeze_backbone=False, freeze_cls=False):
        super().__init__(config)
        print(config)
        self.vision_model = CLIPVisionTransformer(config)
        self.num_class = num_class
        self.freeze_backbone = freeze_backbone
        self.freeze_cls = freeze_cls

        self.fc = nn.Linear(config.hidden_size, 512)
        self.classifier = nn.Linear(512, self.num_class)

        if self.freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        else:
            for param in self.vision_model.parameters():
                param.requires_grad = True

        if self.freeze_cls:
            for param in self.classifier.parameters():
                param.requires_grad = False
        else:
            for param in self.classifier.parameters():
                param.requires_grad = True


    def forward(self, x):
        x = self.vision_model(pixel_values=x, output_hidden_states=False).pooler_output  # x: [bs, 768]
        # x = x[:, 0, :]
        x = self.fc(x)
        out = self.classifier(x)
        return out


def build_clip_visual_encoder(args=None, freeze_backbone=False, freeze_cls=False):

    # model = CLIP_visual_encoder.from_pretrained('./clip_vit_base_patch16', num_class=100, freeze_backbone=freeze_backbone)
    model = CLIP_visual_encoder.from_pretrained(args.network, num_class=100, freeze_backbone=freeze_backbone, freeze_cls=freeze_cls)
    return model


def build_clip_visual_encoder_decoder(args, freeze_backbone):
    pass


if __name__ == "__main__":
    model = build_clip_visual_encoder(freeze_backbone=True)
    x = torch.rand(8, 3, 224, 224)
    y = model(x)
    pdb.set_trace()
    print(y)
#   y['logits']: [bs, 100], y['features']: [bs, 768]
