import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet50: bỏ fc, lấy feature
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modules = list(resnet.children())[:-1]
cnn_extractor = nn.Sequential(*modules).to(device)
cnn_extractor.eval()

# BERT base uncased
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Projection layers
image_projection = nn.Linear(2048, 256).to(device)
text_projection = nn.Linear(768, 256).to(device)

# Transform ảnh
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
