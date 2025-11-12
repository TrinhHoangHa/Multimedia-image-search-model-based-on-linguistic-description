import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import clip  
from torch import nn, optim

class ImageTextDataset(Dataset):
    def __init__(self, images_dir, metadata_csv, transform=None,
                 filename_col='filename', caption_col='caption'):
        self.images_dir = images_dir

        df = pd.read_csv(
            metadata_csv,
            engine='python',
            on_bad_lines='warn'
        )

        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)

        if filename_col not in df.columns or caption_col not in df.columns:
            raise ValueError(f"metadata.csv must contain columns '{filename_col}' and '{caption_col}'")

        self.items = df[[filename_col, caption_col]].dropna().values.tolist()
        self.transform = transform
        print(f"Đã tải thành công {len(self.items)} cặp ảnh/caption từ {metadata_csv}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, caption = self.items[idx]
        
        is_valid_filename = str(fname).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        if not is_valid_filename:
            print("\n" + "="*60)
            print("!!! PHÁT HIỆN DỮ LIỆU GÂY LỖI !!!")
            print(f"Lỗi xảy ra tại index trong danh sách dữ liệu: {idx}")
            print(f"Tên file đọc được không hợp lệ: '{fname}'")
            print(f"Caption tương ứng: '{caption}'")
            print(">>> Vấn đề có thể nằm ở dòng tương ứng trong file CSV, hoặc ở dòng NGAY TRƯỚC ĐÓ.")
            print("="*60 + "\n")

        path = os.path.join(self.images_dir, str(fname))
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, caption
        except FileNotFoundError:
            print(f"\nLỖI NGHIÊM TRỌNG: Không thể tìm thấy file tại đường dẫn: {path}")
            print("Vui lòng đảm bảo tên file trong metadata.csv và tên file trong thư mục images khớp nhau.")
            raise

def collate_fn(batch, tokenizer, device):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    text_tokens = tokenizer(list(captions), truncate=True).to(device)
    return images, text_tokens

def make_collate_fn(tokenizer, device):
    def _fn(batch):
        return collate_fn(batch, tokenizer, device)
    return _fn

def build_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    for i, (images, text_tokens) in pbar:
        images = images.to(device)
        text_tokens = text_tokens.to(device)

        optimizer.zero_grad()
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        targets = torch.arange(images.shape[0], device=device)
        loss_i = ce(logits_per_image, targets)
        loss_t = ce(logits_per_text, targets)
        loss = (loss_i + loss_t) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits_per_image.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total_correct += correct
        total_samples += images.size(0)

        acc = total_correct / total_samples
        pbar.set_postfix(loss=total_loss / (i + 1), acc=acc)

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load CLIP
    model, _ = clip.load(args.model, device=device, jit=False)
    tokenizer = clip.tokenize
    image_size = model.visual.input_resolution
    transform = build_transforms(image_size)

    dataset = ImageTextDataset(args.images_dir, args.metadata,
                               transform=transform,
                               filename_col=args.filename_col,
                               caption_col=args.caption_col)

    collate = make_collate_fn(tokenizer, device)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=collate,
                            drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_acc = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"clip_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "clip_best.pt")
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"Saved best model to {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='./images')
    parser.add_argument('--metadata', type=str, default='metadata.csv')
    parser.add_argument('--filename_col', type=str, default='filename')
    parser.add_argument('--caption_col', type=str, default='caption')
    parser.add_argument('--model', type=str, default='ViT-B/32')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0)  
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)