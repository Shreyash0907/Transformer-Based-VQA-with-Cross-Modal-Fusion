import json
import os
import torch.optim as optim
import random
import torchvision.models as models
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from transformers import BertModel
from torchvision.models import ResNet101_Weights
import argparse

warnings.filterwarnings("ignore")

emb_dim = 768
n_heads = 8
n_layers = 6
mx_len = 32
l_rte = 1.5e-4
b_sz = 64
n_epocs = 40
w_decay = 1.5e-5
sch_patience = 4
sch_factor = 0.2
dset_path = ""
savepth = ""
mdl_pth_arg = None

class Data(Dataset):
    def __init__(self, img_dir, q_path, tknzer, ans_to_idx, mx_len, transfrm, is_trn=False, sv_path_vocab=None):
        self.img_dir = img_dir
        self.tknzer = tknzer

        self.mx_len = mx_len
        self.ans_to_idx = ans_to_idx

        if ans_to_idx:
            rev_map = {}
            for key, val in ans_to_idx.items():
                rev_map[val] = key
            self.idx_to_ans = rev_map
        else:
            self.idx_to_ans = None

        self.transfrm = transfrm

        self.is_trn = is_trn
        self.sv_path_vocab = sv_path_vocab

        with open(q_path, 'r') as f:
            data = json.load(f)
        self.qns = data['questions']

        if is_trn and not ans_to_idx:

            all_ans_list = []
            for i in self.qns:
                answer_from_q = i['answer']
                all_ans_list.append(answer_from_q)
            all_ans = all_ans_list 

            uniq_ans = sorted(list(set(all_ans)))

            atoi_map, i = {}, 0
            for j in uniq_ans:
                atoi_map[j] = i
                i = i + 1
            self.ans_to_idx = atoi_map

            self.idx_to_ans = {idx: ans for ans, idx in self.ans_to_idx.items()}

            itoa = {}
            for ans_key, i in self.ans_to_idx.items():
                itoa[i] = ans_key
            self.idx_to_ans = itoa

            if self.sv_path_vocab:
                vocab_path = os.path.join(self.sv_path_vocab, "answer_vocab.json")
                os.makedirs(self.sv_path_vocab, exist_ok=True)
                with open(vocab_path, 'w') as f:
                    json.dump(self.ans_to_idx, f)

    def __len__(self):
        return len(self.qns)
    
    def __getitem__(self, idx):
        
        qn_data = self.qns[idx]
        img = self._load_image(qn_data['image_filename'])
        if img is None:
            return None, None, None, None

        qn_txt = qn_data['question']
        ans_txt = qn_data['answer']
        encding = self.tknzer.encode_plus(qn_txt, add_special_tokens=True, max_length=self.mx_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        inpt_ids = encding['input_ids'].squeeze(0)
        attn_mask = encding['attention_mask'].squeeze(0)

        ans_idx = -1

        if self.ans_to_idx:
            default_idx_if_not_found = -1
            if not self.is_trn:
                default_idx_if_not_found = 0
            
            if ans_txt in self.ans_to_idx:
                ans_idx = self.ans_to_idx[ans_txt]
            else:
                ans_idx = default_idx_if_not_found
            
            if ans_idx == -1 and self.is_trn:
                ans_idx = 0
        else:
            pass

        if ans_idx == -1 and self.is_trn:
            ans_idx = 0

        return img, inpt_ids, attn_mask, torch.tensor(ans_idx, dtype=torch.long)
    
    def _load_image(self, filename):
        img_path = os.path.join(self.img_dir, filename)
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert('RGB')
        return self.transfrm(img)


    


class Vqa(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers, n_classes, mx_len, tknzer_pad_id, require_grad=False, bert_model=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.mx_len = mx_len + 1

        self.image_encoder_base = self.init_image_encoder(require_grad)
        self.image_proj = self.init_image_projection(emb_dim)
        self.text_embedding = self.init_text_embedding(vocab_size, emb_dim, tknzer_pad_id, bert_model)
        self.cls_token = self.init_cls_token(emb_dim)
        self.pos_embedding = self.init_pos_embedding(self.mx_len, emb_dim)
        self.transformer_encoder = self.init_transformer_encoder(emb_dim, n_heads, n_layers)
        self.cross_attention = self.init_cross_attention(emb_dim, n_heads)
        self.classifier = self.init_classifier(emb_dim, n_classes)

    def init_image_encoder(self, require_grad):
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        encoder = nn.Sequential(*list(resnet.children())[:-2])
        for param in encoder.parameters():
            param.requires_grad = require_grad
        return encoder

    def init_classifier(self, emb_dim, n_classes):
        return nn.Sequential(nn.Linear(emb_dim, 500),nn.ReLU(),nn.Dropout(0.5),nn.Linear(500, n_classes))

    def init_image_projection(self, emb_dim):
        return nn.Linear(2048, emb_dim)

    def init_text_embedding(self, vocab_size, emb_dim, padding_idx, bert_model):
        embedding =  nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        if bert_model is not None:
            print("Initializing text embeddings with BERT weights...")
            embedding.weight.data.copy_(bert_model.embeddings.word_embeddings.weight.data)
        return embedding

    def init_cls_token(self, emb_dim):
        return nn.Parameter(torch.zeros(1, 1, emb_dim))

    def init_pos_embedding(self, max_len, emb_dim):
        return nn.Parameter(torch.randn(1, max_len, emb_dim))

    def init_transformer_encoder(self, emb_dim, n_heads, n_layers):
        transformer_enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4, dropout=0.1, activation='relu', batch_first=True)
        return nn.TransformerEncoder(transformer_enc_layer, num_layers=n_layers)

    def init_cross_attention(self, emb_dim, n_heads):
        return nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=0.1, batch_first=True)

    
    def forward(self, images, input_ids, attention_mask):
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        cls_representation = text_features[:, 0, :]
        fused_features = self.cross_modal_attention(cls_representation, image_features)
        output_logits = self.classifier(fused_features)
        return output_logits

    def encode_image(self, images):
        base_features = self.image_encoder_base(images)
        batch_size, channels, height, width = base_features.shape
        reshaped = base_features.view(batch_size, channels, height * width).permute(0, 2, 1)
        projected_features = self.image_proj(reshaped)
        return projected_features

    def encode_text(self, input_ids, attention_mask):
        padding_mask = (attention_mask == 0)
        batch_size = input_ids.size(0)
        token_embeddings = self.text_embedding(input_ids)
        cls_tokens_expanded = self.cls_token.expand(batch_size, 1, self.emb_dim)
        token_embeddings = torch.cat((cls_tokens_expanded, token_embeddings), dim=1)

        seq_length = token_embeddings.size(1)
        positionally_encoded = token_embeddings + self.pos_embedding[:, :seq_length, :]

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)
        combined_mask = torch.cat((cls_mask, padding_mask), dim=1)

        encoded_output = self.transformer_encoder(positionally_encoded, src_key_padding_mask=combined_mask)
        return encoded_output

    def cross_modal_attention(self, query_features, image_features):
        query = torch.unsqueeze(query_features, 1)
        attended_output, _ = self.cross_attention(query=query, key=image_features, value=image_features)
        return attended_output.squeeze(1)

def loader(base_path, tknzer, b_sz, mx_len, transfrm, sv_path_vocab):
    def get_data_paths(base_path, split_name):
        img_dir = os.path.join(base_path, "images", f"{split_name}A")
        q_path = os.path.join(base_path, "questions", f"CLEVR_{split_name}A_questions.json")
        return img_dir, q_path

    trn_img_dir, trn_q_path = get_data_paths(base_path, "train")
    val_img_dir, val_q_path = get_data_paths(base_path, "val")
    tst_img_dir, tst_q_path = get_data_paths(base_path, "test")

    ans_vocab_path = os.path.join(sv_path_vocab, "answer_vocab.json")
    ans_to_idx_loaded = None
    if os.path.exists(ans_vocab_path):
        try:
            with open(ans_vocab_path, 'r') as f:
                ans_to_idx_loaded = json.load(f)
        except Exception as e:
            pass
            
    trn_dset = Data(trn_img_dir, trn_q_path, tknzer, ans_to_idx_loaded, mx_len, transfrm, is_trn=True, sv_path_vocab=sv_path_vocab)
    ans_to_idx = trn_dset.ans_to_idx
    idx_to_ans = trn_dset.idx_to_ans

    if not ans_to_idx:
        sys.exit(1)

    val_dset = Data(val_img_dir, val_q_path, tknzer, ans_to_idx, mx_len, transfrm)
    tst_dset = Data(tst_img_dir, tst_q_path, tknzer, ans_to_idx, mx_len, transfrm)

    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    trn_ldr = DataLoader(trn_dset, batch_size=b_sz, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_ldr = DataLoader(val_dset, batch_size=b_sz, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    tst_ldr = DataLoader(tst_dset, batch_size=b_sz, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    
    return trn_ldr, val_ldr, tst_ldr, ans_to_idx, idx_to_ans

def train_ekbaar(mdl, dtlodr, critrn, optmzr, device):
    mdl.train()
    ttl_loss, all_preds, all_labels = 0, [], []

    
    pbar = tqdm(dtlodr, desc="Training Epoch", leave=False, file=sys.stdout)
    for batch in pbar:
        if batch[0] is None:
            continue

        # Unpack and move to device
        try:
            imgs, inpt_ids, attn_mask, lbls = [
                tensor.to(device) for tensor in batch if tensor is not None
            ]
        except ValueError:
            continue  # Skip malformed batch

        if imgs is None:
            continue

        optmzr.zero_grad()
        outputs = mdl(imgs, inpt_ids, attn_mask)
        loss = critrn(outputs, lbls)

        loss.backward()
        optmzr.step()

        ttl_loss = ttl_loss + loss.item()
        preds = torch.argmax(outputs, dim=1)
        predcpu = preds.cpu().numpy()
        if predcpu is not None or len(predcpu) >= 0: 
            all_preds.extend(predcpu)
        alllabel = lbls.cpu().numpy()
        if alllabel is not None or len(alllabel) >= 0:
            all_labels.extend(alllabel)

        pbar.set_postfix(loss=loss.item())


    if len(dtlodr) > 0:
        avg_loss = ttl_loss / len(dtlodr)
    else:
        avg_loss = 0

    if all_labels:
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        accuracy = 0


    return avg_loss, accuracy

def evaluate(mdl, dtlodr, critrn, device):
    mdl.eval()
    ttl_loss, all_preds, all_labels = 0, [], []

    
    pbar = tqdm(dtlodr, desc="Evaluating", leave=False, file=sys.stdout)
    with torch.no_grad():
        for btch in pbar:
            if btch[0] is None:
                continue
            tensors_on_device = []

            for b in btch:
                if b is not None:
                    tensors_on_device.append(b.to(device))

            imgs, inpt_ids, attn_mask, lbls = tensors_on_device
            if imgs is None:
                continue

            output = mdl(imgs, inpt_ids, attn_mask)
            loss = critrn(output, lbls)

            ttl_loss = ttl_loss + loss.item()
            preds = torch.argmax(output, dim=1)

            preds_on_cpu = preds.cpu()
            preds_as_numpy = preds_on_cpu.numpy()
            all_preds.extend(preds_as_numpy)
            
            lbls_on_cpu = lbls.cpu()
            lbls_as_numpy = lbls_on_cpu.numpy()
            all_labels.extend(lbls_as_numpy)


    if len(dtlodr) > 0:
        avg_loss = ttl_loss / len(dtlodr)
    else:
        avg_loss = 0

    if all_labels: # Checks if the list all_labels is not empty
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        accuracy = 0

    if all_labels:
        metrics_result = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        precision = metrics_result[0]
        recall = metrics_result[1]
        f1 = metrics_result[2]
        
        _ = metrics_result[3] 
    else:
        # If there are no labels, set metrics to default values
        precision = 0
        recall = 0
        f1 = 0
        _ = None # Match the structure of the tuple assignment


    return avg_loss, accuracy, precision, recall, f1



def vis_pred(mdl, dset, idx_to_ans, device, n_samples=5, find_errs=False, sv_dir=None, part_name="part8"):
    mdl.eval()
    samps_shown = 0
    
    print(f"\nVisualizing {'Error' if find_errs else 'Correct'} Predictions for {part_name}...")
    indices = list(range(len(dset)))
    random.shuffle(indices)

    fig_height_per_sample = 4 
    plt.figure(figsize=(15, fig_height_per_sample * n_samples))
    
    plot_idx = 1
    actual_samps_plotted = 0


    def extract_question_data(qn_data):
        qn_txt = qn_data['question']
        gt_ans_txt = qn_data['answer']
        img_fname = qn_data['image_filename']
        return qn_txt, gt_ans_txt, img_fname

    def get_image_path(img_dir, img_fname):
        return os.path.join(img_dir, img_fname)

    def prepare_batches(img_tensor, inpt_ids, attn_mask, device):
        img_tensor_batch = img_tensor.unsqueeze(0).to(device)
        inpt_ids_batch = inpt_ids.unsqueeze(0).to(device)
        attn_mask_batch = attn_mask.unsqueeze(0).to(device)
        return img_tensor_batch, inpt_ids_batch, attn_mask_batch

    def get_model_output(mdl, img_tensor_batch, inpt_ids_batch, attn_mask_batch):
        return mdl(img_tensor_batch, inpt_ids_batch, attn_mask_batch)

    def get_predicted_index(output):
        return torch.argmax(output, dim=1).item()

    def map_pred_to_answer(pred_idx, idx_to_ans):
        return idx_to_ans.get(pred_idx, "UNK_PRED")

    def check_correctness(pred_idx, lbl_idx):
        return pred_idx == lbl_idx.item()

    with torch.no_grad():
        for i in indices:
            if actual_samps_plotted >= n_samples:
                break

            itm = dset[i]
            if itm[0] is None: continue
            img_tensor, inpt_ids, attn_mask, lbl_idx = itm
            
            qn_data = dset.qns[i]
            qn_txt, gt_ans_txt, img_fname = extract_question_data(qn_data)
            img_path = get_image_path(dset.img_dir, img_fname)
            img_tensor_batch, inpt_ids_batch, attn_mask_batch = prepare_batches(img_tensor, inpt_ids, attn_mask, device)

            output = get_model_output(mdl, img_tensor_batch, inpt_ids_batch, attn_mask_batch)
            pred_idx = get_predicted_index(output)
            pred_ans_txt = map_pred_to_answer(pred_idx, idx_to_ans)
            is_correct = check_correctness(pred_idx, lbl_idx)


            if (find_errs and not is_correct) or (not find_errs and is_correct):
                try:
                    original_img = Image.open(img_path).convert('RGB')
                    ax = plt.subplot(n_samples, 1, plot_idx)
                    ax.imshow(original_img)
                    ax.set_title(f"Q: {qn_txt}\nGT: {gt_ans_txt} | Pred: {pred_ans_txt}", fontsize=10)
                    ax.axis('off')
                    plot_idx = plot_idx + 1
                    actual_samps_plotted = actual_samps_plotted + 1
                except FileNotFoundError:
                    continue
        
    if actual_samps_plotted > 0:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if find_errs:
            status_type = "errors"
        else:
            status_type = "correct"
        filename = f"visualize_predictions_{part_name}_{status_type}.png"
        vis_fname = os.path.join(sv_dir, filename)
        plt.savefig(vis_fname)
        print(f"Visualize predictions saved to {vis_fname}")
    else:
        print(f"No {'error' if find_errs else 'correct'} samples found to visualize.")
    plt.close()

def plot(trn_losses, val_losses, trn_accs, val_accs, sv_dir):
    plt.figure(figsize=(12, 5))
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 1)
    epocs = range(1, len(trn_losses) + 1)
    plt.plot(epocs, val_losses, 'ro-', label='Validation Loss')
    plt.plot(epocs, trn_losses, 'bo-', label='Training Loss')
    
    plot_fname = os.path.join(sv_dir, "training_curves_part8.png")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.title('Training and Validation Accuracy')
    plt.plot(epocs, trn_accs, 'bo-', label='Training Accuracy')
    plt.plot(epocs, val_accs, 'ro-', label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(plot_fname)
    print(f"Training curves saved to {plot_fname}")
    plt.close()

def save_model_checkpoint(epoch, mdl, optmzr, scheduler, best_val_accuracy, ans_to_idx, 
                          trn_losses, val_losses, trn_accs, val_accs, best_model_path):
    
    

    torch.save({
        'epoch': epoch + 1,
        'val_accs': val_accs,
        'train_accs': trn_accs,
        'val_losses': val_losses,
        'train_losses': trn_losses,
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'answer_to_index': ans_to_idx,
        'model_state_dict': mdl.state_dict(),
        'optimizer_state_dict': optmzr.state_dict(),
    }, best_model_path)

    print(f"New best model saved with Val Acc: {best_val_accuracy:.4f} at {best_model_path}")

def train_model_main(device, tknzer, img_transfrm):
    global savepth, dset_path, mdl_pth_arg

    print("Starting part 10b: Training")
    print(f"Dataset Base Path: {dset_path}")
    print(f"Save Path: {savepth}")
    if mdl_pth_arg:
        print(f"Resuming training from: {mdl_pth_arg}")

    trn_ldr, val_ldr, tst_ldr, ans_to_idx, idx_to_ans = loader(dset_path, tknzer, b_sz, mx_len, img_transfrm, savepth)
    n_classes = len(ans_to_idx)
    vocab_size = tknzer.vocab_size
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    mdl = Vqa(emb_dim=emb_dim,n_classes=n_classes, vocab_size=vocab_size,mx_len=mx_len, n_heads=n_heads, n_layers=n_layers,  tknzer_pad_id=tknzer.pad_token_id, bert_model=bert_model).to(device)

    critrn = nn.CrossEntropyLoss()
    optmzr = optim.Adam(mdl.parameters(), lr=l_rte, weight_decay=w_decay)
    scheduler = ReduceLROnPlateau(optmzr, mode='min', factor=sch_factor, patience=sch_patience, verbose=True)

    start_epoc = 0
    best_val_accuracy = 0.0
    best_epoc = 0
    trn_losses, val_losses, trn_accs, val_accs = [], [], [], []

    if mdl_pth_arg and os.path.exists(mdl_pth_arg):
        print(f"Loading checkpoint from {mdl_pth_arg}...")
        checkpoint = torch.load(mdl_pth_arg, map_location=device)
        mdl.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
             optmzr.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoc = checkpoint.get('epoch', 0)
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        trn_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        trn_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])

        print(f"Resuming from epoch {start_epoc + 1}, best_val_accuracy: {best_val_accuracy:.4f}")
    best_mdl_path = os.path.join(savepth, "best_model_part8.pth")
    print("Starting training loop...")
    for epoc in range(start_epoc, n_epocs):
        print(f"\nEpoch {epoc + 1}/{n_epocs}")
        trn_loss, trn_acc = train_ekbaar(mdl, trn_ldr, critrn, optmzr, device)
        val_loss, val_acc, precision, recall, f1 = evaluate(mdl, val_ldr, critrn, device)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoc = epoc + 1
            save_model_checkpoint(epoc, mdl, optmzr, scheduler, best_val_accuracy, ans_to_idx, 
                              trn_losses, val_losses, trn_accs, val_accs, best_mdl_path)

            print(f"New best model saved with Val Acc: {best_val_accuracy:.4f} at {best_mdl_path}")

        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
        trn_accs.append(trn_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoc + 1}: Train Loss={trn_loss:.4f}, Train Acc={trn_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step(val_loss)

        
    
    print(f"\nTraining finished. Best validation accuracy ({best_val_accuracy:.4f}) achieved at epoch {best_epoc if best_epoc > 0 else 'N/A'}.")
    plot(trn_losses, val_losses, trn_accs, val_accs, savepth)

    print("\nLoading best model for final evaluation on testA")
    

    if os.path.exists(best_mdl_path):
        checkpoint = torch.load(best_mdl_path, map_location=device)
        loaded_ans_to_idx = checkpoint['answer_to_index']
        mdl.load_state_dict(checkpoint['model_state_dict'])
        loaded_idx_to_ans = {v: k for k,v in loaded_ans_to_idx.items()}
        print(f"Best model from epoch {checkpoint['epoch']} loaded for test evaluation.")

        _, _, tst_ldr_final, _, _ = loader(dset_path, tknzer, b_sz, mx_len, img_transfrm, savepth)

        tst_loss, tst_acc, tst_prec, tst_recall, tst_f1 = evaluate(mdl, tst_ldr_final, critrn, device)
        print("\nTest Set A Results (part 10b)")
        metrics = {
            "Accuracy": tst_acc,
            "Precision": tst_prec,
            "Recall": tst_recall,
            "F1-score": tst_f1,
            "Loss": tst_loss
        }
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        tst_dset_for_viz = tst_ldr_final.dataset
        vis_pred(mdl, tst_dset_for_viz, loaded_idx_to_ans, device, n_samples=5, find_errs=False, sv_dir=savepth, part_name="part8_testA")
        vis_pred(mdl, tst_dset_for_viz, loaded_idx_to_ans, device, n_samples=5, find_errs=True, sv_dir=savepth, part_name="part8_testA")

    else:
        print("Best model checkpoint not found. Skipping test evaluation and visualization.")

    print("\npart 10b training script finished")

def inference_model_main(device, tknzer, img_transfrm):

    global dset_path, mdl_pth_arg
    print("Starting part 10b: Inference")

    if not mdl_pth_arg or not os.path.exists(mdl_pth_arg):
        print(f"Error: Model path '{mdl_pth_arg}' not found or not specified for inference.")
        sys.exit(1)

    print(f"Loading model from: {mdl_pth_arg}")
    checkpoint = torch.load(mdl_pth_arg, map_location=device)
    
    ans_to_idx = checkpoint['answer_to_index']

    d = {}
    for key, val in ans_to_idx.items():
        d[val] = key
    idx_to_ans = d

    n_classes = len(ans_to_idx)
    vocab_size = tknzer.vocab_size

    mdl = Vqa(vocab_size=vocab_size, emb_dim=emb_dim, n_heads=n_heads, n_layers=n_layers, n_classes=n_classes, mx_len=mx_len, tknzer_pad_id=tknzer.pad_token_id).to(device)
    mdl.load_state_dict(checkpoint['model_state_dict'])
    mdl.eval()
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}.")

    imagedirectory = os.path.join(dset_path, "images", "testA")
    questionpath = os.path.join(dset_path, "questions", "CLEVR_testA_questions.json")
    test_data = Data(imagedirectory, questionpath, tknzer, ans_to_idx, mx_len, img_transfrm)
    print("Inference from: ",questionpath)

    def collate_fn_inf(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: return None, None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    tst_ldr = DataLoader(test_data, batch_size=b_sz, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn_inf)
    
    critrn = nn.CrossEntropyLoss()
    tst_loss, tst_acc, tst_prec, tst_recall, tst_f1 = evaluate(mdl, tst_ldr, critrn, device)

    print("\nTest Set A Results (part 10b - Inference Mode)")
    metrics = {
        "Accuracy": tst_acc,
        "Precision": tst_prec,
        "Recall": tst_recall,
        "F1-score": tst_f1,
        "Loss": tst_loss
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    viz_sv_dir = "/kaggle/working/"
    os.makedirs(viz_sv_dir, exist_ok=True)

    vis_pred(mdl, test_data, idx_to_ans, device, n_samples=5, find_errs=False, sv_dir=viz_sv_dir, part_name="part8_inference_testA")
    vis_pred(mdl, test_data, idx_to_ans, device, n_samples=5, find_errs=True, sv_dir=viz_sv_dir, part_name="part8_inference_testA")
    
    print("\npart 10b inference script finished")

def main():
    parser = argparse.ArgumentParser(description="VQA Model part 10b: Training and Inference")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='Mode: train or inference')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the CLEVR dataset directory')
    parser.add_argument('--save_path', type=str, help='Path to save models, plots, vocab (required for training)')
    parser.add_argument('--model_path', type=str, help='Path to a pre-trained model checkpoint (for inference or resuming training)')
    
    args = parser.parse_args()

    global dset_path, savepth, mdl_pth_arg
    dset_path = args.dataset
    mdl_pth_arg = args.model_path

    if args.mode == 'train':
        if not args.save_path:
            print("Error: --save_path is required for training mode.")
            sys.exit(1)
        savepth = args.save_path
        os.makedirs(savepth, exist_ok=True)
        
    elif args.mode == 'inference':
        if not args.model_path:
            print("Error: --model_path is required for inference mode.")
            sys.exit(1)
        savepth = args.save_path if args.save_path else os.path.dirname(args.model_path)
        if not savepth: savepth = "."
        os.makedirs(savepth, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tknzer = BertTokenizer.from_pretrained("bert-base-uncased")
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Ensure internet or pre-downloaded 'bert-base-uncased'.")
        sys.exit(1)

    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_transfrm = transforms.Compose([
        resize,
        to_tensor,
        normalize
    ])
    if args.mode == 'train':
        train_model_main(device, tknzer, img_transfrm)
    elif args.mode == 'inference':
        inference_model_main(device, tknzer, img_transfrm)

if __name__ == '__main__':
    main()
