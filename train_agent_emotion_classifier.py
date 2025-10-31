"""
=== MIA · Script de Entrenamiento del AgentEmotionPredictClassifier (v2) ===
- Lee JSON v2 con `label` (usuario) y `label_agent` (objetivo del agente).
- Soporta encoder preentrenado (BETO) con fine-tuning controlado.
- Class weights + label smoothing para desbalance.
- Early stopping por macro-F1.
- AdamW con dos grupos de LR (encoder vs head) + scheduler lineal con warmup.
- Clip de gradientes, dump de misclasificados, matrices de confusión (absoluta y normalizada).

Requisitos:
  pip install transformers scikit-learn seaborn matplotlib tqdm
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from agent_emotion_predict_classifier import AgentEmotionPredictClassifier

# ==================== DATASET ====================
class AgentEmotionDataset(Dataset):
    def __init__(self, data_path: str, label_map: Optional[dict] = None):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        data = raw.get('data', raw)
        self.texts: List[str] = []
        self.user_labels: List[int] = []
        self.agent_labels: List[int] = []
        for it in data:
            self.texts.append(it['text'])
            u = it['label']; a = it['label_agent']
            u = int(u) if isinstance(u, str) else u
            a = int(a) if isinstance(a, str) else a
            if label_map is not None:
                a = label_map[a]  # remapea a {0..K-1}
            self.user_labels.append(u)
            self.agent_labels.append(a)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.user_labels[idx], self.agent_labels[idx]

def collate_fn(batch):
    texts, ulabels, alabels = zip(*batch)
    ulabels = torch.tensor(ulabels, dtype=torch.long)
    alabels = torch.tensor(alabels, dtype=torch.long)
    return list(texts), ulabels, alabels


# ==================== TRAINER ====================
class AgentEmotionTrainer:
    def __init__(
        self,
        model: AgentEmotionPredictClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        lr_encoder: float = 2e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        num_epochs: int = 20,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 3,
        warmup_freeze_epochs: int = 2,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = class_names or ["alegría","amor"]

        # ---------- Pérdida con pesos de clase + label smoothing ----------
        labels_tensor = torch.tensor(train_loader.dataset.agent_labels)
        class_counts = torch.bincount(labels_tensor, minlength=self.num_classes).float()
        safe_counts = class_counts.clamp(min=1.0)
        inv_freq = (safe_counts.sum() / (self.num_classes * safe_counts)).to(self.device)
        class_weights = inv_freq / inv_freq.mean()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # ---------- Optimizador AdamW con 2 grupos (encoder vs head) ----------
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        # Warmup: encoder congelado n épocas
        self.warmup_freeze_epochs = warmup_freeze_epochs
        self.model.freeze_encoder()

        self.optimizer = self._build_optimizer()

        # ---------- Scheduler lineal con warmup ----------
        self.num_epochs = num_epochs
        total_steps = len(self.train_loader) * self.num_epochs
        from transformers import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        # ---------- Tracking ----------
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.val_f1s = []
        self.best_val_f1 = 0.0
        self.best_state = None
        self.early_stopping_patience = early_stopping_patience

    def _build_optimizer(self):
        encoder_params, head_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "embedder.encoder" in n:  # parámetros del Transformer
                encoder_params.append(p)
            else:
                head_params.append(p)
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_encoder, "weight_decay": self.weight_decay},
                {"params": head_params,   "lr": self.lr_head,    "weight_decay": 0.0},
            ]
        )

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training")
        for texts, ulabels, alabels in pbar:
            alabels = alabels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(texts, ulabels.to(self.device))
            loss = self.criterion(logits, alabels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == alabels).sum().item()
            total += alabels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for texts, ulabels, alabels in tqdm(self.val_loader, desc="Validation"):
                alabels = alabels.to(self.device)
                logits = self.model(texts, ulabels.to(self.device))
                loss = self.criterion(logits, alabels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == alabels).sum().item()
                total += alabels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(alabels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, macro_f1

    def test(self, save_dir: Path) -> Dict:
        self.model.eval()
        all_predictions, all_labels, all_texts = [], [], []
        with torch.no_grad():
            for texts, ulabels, alabels in tqdm(self.test_loader, desc="Testing"):
                logits = self.model(texts, ulabels.to(self.device))
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(alabels.numpy())
                all_texts.extend(texts)

        accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
        labels_ord = list(range(self.num_classes))
        target_names = self.class_names

        report = classification_report(
            all_labels,
            all_predictions,
            labels=labels_ord,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        cm_abs = confusion_matrix(all_labels, all_predictions, labels=labels_ord)
        cm_norm = confusion_matrix(all_labels, all_predictions, labels=labels_ord, normalize='true')

        self._dump_misclassified(all_texts, all_labels, all_predictions, target_names, save_dir / "misclassified.txt")
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix_abs': cm_abs,
            'confusion_matrix_norm': cm_norm,
        }

    @staticmethod
    def _dump_misclassified(texts, y_true, y_pred, names, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for t, yt, yp in zip(texts, y_true, y_pred):
                if yt != yp:
                    f.write(f"[gold={names[yt]} | pred={names[yp]}] {t}\n")

    def train(self, num_epochs: Optional[int] = None, early_stopping_patience: Optional[int] = None, save_dir: str = "models/agent_emotion"):
        num_epochs = num_epochs or self.num_epochs
        if early_stopping_patience is not None:
            self.early_stopping_patience = early_stopping_patience

        print(f"\n{'='*60}\nIniciando entrenamiento por {num_epochs} épocas\nDevice: {self.device}\n{'='*60}")
        patience_counter = 0
        best_f1 = -1.0

        for epoch in range(1, num_epochs + 1):
            # Unfreeze encoder después del warmup
            if epoch == self.warmup_freeze_epochs + 1:
                print("→ Descongelando encoder para fine-tuning...")
                self.model.unfreeze_encoder()
                self.optimizer = self._build_optimizer()

            print(f"\nÉpoca {epoch}/{num_epochs}\n" + '-'*60)
            tr_loss, tr_acc = self.train_epoch()
            self.train_losses.append(tr_loss); self.train_accs.append(tr_acc)
            va_loss, va_acc, va_f1 = self.validate()
            self.val_losses.append(va_loss); self.val_accs.append(va_acc); self.val_f1s.append(va_f1)

            print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.2f}% | Val: loss={va_loss:.4f} acc={va_acc:.2f}% f1m={va_f1:.4f}")

            # Guardar mejor por Macro-F1
            if va_f1 > best_f1:
                best_f1 = va_f1
                patience_counter = 0
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': tr_loss,
                    'val_loss': va_loss,
                    'val_acc': va_acc,
                    'val_f1': va_f1,
                }, f"{save_dir}/best_model.pt")
                print(f"  ✓ Mejor modelo guardado (Val Macro-F1: {va_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")
            if patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping activado en época {epoch}")
                break

        print(f"\n{'='*60}\nEntrenamiento completado! Mejor Val Macro-F1: {best_f1:.4f}\n{'='*60}")

        # Evaluar en test con mejor checkpoint
        ckpt = torch.load(f"{save_dir}/best_model.pt", map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        test_results = self.test(Path(save_dir))

        print(f"\n{'='*60}\nRESULTADOS EN TEST SET\n{'='*60}")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")

        # Graficar y guardar reportes
        self.plot_training_history(save_dir)
        self.plot_confusion_matrix(test_results['confusion_matrix_abs'], save_dir, norm=False)
        self.plot_confusion_matrix(test_results['confusion_matrix_norm'], save_dir, norm=True)
        self.save_classification_report(test_results['classification_report'], save_dir)
        return test_results

    # ---------- utilidades de guardado/plot ----------
    def plot_training_history(self, save_dir: str):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Época'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc', marker='o')
        plt.plot(self.val_accs, label='Val Acc', marker='s')
        plt.plot(self.val_f1s, label='Val Macro-F1', marker='^')
        plt.xlabel('Época'); plt.ylabel('Score'); plt.title('Acc / Macro-F1'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_dir}/training_history.png", dpi=300, bbox_inches='tight'); plt.close()
        print(f"✓ Gráfica de entrenamiento guardada en: {save_dir}/training_history.png")

    def plot_confusion_matrix(self, cm: np.ndarray, save_dir: str, norm: bool = False):
        plt.figure(figsize=(8, 6))
        fmt = '.2f' if norm else 'd'
        cmap = 'Blues'
        ticklabels = self.class_names
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=ticklabels, yticklabels=ticklabels,
                    vmin=0, vmax=1 if norm else None)
        plt.title('Matriz de Confusión ' + ('(Normalizada)' if norm else '(Absoluta)'))
        plt.ylabel('Etiqueta Real'); plt.xlabel('Etiqueta Predicha')
        plt.tight_layout(); Path(save_dir).mkdir(parents=True, exist_ok=True)
        fname = "confusion_matrix_norm.png" if norm else "confusion_matrix_abs.png"
        plt.savefig(f"{save_dir}/{fname}", dpi=300, bbox_inches='tight'); plt.close()
        print(f"✓ Matriz de confusión guardada en: {save_dir}/{fname}")

    def save_classification_report(self, report: Dict, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/classification_report.txt", 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("REPORTE DE CLASIFICACIÓN - TEST SET\n")
            f.write("="*60 + "\n\n")
            for emotion, metrics in report.items():
                if emotion in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                f.write(f"\n{emotion.upper()}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
                f.write(f"  Support:   {metrics['support']}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"MACRO AVG:\n  Precision: {report['macro avg']['precision']:.4f}\n  Recall:    {report['macro avg']['recall']:.4f}\n  F1-Score:  {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"\nWEIGHTED AVG:\n  Precision: {report['weighted avg']['precision']:.4f}\n  Recall:    {report['weighted avg']['recall']:.4f}\n  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
            acc = report.get('accuracy', None)
            if acc is not None:
                f.write(f"\nACCURACY: {acc:.4f}\n")
            f.write("="*60 + "\n")
        print(f"✓ Reporte de clasificación guardado en: {save_dir}/classification_report.txt")


# ==================== MAIN ====================
def main():
    # Rutas (usa los *v2.json*)
    DATA_DIR = "models/emotion_classifier/data"
    TRAIN_PATH = f"{DATA_DIR}/emotion_dataset_train_es_v2.json"
    VAL_PATH   = f"{DATA_DIR}/emotion_dataset_validation_es_v2.json"
    TEST_PATH  = f"{DATA_DIR}/emotion_dataset_test_es_v2.json"
    SAVE_DIR   = "models/agent_emotion"

    # Hiperparámetros
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3
    WARMUP_FREEZE_EPOCHS = 2
    LR_ENCODER = 2e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    print("="*60)
    print("CONFIGURACIÓN DEL ENTRENAMIENTO (AgentEmotion v2)")
    print("="*60)
    print(f"Train: {TRAIN_PATH}")
    print(f"Val:   {VAL_PATH}")
    print(f"Test:  {TEST_PATH}")
    print(f"Batch size: {BATCH_SIZE} | Épocas: {NUM_EPOCHS}")
    print(f"Freeze warmup epochs: {WARMUP_FREEZE_EPOCHS}")
    print(f"LR encoder: {LR_ENCODER} | LR head: {LR_HEAD}")

    # 1) Detectar clases presentes en TRAIN y remapear a {0..K-1}
    probe = AgentEmotionDataset(TRAIN_PATH)
    present_classes = sorted(set(probe.agent_labels))  # p.ej., [1,2]
    label_names_full = {0:"tristeza",1:"alegría",2:"amor",3:"ira",4:"miedo",5:"sorpresa"}
    class_names = [label_names_full[c] for c in present_classes]
    K = len(present_classes)
    label_map = {orig:i for i, orig in enumerate(present_classes)}
    print(f"Clases de agente en train: {present_classes} → K={K} ({class_names})")

    # 2) Recrear datasets aplicando el mapeo
    train_ds = AgentEmotionDataset(TRAIN_PATH, label_map=label_map)
    val_ds   = AgentEmotionDataset(VAL_PATH,   label_map=label_map)
    test_ds  = AgentEmotionDataset(TEST_PATH,  label_map=label_map)

    # 3) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4) Modelo (salida con K clases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgentEmotionPredictClassifier(
        pretrained_encoder="beto",
        max_length=128,
        hidden1=256,
        hidden2=64,
        dropout=0.2,
        label_feature_dropout=0.15,
        device=device,
        num_classes=K,
    )
    model.freeze_encoder()

    # 5) Trainer con num_classes y nombres
    trainer = AgentEmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr_encoder=LR_ENCODER,
        lr_head=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        warmup_freeze_epochs=WARMUP_FREEZE_EPOCHS,
        num_classes=K,
        class_names=class_names,
    )

    # 6) Entrenar y evaluar
    trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    main()
