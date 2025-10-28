"""
=== MIA · Script de Entrenamiento del Clasificador de Emociones (v2) ===
- Soporta encoder preentrenado (BETO) con fine-tuning controlado.
- Class weights + label smoothing para desbalance.
- Early stopping por macro-F1.
- AdamW con dos grupos de LR (encoder vs head) + scheduler lineal con warmup.
- Clip de gradientes, misclassified dump, matrices de confusión (absoluta y normalizada).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

from emotion_classifier_model import EmotionClassifier

# ==================== DATASET ====================
class EmotionDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.texts = [item['text'] for item in self.data['data']]
        self.labels = [item['label'] for item in self.data['data']]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(texts), labels


# ==================== TRAINER ====================
class EmotionTrainer:
    def __init__(
        self,
        model: EmotionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        lr_encoder: float = 2e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        num_epochs: int = 30,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 3,
        warmup_freeze_epochs: int = 2,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- Pérdida con pesos de clase + label smoothing ----------
        labels_tensor = torch.tensor(train_loader.dataset.labels)
        num_classes = self.model.classifier.fc3.out_features
        class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
        class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # ---------- Optimizador AdamW con 2 grupos (encoder vs head) ----------
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        self.weight_decay = weight_decay

        # Al inicio: congelamos el encoder por warmup_freeze_epochs
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
            # Heurística: parámetros del encoder BETO contienen "embedder.encoder"
            if "embedder.encoder" in n:
                encoder_params.append(p)
            else:
                head_params.append(p)

        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_encoder, "weight_decay": self.weight_decay},
                {"params": head_params, "lr": self.lr_head, "weight_decay": 0.0},
            ]
        )

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training")

        for texts, labels in pbar:
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(texts)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for texts, labels in tqdm(self.val_loader, desc="Validation"):
                labels = labels.to(self.device)
                logits = self.model(texts)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        return avg_loss, accuracy, macro_f1

    def test(self, save_path: str) -> Dict:
        self.model.eval()
        all_predictions, all_labels = [], []
        all_texts = []

        with torch.no_grad():
            for texts, labels in tqdm(self.test_loader, desc="Testing"):
                labels = labels.to(self.device)
                logits = self.model(texts)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_texts.extend(texts)

        accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))

        labels_ord = list(self.model.label_map.keys())
        target_names = [self.model.label_map[i] for i in labels_ord]

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

        # Guardar misclasificados
        self._dump_misclassified(all_texts, all_labels, all_predictions, self.model.label_map, Path(save_path) / "misclassified.txt")

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix_abs': cm_abs,
            'confusion_matrix_norm': cm_norm,
            'predictions': all_predictions,
            'labels': all_labels
        }

    @staticmethod
    def _dump_misclassified(texts, y_true, y_pred, label_map, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for t, yt, yp in zip(texts, y_true, y_pred):
                if yt != yp:
                    f.write(f"[gold={label_map[yt]} | pred={label_map[yp]}] {t}\n")

    def train(
        self,
        num_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        save_path: str = "models/emotion_classifier"
    ):
        num_epochs = num_epochs or self.num_epochs
        if early_stopping_patience is not None:
            self.early_stopping_patience = early_stopping_patience

        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento por {num_epochs} épocas")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Unfreeze encoder después del warmup de congelamiento
            if epoch == self.warmup_freeze_epochs + 1:
                print("→ Descongelando encoder para fine-tuning...")
                self.model.unfreeze_encoder()
                self.optimizer = self._build_optimizer()  # re-construye para incluir encoder entrenable

            print(f"\nÉpoca {epoch}/{num_epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss); self.train_accs.append(train_acc)

            val_loss, val_acc, val_f1 = self.validate()
            self.val_losses.append(val_loss); self.val_accs.append(val_acc); self.val_f1s.append(val_f1)

            print(f"\nResultados época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val   Loss: {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val Macro-F1: {val_f1:.4f}")

            # Guardar mejor por Macro-F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                patience_counter = 0
                Path(save_path).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, f"{save_path}/best_model.pt")
                print(f"  ✓ Mejor modelo guardado (Val Macro-F1: {val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

            if patience_counter >= self.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping activado en época {epoch}")
                print(f"Mejor Val Macro-F1: {self.best_val_f1:.4f}")
                print(f"{'='*60}")
                break

        print(f"\n{'='*60}")
        print("Entrenamiento completado!")
        print(f"Mejor Val Macro-F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")

        # Cargar mejor modelo y evaluar en test
        checkpoint = torch.load(f"{save_path}/best_model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print("\nEvaluando en el conjunto de prueba...")
        test_results = self.test(save_path=save_path)

        print(f"\n{'='*60}")
        print(f"RESULTADOS EN TEST SET")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")

        # Guardar visualizaciones y reportes
        self.plot_training_history(save_path)
        self.plot_confusion_matrix(test_results['confusion_matrix_abs'], save_path, norm=False)
        self.plot_confusion_matrix(test_results['confusion_matrix_norm'], save_path, norm=True)
        self.save_classification_report(test_results['classification_report'], save_path)

        return test_results

    def plot_training_history(self, save_path: str):
        plt.figure(figsize=(12, 5))
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Época'); plt.ylabel('Loss')
        plt.title('Loss durante el entrenamiento'); plt.legend(); plt.grid(True, alpha=0.3)

        # Accuracy y Macro-F1 (dos ejes)
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc', marker='o')
        plt.plot(self.val_accs, label='Val Acc', marker='s')
        plt.plot(self.val_f1s, label='Val Macro-F1', marker='^')
        plt.xlabel('Época'); plt.ylabel('Score')
        plt.title('Accuracy / Macro-F1 durante el entrenamiento'); plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica de entrenamiento guardada en: {save_path}/training_history.png")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str, norm: bool = False):
        plt.figure(figsize=(10, 8))
        fmt = '.2f' if norm else 'd'
        cmap = 'Blues'
        labels_ord = list(self.model.label_map.keys())
        ticklabels = [self.model.label_map[i] for i in labels_ord]
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=ticklabels, yticklabels=ticklabels, vmin=0, vmax=1 if norm else None)
        plt.title('Matriz de Confusión ' + ('(Normalizada)' if norm else '(Absoluta)'))
        plt.ylabel('Etiqueta Real'); plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        Path(save_path).mkdir(parents=True, exist_ok=True)
        fname = "confusion_matrix_norm.png" if norm else "confusion_matrix_abs.png"
        plt.savefig(f"{save_path}/{fname}", dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {save_path}/{fname}")
        plt.close()

    def save_classification_report(self, report: Dict, save_path: str):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(f"{save_path}/classification_report.txt", 'w', encoding='utf-8') as f:
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
            f.write(f"MACRO AVG:\n")
            f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall:    {report['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"\nWEIGHTED AVG:\n")
            f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
            f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
            acc = report.get('accuracy', None)
            if acc is not None:
                f.write(f"\nACCURACY: {acc:.4f}\n")
            f.write("="*60 + "\n")
        print(f"✓ Reporte de clasificación guardado en: {save_path}/classification_report.txt")


# ==================== MAIN ====================
def main():
    # Configuración
    DATA_DIR = "models/emotion_classifier/data"
    TRAIN_PATH = f"{DATA_DIR}/emotion_dataset_train_es.json"
    VAL_PATH = f"{DATA_DIR}/emotion_dataset_validation_es.json"
    TEST_PATH = f"{DATA_DIR}/emotion_dataset_test_es.json"
    SAVE_PATH = "models/emotion_classifier"

    # Hiperparámetros
    BATCH_SIZE = 16  # recomendado para BETO; sube a 32 si la GPU lo permite
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3
    WARMUP_FREEZE_EPOCHS = 2
    LR_ENCODER = 2e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    print("="*60)
    print("CONFIGURACIÓN DEL ENTRENAMIENTO (v2)")
    print("="*60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Épocas máximas: {NUM_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Freeze warmup epochs: {WARMUP_FREEZE_EPOCHS}")
    print(f"LR encoder: {LR_ENCODER} | LR head: {LR_HEAD}")
    print("="*60)

    # Cargar datasets
    print("\nCargando datasets...")
    train_dataset = EmotionDataset(TRAIN_PATH)
    val_dataset = EmotionDataset(VAL_PATH)
    test_dataset = EmotionDataset(TEST_PATH)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Modelo con encoder preentrenado (BETO)
    print("\nInicializando modelo (BETO + MLP)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        emb_dim=300,           # Ignorado si usamos pretrained_encoder="beto"
        max_length=128,
        hidden1=128,
        hidden2=64,
        num_classes=6,
        dropout=0.3,
        device=device,
        pretrained_encoder="beto"  # <-- activar encoder preentrenado
    )

    print(f"Modelo en: {device}")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parámetros entrenables (inicio, encoder congelado): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Trainer
    trainer = EmotionTrainer(
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
    )

    # Entrenar (guardado por Macro-F1)
    test_results = trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_path=SAVE_PATH
    )

    print("\n" + "="*60)
    print("¡Entrenamiento finalizado exitosamente!")
    print("="*60)


if __name__ == "__main__":
    main()