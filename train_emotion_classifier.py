"""
=== MIA · Script de Entrenamiento del Clasificador de Emociones ===
Entrena el modelo completo (TextEmbedder + MLP) end-to-end
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from emotion_classifier_model import EmotionClassifier


# ==================== DATASET ====================
class EmotionDataset(Dataset):
    """
    Dataset personalizado para el clasificador de emociones
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: ruta al archivo JSON con el dataset
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.texts = [item['text'] for item in self.data['data']]
        self.labels = [item['label'] for item in self.data['data']]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    """
    Función de colación personalizada para el DataLoader
    """
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(texts), labels


# ==================== TRAINER ====================
class EmotionTrainer:
    """
    Clase para entrenar el clasificador de emociones
    """
    def __init__(
        self,
        model: EmotionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        device: torch.device = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (Adam)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Para tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Entrena una época completa
        Returns: (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for texts, labels in pbar:
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(texts)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Actualizar progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Valida el modelo
        Returns: (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in tqdm(self.val_loader, desc="Validation"):
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(texts)
                loss = self.criterion(logits, labels)
                
                # Métricas
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy
    
    def test(self) -> Dict:
        """
        Evalúa el modelo en el set de prueba
        Returns: diccionario con métricas detalladas
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in tqdm(self.test_loader, desc="Testing"):
                labels = labels.to(self.device)
                logits = self.model(texts)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Reporte de clasificación
        target_names = list(self.model.label_map.values())
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=target_names,
            output_dict=True
        )
        
        # Matriz de confusión
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(
        self,
        num_epochs: int = 30,
        early_stopping_patience: int = 5,
        save_path: str = "models/emotion_classifier"
    ):
        """
        Entrena el modelo con early stopping
        """
        print(f"\n{'='*60}")
        print(f"Iniciando entrenamiento por {num_epochs} épocas")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nÉpoca {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validar
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Imprimir resultados
            print(f"\nResultados época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                patience_counter = 0
                
                # Guardar modelo
                Path(save_path).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, f"{save_path}/best_model.pt")
                
                print(f"  ✓ Mejor modelo guardado (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping activado en época {epoch}")
                print(f"Mejor Val Loss: {self.best_val_loss:.4f}")
                print(f"Mejor Val Acc: {self.best_val_acc:.2f}%")
                print(f"{'='*60}")
                break
        
        print(f"\n{'='*60}")
        print("Entrenamiento completado!")
        print(f"Mejor Val Loss: {self.best_val_loss:.4f}")
        print(f"Mejor Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Cargar mejor modelo para testing
        checkpoint = torch.load(f"{save_path}/best_model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluar en test set
        print("\nEvaluando en el conjunto de prueba...")
        test_results = self.test()
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS EN TEST SET")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")
        
        # Guardar resultados
        self.plot_training_history(save_path)
        self.plot_confusion_matrix(test_results['confusion_matrix'], save_path)
        self.save_classification_report(test_results['classification_report'], save_path)
        
        return test_results
    
    def plot_training_history(self, save_path: str):
        """
        Grafica la historia del entrenamiento
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss durante el entrenamiento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy durante el entrenamiento')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica de entrenamiento guardada en: {save_path}/training_history.png")
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """
        Grafica la matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.model.label_map.values()),
            yticklabels=list(self.model.label_map.values())
        )
        plt.title('Matriz de Confusión - Test Set')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {save_path}/confusion_matrix.png")
        plt.close()
    
    def save_classification_report(self, report: Dict, save_path: str):
        """
        Guarda el reporte de clasificación
        """
        with open(f"{save_path}/classification_report.txt", 'w') as f:
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
            
            f.write(f"\nACCURACY: {report['accuracy']:.4f}\n")
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
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    
    print("="*60)
    print("CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("="*60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Épocas máximas: {NUM_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print("="*60)
    
    # Crear datasets
    print("\nCargando datasets...")
    train_dataset = EmotionDataset(TRAIN_PATH)
    val_dataset = EmotionDataset(VAL_PATH)
    test_dataset = EmotionDataset(TEST_PATH)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Crear modelo
    print("\nInicializando modelo...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(
        model_name="dccuchile/bert-base-spanish-wwm-cased",
        emb_dim=300,
        max_length=128,
        hidden1=128,
        hidden2=64,
        num_classes=6,
        dropout=0.3,
        device=device
    )
    
    print(f"Modelo en: {device}")
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Crear trainer
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        device=device
    )
    
    # Entrenar
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