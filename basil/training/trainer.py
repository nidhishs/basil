import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

# Top-level optional import to check availability
try:
    import wandb
except ImportError:
    wandb = None

from basil.config import BasilDataConfig, BasilModelConfig, BasilTrainConfig
from basil.training.dataset import get_dataset
from basil.training.model import RQVAE
from basil.utils import save_checkpoint, setup_device, setup_logging

logger = setup_logging(__name__)


class BasilTrainer:
    def __init__(
        self,
        model_cfg: BasilModelConfig,
        train_cfg: BasilTrainConfig,
        data_cfg: BasilDataConfig,
        output_dir: str,
    ):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.data_cfg = data_cfg
        self.out_root = Path(output_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

        # 1. Device Selection (Centralized logic)
        self.device = setup_device(train_cfg.device)
        logger.info(f"Training on device: {self.device}")

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        # 2. Seeding
        torch.manual_seed(train_cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(train_cfg.seed)

        # 3. Precision Setup
        # MPS is unstable with BF16 autocast. We force FP32 on Mac.
        # CUDA/CPU get BF16 if requested.
        self.use_amp = (train_cfg.precision == "bf16-mixed") and (
            self.device.type != "mps"
        )
        if train_cfg.precision == "bf16-mixed" and self.device.type == "mps":
            logger.warning("Mixed-precision disabled for MPS stability. Using FP32.")

        # 4. Logging Setup (Opt-in)
        self.wandb_active = False
        if train_cfg.project_name:
            if wandb is None:
                logger.warning(
                    "WandB project name set, but 'wandb' not installed. Logging disabled."
                )
            else:
                wandb.init(
                    project=train_cfg.project_name,
                    name=train_cfg.run_name,
                    config={"model": model_cfg, "train": train_cfg, "data": data_cfg},
                )
                self.wandb_active = True

    def setup(self):
        # --- Data ---
        dataset = get_dataset(self.data_cfg)
        # If streaming, data is pre-shuffled on disk. In-memory needs shuffle.
        should_shuffle = not self.data_cfg.stream

        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=should_shuffle,
            num_workers=self.data_cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            prefetch_factor=(
                self.data_cfg.prefetch_factor if self.data_cfg.num_workers > 0 else None
            ),
        )

        # --- Model & Optimization ---
        self.model = RQVAE(self.model_cfg).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )

        self.total_steps = len(self.train_loader) * self.train_cfg.epochs
        self.save_interval = max(1, int(self.total_steps * self.train_cfg.save_ratio))

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.train_cfg.lr,
            total_steps=self.total_steps,
            pct_start=self.train_cfg.warmup_ratio,
            anneal_strategy="cos",
        )

        if self.train_cfg.kmeans_init:
            self._run_kmeans()

    def train(self):
        self.setup()
        self.global_step = 0
        logger.info(f"Starting training. Total steps: {self.total_steps}")

        for epoch in range(self.train_cfg.epochs):
            self._train_epoch(epoch)

            if self.train_cfg.reset_unused_codes:
                self._reset_codes()

            if self.wandb_active:
                self._log_epoch_histograms(epoch)

            if self.global_step > 0:
                self._save_checkpoint(self.out_root / f"epoch-{epoch}")

        # Final Save (with ONNX export)
        self._save_checkpoint(self.out_root, final=True)
        logger.info(f"Training complete. Saved to {self.out_root}")

    def _train_epoch(self, epoch):
        self.model.train()
        accum_steps = self.train_cfg.gradient_accumulation_steps

        for i, batch in enumerate(self.train_loader):
            # 1. Move Data
            batch = batch.to(self.device, non_blocking=True)

            # 2. Mixed Precision Context
            # BF16 needs no GradScaler. FP16 is forbidden per config.
            ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.use_amp
                else nullcontext()
            )

            with ctx:
                x_recon, indices, metrics = self.model(batch)
                recon_loss = F.mse_loss(x_recon, batch)
                loss = recon_loss + metrics["vq_loss"]
                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            # 3. Backward (Accumulates gradients)
            loss.backward()

            # 4. Update Step (Conditional)
            is_update_step = ((i + 1) % accum_steps == 0) or (
                (i + 1) == len(self.train_loader)
            )
            if not is_update_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_cfg.gradient_clip_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

            # 5. Logging (Only on update steps)
            if self.global_step % self.train_cfg.log_interval == 0:
                # Rescale loss back for logging visibility
                logged_loss = loss.item() * accum_steps
                self._log_step(logged_loss, recon_loss, metrics, indices, batch.size(0))

            if self.global_step > 0 and self.global_step % self.save_interval == 0:
                self._save_checkpoint(self.out_root / f"checkpoint-{self.global_step}")

    def _log_step(self, loss_val, recon, metrics, indices, batch_size):
        unique_rows = len(torch.unique(indices, dim=0))
        utilization = unique_rows / batch_size
        lr = self.scheduler.get_last_lr()[0]

        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {loss_val:.4f} | "
            f"Ppl: {metrics['perplexity'].item():.1f} | "
            f"Util: {utilization:.2f}"
        )

        if self.wandb_active:
            wandb.log(
                {
                    "train/total_loss": loss_val,
                    "train/recon_loss": recon.item(),
                    "train/vq_loss": metrics["vq_loss"].item(),
                    "train/codebook_loss": metrics["codebook_loss"].item(),
                    "train/commitment_loss": metrics["commitment_loss"].item(),
                    "train/perplexity": metrics["perplexity"].item(),
                    "train/batch_utilization": utilization,
                    "train/lr": lr,
                },
                step=self.global_step,
            )

    def _log_epoch_histograms(self, epoch):
        if not self.wandb_active:
            return

        try:
            batch = next(iter(self.train_loader)).to(self.device)
        except StopIteration:
            return

        self.model.eval()
        with torch.no_grad():
            _, indices, _ = self.model(batch)
        self.model.train()

        hist_dict = {}
        indices_cpu = indices.cpu().numpy()
        for level in range(indices_cpu.shape[1]):
            hist_dict[f"health/codebook_usage_level_{level}"] = wandb.Histogram(
                indices_cpu[:, level]
            )

        wandb.log(hist_dict, step=self.global_step)

    def _run_kmeans(self):
        logger.info("Running KMeans Init...")
        try:
            # Load on CPU first to avoid GPU OOM if batch is huge
            batch = next(iter(self.train_loader)).cpu()
        except StopIteration:
            return

        with torch.no_grad():
            # Move to device for encoding
            residual = self.model.encoder(batch.to(self.device)).cpu()

            for layer in self.model.quantizer.codebooks:
                km = MiniBatchKMeans(
                    n_clusters=self.model_cfg.codebook_size, n_init="auto"
                )
                km.fit(residual.float().numpy())

                centroids = torch.from_numpy(km.cluster_centers_)
                layer.weight.data.copy_(centroids.to(self.device))

                dists = self.model.quantizer._compute_distances(
                    residual.to(self.device), layer.weight
                )
                idx = torch.argmin(dists, dim=-1)
                residual = residual - layer.weight[idx].cpu()

        logger.info("KMeans Init Complete.")

    def _reset_codes(self):
        try:
            batch = next(iter(self.train_loader)).to(self.device)
        except StopIteration:
            return

        with torch.no_grad():
            z = self.model.encoder(batch)
            self.model.quantizer.reset_dead_codes(z)

    def _save_checkpoint(self, target_path: Path, final: bool = False):
        save_checkpoint(
            self.model,
            self.model_cfg,
            self.train_cfg,
            self.data_cfg,
            target_path,
            export_onnx=final,
        )
