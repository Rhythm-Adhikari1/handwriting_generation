import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image
import torchvision
from tqdm import tqdm
from data_loader.loader import ContentData
import torch.distributed as dist
import torch.nn.functional as F
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader, 
                logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None, drive_backup_dir=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.ocr_model = ocr_model
        self.ctc_criterion = ctc_loss
        self.device = device
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.SOLVER.EPOCHS, eta_min=1e-6)

        # Google Drive backup directory
        self.drive_backup_dir = drive_backup_dir
        if self.drive_backup_dir:
            os.makedirs(self.drive_backup_dir, exist_ok=True)
            if dist.get_rank() == 0:
                print(f"✅ Drive backup enabled: {self.drive_backup_dir}")
      
    def _train_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device)
        
        # vae encode
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)
        
       
        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')


        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + 0.4 * high_nce_loss + 0.4 * low_nce_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid, target, target_lengths = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device), \
            data['target'].to(self.device), \
            data['target_lengths'].to(self.device)
        
        # vae encode
        latent_images = self.vae.encode(images).latent_dist.sample()
        latent_images = latent_images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)
        
        x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(self.model, x_t, style_ref, laplace_ref,
                                                        content_ref, t, sampling_timesteps=10)
 
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        rec_out = self.ocr_model(x_start)
        input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]])
        ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss + 0.1*ctc_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item(), "ctc_loss": ctc_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)
            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save(path)
        return im
    
    # for visual check
    @torch.no_grad()
    def _valid_iter(self, epoch):
        print('loading test dataset, the number is', len(self.valid_data_loader))
        self.model.eval()
        # use the first batch of dataloader in all validations for better visualization comparisons
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)
        # prepare input
        images, style_ref, laplace_ref, content_ref = test_data['img'].to(self.device), \
            test_data['style'].to(self.device), \
            test_data['laplace'].to(self.device), \
            test_data['content'].to(self.device)
    
        load_content = ContentData()
        # forward
        texts = ['संगीत', 'विश' , 'पछि', 'ज्यान' , 'काम']
        for text in texts:
            rank = dist.get_rank()
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(self.device).repeat(style_ref.shape[0], 1, 1, 1)
            x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (text_ref.shape[1]*32)//8)).to(self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, images.shape[0], x, style_ref, laplace_ref, text_ref)
            out_path = os.path.join(self.save_sample_dir, f"epoch-{epoch}-{text}-process-{rank}.png")
            self._save_images(preds, out_path)

    def _get_latest_checkpoint(self, check_drive_first=True):
        """
        Find the most recent checkpoint from Drive (priority) or local directory
        Returns: (checkpoint_path, epoch_number) or (None, 0) if no checkpoint found
        """
        latest_checkpoint = None
        latest_epoch = 0
        
        # Check Drive backup first if enabled
        if check_drive_first and self.drive_backup_dir and os.path.exists(self.drive_backup_dir):
            print(f"🔍 Searching for checkpoints in Drive: {self.drive_backup_dir}")
            checkpoint_files = []
            
            try:
                for filename in os.listdir(self.drive_backup_dir):
                    if filename.endswith('-ckpt.pt'):
                        # Extract epoch number from filename (e.g., "50-ckpt.pt" -> 50)
                        try:
                            epoch = int(filename.split('-')[0])
                            filepath = os.path.join(self.drive_backup_dir, filename)
                            checkpoint_files.append((filepath, epoch))
                        except ValueError:
                            continue
                
                if checkpoint_files:
                    # Sort by epoch and get the latest
                    checkpoint_files.sort(key=lambda x: x[1])
                    latest_checkpoint, latest_epoch = checkpoint_files[-1]
                    print(f"✅ Found latest checkpoint in Drive: {os.path.basename(latest_checkpoint)} (Epoch {latest_epoch})")
                    return latest_checkpoint, latest_epoch
            except Exception as e:
                print(f"⚠️  Could not access Drive: {e}")
        
        # Fallback to local directory
        print(f"🔍 Searching for checkpoints in local dir: {self.save_model_dir}")
        checkpoint_files = []
        
        try:
            for filename in os.listdir(self.save_model_dir):
                if filename.endswith('-ckpt.pt'):
                    try:
                        epoch = int(filename.split('-')[0])
                        filepath = os.path.join(self.save_model_dir, filename)
                        checkpoint_files.append((filepath, epoch))
                    except ValueError:
                        continue
            
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: x[1])
                latest_checkpoint, latest_epoch = checkpoint_files[-1]
                print(f"✅ Found latest checkpoint locally: {os.path.basename(latest_checkpoint)} (Epoch {latest_epoch})")
                return latest_checkpoint, latest_epoch
        except Exception as e:
            print(f"⚠️  Could not access local directory: {e}")
        
        print("❌ No checkpoints found")
        return None, 0

    def _load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint
        """
        try:
            print(f"📥 Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.module.load_state_dict(state_dict)
            print(f"✅ Successfully loaded checkpoint!")
            return True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False

    def train(self, resume=False, start_epoch=0):
        """
        Start training iterations
        Args:
            resume: If True, attempt to load latest checkpoint
            start_epoch: Starting epoch (used when resuming)
        """
        # Resume from checkpoint if requested
        if resume:
            checkpoint_path, loaded_epoch = self._get_latest_checkpoint(check_drive_first=True)
            if checkpoint_path:
                if self._load_checkpoint(checkpoint_path):
                    start_epoch = loaded_epoch + 1  # Continue from next epoch
                    print(f"\n{'='*70}")
                    print(f"🔄 RESUMING TRAINING from Epoch {start_epoch}")
                    print(f"{'='*70}\n")
                else:
                    print(f"⚠️  Failed to load checkpoint, starting from scratch")
                    start_epoch = 0
            else:
                print(f"⚠️  No checkpoint found, starting from scratch")
                start_epoch = 0
        
        # Training loop
        for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
            self.data_loader.sampler.set_epoch(epoch)
            print(f"Epoch:{epoch} of process {dist.get_rank()}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False)
            else:
                pbar = self.data_loader

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        if dist.get_rank() == 0:
                            self._save_checkpoint(total_step)
                    else:
                        pass
                    if self.valid_data_loader is not None:
                        if (total_step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (total_step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                        else:
                            pass 
                else:
                    self._train_iter(data, total_step, pbar)


            self.scheduler.step()

            # Log learning rate to TensorBoard
            if dist.get_rank() == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tb_summary.add_scalar('learning_rate', current_lr, epoch)
                
            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                if dist.get_rank() == 0:
                    self._save_checkpoint(epoch)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    self._valid_iter(epoch)
            else:
                pass

            if dist.get_rank() == 0:
                pbar.close()

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch):
        """
        Save checkpoint to both local directory and Google Drive
        Keeps only the 3 most recent checkpoints in Drive
        """
        checkpoint_filename = str(epoch) + '-' + "ckpt.pt"
        
        # Save to local directory (Colab temporary storage)
        local_path = os.path.join(self.save_model_dir, checkpoint_filename)
        torch.save(self.model.module.state_dict(), local_path)
        
        # Get file size for logging
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        
        print(f"\n{'='*70}")
        print(f"💾 CHECKPOINT SAVED - Epoch {epoch}")
        print(f"{'='*70}")
        print(f"📁 Local:  {local_path}")
        print(f"📊 Size:   {file_size_mb:.2f} MB")
        
        # Copy to Google Drive with checkpoint rotation
        if self.drive_backup_dir:
            try:
                drive_path = os.path.join(self.drive_backup_dir, checkpoint_filename)
                
                print(f"☁️  Copying to Drive...")
                shutil.copy2(local_path, drive_path)
                
                print(f"✅ DRIVE:  {drive_path}")
                
                # Keep only the 3 most recent checkpoints
                self._manage_drive_checkpoints(max_checkpoints=3)
                
                print(f"{'='*70}\n")
                
            except Exception as e:
                print(f"❌ ERROR: Failed to copy to Drive: {e}")
                print(f"⚠️  Checkpoint only saved locally at: {local_path}")
                print(f"{'='*70}\n")
        else:
            print(f"⚠️  No Drive backup configured")
            print(f"{'='*70}\n")

    def _manage_drive_checkpoints(self, max_checkpoints=3):
        """
        Keep only the most recent N checkpoints in Drive backup directory
        Deletes older checkpoints when limit is exceeded
        """
        try:
            # Get all checkpoint files in Drive backup directory
            checkpoint_files = []
            for filename in os.listdir(self.drive_backup_dir):
                if filename.endswith('-ckpt.pt'):
                    filepath = os.path.join(self.drive_backup_dir, filename)
                    # Get file creation/modification time
                    mtime = os.path.getmtime(filepath)
                    checkpoint_files.append((filepath, mtime, filename))
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=lambda x: x[1])
            
            # Delete oldest checkpoints if we exceed the limit
            num_to_delete = len(checkpoint_files) - max_checkpoints
            if num_to_delete > 0:
                print(f"🗑️  Removing {num_to_delete} old checkpoint(s) from Drive...")
                for i in range(num_to_delete):
                    filepath, _, filename = checkpoint_files[i]
                    os.remove(filepath)
                    print(f"   Deleted: {filename}")
                
                # Show remaining checkpoints
                print(f"📦 Keeping {max_checkpoints} most recent checkpoints:")
                for i in range(num_to_delete, len(checkpoint_files)):
                    _, _, filename = checkpoint_files[i]
                    print(f"   ✓ {filename}")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not manage old checkpoints: {e}")