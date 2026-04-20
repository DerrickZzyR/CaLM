import torch
import torch.nn as nn
import open_clip

class OpenClipTextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            cache_dir='checkpoints/open_clip_cache',
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = self.tokenizer(["test"]).to(device)
            self.embed_dim = int(self.model.encode_text(dummy).shape[-1])

    def encode_tokens(self, text: str) -> torch.Tensor:
        """Return token-level [1, 77, 512] embeddings."""
        tok = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            x = self.model.token_embedding(tok)
            x = x + self.model.positional_embedding[:x.shape[1]]
            x = x.permute(1, 0, 2)
            x = self.model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.model.ln_final(x)
            x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return x.cpu()