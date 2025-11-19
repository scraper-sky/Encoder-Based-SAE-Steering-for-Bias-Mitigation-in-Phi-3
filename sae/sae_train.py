import argparse, numpy as np, torch as T
import torch.nn as nn, torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.E = nn.Linear(d_model, d_hidden, bias=False)
        self.D = nn.Linear(d_hidden, d_model, bias=False)
        nn.init.kaiming_uniform_(self.E.weight, a=0.0)
        nn.init.xavier_uniform_(self.D.weight)

    def forward(self, x):
        z = F.relu(self.E(x))
        x_hat = self.D(z)
        return x_hat, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="cache/acts/lm2_train.npy")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--mult", type=float, default=2.0)   # width multiplier
    ap.add_argument("--l1", type=float, default=3e-3)
    ap.add_argument("--out", default="cache/sae_lm2.pt")
    args = ap.parse_args()

    X = np.load(args.acts, mmap_mode="r")
    N, H = X.shape
    d_hidden = int(args.mult * H)
    dev = "cuda" if T.cuda.is_available() else ("mps" if T.backends.mps.is_available() else "cpu")
    model = SAE(H, d_hidden).to(dev)
    opt = T.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    idx = np.arange(N)
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, N, args.batch):
            batch = T.tensor(X[idx[i:i+args.batch]], dtype=T.float32, device=dev)
            x_hat, z = model(batch)
            recon = F.mse_loss(x_hat, batch)
            l1 = args.l1 * z.abs().mean()
            loss = recon + l1
            opt.zero_grad(); loss.backward(); opt.step()
            tot += recon.item() * len(batch)
        print(f"epoch {ep}: recon={tot/N:.6f}")

    T.save({"E": model.E.weight.detach().cpu(), "D": model.D.weight.detach().cpu()}, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
