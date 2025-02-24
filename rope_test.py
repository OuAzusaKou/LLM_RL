def ROPE():
    def __init__(self,hidden_dim,max_length,base):
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.rope_cache = {}
        self.inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim))
        self.trange = torch.arange(max_length)
        self.freqs = torch.outer(self.trange, self.inv_freq)
        self.emb = torch.cat([self.freqs, self.freqs], dim=-1)
        self.cos_cached = self.emb.cos()
        self.sin_cached = self.emb.sin()

    def half_rope(self,x,seq_len):
        half_seq_len = seq_len // 2
        x1 = x[:, :half_seq_len]
        x2 = x[:, half_seq_len:]
        
        return torch.cat([-x2_embed,x1_embed], dim=-1)

def apply_rope(x,rope,seq_len):
    return x*rope.cos_cached[seq_len] + self.half_rope(x,seq_len)*rope.sin_cached[seq_len]