import torch

class CharTokenizer:
    def __init__(self, text=None, pad_token='<PAD>', unk_token='<UNK>'):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_built = False

        if text:
            self.build_vocab(text)

    def build_vocab(self, text):
        unique_chars = sorted(set(text))
        # Reserve indices for PAD and UNK
        self.char2idx = {self.pad_token: 0, self.unk_token: 1}
        for i, ch in enumerate(unique_chars, start=2):
            self.char2idx[ch] = i
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_built = True

    def encode(self, text, max_length=None):
        if not self.vocab_built:
            raise ValueError("Vocabulary not built yet. Call build_vocab first.")

        encoded = [self.char2idx.get(ch, self.char2idx[self.unk_token]) for ch in text]

        # Truncate if needed
        if max_length is not None:
            encoded = encoded[:max_length]

        # Pad if needed
        if max_length is not None and len(encoded) < max_length:
            pad_length = max_length - len(encoded)
            encoded += [self.char2idx[self.pad_token]] * pad_length

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, indices):
        # Accept either tensor or list
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        chars = [self.idx2char.get(i, self.unk_token) for i in indices]
        # Strip padding tokens from end
        while chars and chars[-1] == self.pad_token:
            chars.pop()
        return ''.join(chars)
    
    def create_mask(self, encoded_tensor):
        pad_token_idx = self.char2idx[self.pad_token]
        return (encoded_tensor != pad_token_idx).long()
