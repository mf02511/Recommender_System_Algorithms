import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    embed_dim=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout=args.dropout_rate,
                )
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs):
        # Ensure input is on the correct device and dtype
        seqs = self.item_emb(log_seqs.to(self.dev).long())
        seqs *= self.item_emb.embedding_dim ** 0.5
    
        # Positional encoding
        poss = torch.arange(1, log_seqs.size(1) + 1, device=self.dev).unsqueeze(0)
        poss = poss.repeat(log_seqs.size(0), 1) * (log_seqs != 0).int()
    
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
    
        # Enforce causality with attention mask
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
    
        # Apply attention and feedforward layers
        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = seqs.transpose(0, 1)
    
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
    
        # Final layer normalization
        log_feats = self.last_layernorm(seqs)
    
        return log_feats



    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """
        Forward pass to compute positive and negative logits.
        Args:
            user_ids: IDs of users (not used in computation here but might be useful for extensions).
            log_seqs: Logged sequences of items.
            pos_seqs: Positive sequences (target items).
            neg_seqs: Negative sequences (negative samples).
    
        Returns:
            pos_logits: Logits for positive sequences.
            neg_logits: Logits for negative sequences.
        """
        log_feats = self.log2feats(log_seqs)
    
        # Embedding lookup
        pos_embs = self.item_emb(pos_seqs.to(self.dev).long())
        neg_embs = self.item_emb(neg_seqs.to(self.dev).long())
    
        # Compute logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
    
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        Prediction for a batch of items.
        Args:
            user_ids: IDs of users (not used in computation here but might be useful for extensions).
            log_seqs: Logged sequences of items (NumPy array or tensor).
            item_indices: Items to compute predictions for.
    
        Returns:
            logits: Predicted logits for the given item indices.
        """
        # Convert log_seqs to a PyTorch tensor if it is not already one
        if isinstance(log_seqs, np.ndarray):
            log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.dev)
    
        log_feats = self.log2feats(log_seqs)
    
        # Use the final feature for prediction
        final_feat = log_feats[:, -1, :]  # (batch_size, embedding_dim)
    
        # Embedding lookup for items
        if isinstance(item_indices, np.ndarray):
            item_indices = torch.tensor(item_indices, dtype=torch.long, device=self.dev)
    
        item_embs = self.item_emb(item_indices)  # (num_items, embedding_dim)
    
        # Compute logits
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_items)
    
        return logits

