import torch
import torch.nn as nn


class Encoder(nn.Module):
	def __init__(self, config):
		super(Encoder, self).__init__()

	def forward(self, x, e_mask):
		return


class Deocder(nn.Module):
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
	def forward(self, trg, memory, e_mask, d_mask):
		return


class RecursiveTransformer(nn.Module):
	def __init__(self, config):
		super(RecursiveTransformer, self).__init__()
		self.encoder = Encoder(config)
		self.decoder = Decoder(config)

	def pad_mask(self, x):
		return

	def dec_mask(self, x):
		return

	def forward(self, src, trg):
		e_mask, d_mask = self.pad_mask(src), self.dec_mask(trg)
		memory = self.encoder(src, e_mask)

		return self.decoder(trg, memory, e_mask, d_mask)		