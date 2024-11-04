import torch
import torch.nn as nn

_CONST = 10_000

class RopeEmbeddings(nn.Module):

	def __init__(self, embedding_dim, max_seq_len, batch_size):

		self.D = embedding_dim  # even number
		self.L = max_seq_len
		self.N = batch_size


	def forward(X: torch.Tensor) -> torch.Tensor:
		# assume X is (L x D x N)
		for i in range(self.D // 2):
			# create a new (2L,1) tensor, braided rope
			tensor_one = X[:,i,:]
			tensor_two = X[:,i+(self.D//2),:]
			stacked = torch.stack([tensor_one, tensor_two], dim=1)
			interleaved = torch.flatten(stacked, start_dim=0, end_dim=1)
			# check the above
			# interleaved should be (2L x 1 x N)
			# can flatten to (2L x N)

			# create block diagonal matrix for theta_i
			# calcualte theta_i
			theta_i = _CONST**(2*i/self.D)
			matrix_list = []
			for k in range(self.L):
				# create blocks list
				matrix_list.append(torch.Tensor([[torch.cos(k*theta_i), -torch.sin(k*theta_i)][torch.sin(k*theta_i), torch.cos(k*theta_i)]]))

			rotation_matrix = torch.block_diag(matrix_list)
			rotated_interleaved = torch.matmul(rotation_matrix, interleaved)
			# should be of shape 2L x N

			column_one = rotation_interleaved[::2,:]   # L x N
			column_two = rotation_interleaved[1::2,:]  # L x N

			X[:,i,:] = column_one
			X[:,i+(self.D//2),:] = column_two

		return X
