"""
python -m unittest rope_embeddings_test.basicTest.twobytwo_test
"""

import numpy as np
import torch
import unittest
from rope_embeddings import RopeEmbeddings


class basicTest(unittest.TestCase):

	def twobytwo_test(self):

		approximation = [[-1.142639, 1.9220756],[-4.8856302, 1.0633049]]
		approximation = np.array(approximation)
		approximation = approximation[:,:,None]

		rope = RopeEmbeddings(2,2,1)
		X = torch.Tensor([[1,2],[3,4]])
		X = X[:,:,None]
		X = rope(X)
		X = X.detach().numpy()
		np.testing.assert_allclose(approximation,X,rtol=0.05,atol=0.01)



