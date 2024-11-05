"""
python -m unittest rope_embeddings_tests.basicTest.twobytwo_test
"""

import numpy as np
import torch
import unittest
from rope_embedding import RopeEmbedding


class basicTest(unittest.TestCase):

	def twobytwo_test(self):

		approximation = [[-2.1432636, 0.637511397],[-1.450997289, -4.7848309]]
		approximation = torch.Tensor(approximation)
		rope = RopeEmbedding(2,2,1)
		X = torch.Tensor([[1,2],[3,4]])
		X = rope(X)
		np.testing.assert_allclose(appromixation,X,rtol=0.05,atol=0.01)



