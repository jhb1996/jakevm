import numpy as np
from six.moves import cPickle as pkl
import unittest
import warnings
import segment
warnings.filterwarnings("ignore")  # ignore instability warnings in Kmeans.

DATA_FILE = 'release_data.pkl'

EPSILON = 1e-100

class TestSegment(unittest.TestCase):
    ''' dummy class for fitting to unittest framework
    real test cases are added dynamically'''
    def setUp(self):
        with open(DATA_FILE, 'rb') as f:
            self.data = pkl.load(f)

    def test_normalize(self):
        img, (params, exp_out) = self.data['A_norm']
        student_out = segment.normalizeImage(img, *params)
        self.assertTrue(np.allclose(exp_out, student_out, atol=EPSILON))
    
    def test_grad(self):
        img, xgrad_exp, ygrad_exp, maggrad_exp = self.data['A_grad']
        xgrad, ygrad, maggrad = segment.takeXGradient(img),segment.takeYGradient(img),\
                segment.takeGradientMag(img)
        print (xgrad)
        print (xgrad_exp)
        self.assertTrue(np.allclose(xgrad_exp, xgrad, atol=EPSILON))
        self.assertTrue(np.allclose(ygrad_exp, ygrad, atol=EPSILON))
        self.assertTrue(np.allclose(maggrad_exp, maggrad, atol=EPSILON))
        
        # img, xgrad_exp, ygrad_exp, maggrad_exp = self.data['A_grad']
        # xgrad = segment.takeXGradient(img)#,segment.takeYGradient(img),\
        # 
        # #print (xgrad[0])
        # #print (xgrad_exp[0])
        #         #segment.takeGradientMag(img)
        # self.assertTrue(np.allclose(xgrad_exp, xgrad, atol=EPSILON))
    
    
    def test_kmeans_solve(self):
        centers, k, img, exp_niter = self.data['B_ksolver']
        exp_labels = img[:,-1]
        img[:, -1] = -1
        res_niter = segment.kMeansSolver(img, k, centers = centers)
        res_labels = img[:,-1]
        self.assertTrue(np.allclose(exp_labels, res_labels, atol=EPSILON))
    
    def test_ncut_node_weight(self):
        W, exp_out = self.data['C']['Ws'], self.data['C']['ds']
        res_out = segment.getTotalNodeWeights(W)
        self.assertTrue(np.allclose(exp_out, res_out, atol=EPSILON))
    
    def test_ncut_color_weight(self):
        img, r, exp_out = self.data['C']['imgs'], self.data['C']['rs'], self.data['C']['Ws']
        res_out = segment.getColorWeights(img, r)
        self.assertTrue(np.allclose(exp_out, res_out, atol=EPSILON))
    
    def test_ncut_approx_norm_bisect(self):
        W, d, exp_out = self.data['C']['Ws'], self.data['C']['ds'], self.data['C']['ys']
        res_out = segment.approxNormalizedBisect(W,d)
        self.assertTrue(np.allclose(exp_out, res_out, atol=EPSILON))
    
    def test_ncut_reconstruct(self):
        img, y, exp_out = self.data['C']['imgs'], self.data['C']['ys'], self.data['C']['outs']
        res_out = segment.reconstructNCutSegments(img, y)
        self.assertTrue(np.allclose(exp_out, res_out, atol=EPSILON))

if __name__ == '__main__':
    unittest.main()
