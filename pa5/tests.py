import student
import numpy as np

def skip_not_implemented(func):
    from nose.plugins.skip import SkipTest

    def wrapper():
        try:
            func()
        except NotImplementedError, exc:
            raise SkipTest(
                "Test {0} is skipped {1}".format(func.__name__, exc))
    wrapper.__name__ = func.__name__
    return wrapper

@skip_not_implemented
def test_labels_to_one_hot():
	labels = np.array([[0],[1],[2]])
	num_classes = 4
	returned = student.labels_to_one_hot(labels, num_classes)
	expected = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
	assert np.array_equal(returned, expected)

@skip_not_implemented	
def test_split_dataset():
	train_size = 90
	val_size = 10
	test_size = 20
	x = np.random.rand(200,32,32,3)
	y = np.random.rand(200,10)
	x_t = np.random.rand(50,16,16,3)
	y_t = np.random.rand(50,10)
	x_train, y_train, x_val, y_val, x_test, y_test = student.split_dataset(x, y, x_t, y_t, train_size, val_size, test_size)
	assert x_train.shape == (train_size,32,32,3)
	assert y_train.shape == (train_size,10)
	assert x_val.shape == (val_size,32,32,3)
	assert y_val.shape == (val_size,10)
	assert x_test.shape == (test_size,16,16,3)
	assert y_test.shape == (test_size,10)
    print ("test is running")

@skip_not_implemented
def test_preprocess_dataset():
	x = np.random.rand(5,32,32,3)
	image_size = 16
	x_p = student.preprocess_dataset(x,image_size)
	assert x_p.shape == (5,image_size,image_size,3)
	for c in range(3):
		np.testing.assert_almost_equal(np.mean(x_p[:,:,:,c]), 0)
		np.testing.assert_almost_equal(np.std(x_p[:,:,:,c]), 1)

@skip_not_implemented
def test_get_N_cifar_images():
	L = 0
	N = 1
	images = np.random.rand(3,32,32,3)
	labels = np.array([[0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0]])
	class_string, query_images = student.get_N_cifar_images(N, L, images, labels)
	assert class_string == "airplane"
	assert query_images.shape == (1,32,32,3)
	assert np.array_equal(query_images[0], images[1])

@skip_not_implemented
def test_change_labels_av():
	y_train = np.array([[0,0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0]])
	y_val = np.array([[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,0,0]])
	y_test = np.array([[0,0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0]])
	y_train_av, y_val_av, y_test_av = student.change_labels_av(y_train, y_val, y_test)
	y_train_av_expected = np.array([[1,0],[0,1],[1,0]])
	y_val_av_expected = np.array([[0,1],[1,0],[1,0]])
	y_test_av_expected = np.array([[1,0],[1,0],[1,0]])
	assert np.array_equal(y_train_av, y_train_av_expected)
	assert np.array_equal(y_val_av, y_val_av_expected)
	assert np.array_equal(y_test_av, y_test_av_expected)

def mock_get_gradients(x):
	assert len(x) == 1
	assert x[0].ndim == 4
	return [np.mean(x), np.full(x[0].shape, 0.1)]

@skip_not_implemented
def test_generate_adversarial():
	image = np.zeros((32, 32, 3))
	get_gradients = mock_get_gradients
	alpha = 1
	num_steps = 10
	changed_image, diff_image = student.generate_adversarial(image, get_gradients, alpha, num_steps)
	changed_image_expected = np.ones((32,32,3))
	diff_image_expected = np.full((32,32), np.sqrt(3))
	assert np.allclose(changed_image, changed_image_expected)
	assert np.allclose(diff_image, diff_image_expected)

