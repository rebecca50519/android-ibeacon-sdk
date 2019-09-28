import pickle
def test(b1,b2,b3):

	decision_tree_pkl_filename = 'decision_tree_classifier_20170212.pkl'



	# Loading the saved decision tree model pickle
	decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
	decision_tree_model = pickle.load(decision_tree_model_pkl)

	clfX = decision_tree_model.predict([[b1, b2, b3]])
	return clfX
